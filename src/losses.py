"""
src/losses.py
=============
Loss functions for the Physics-Informed Neural Network.

Total loss:
-----------
    L = λ_data * L_data
      + λ_phys * L_phys
      + λ_bc   * L_bc
      + λ_ic   * L_ic

1. L_data  — MSE between PINN output and noisy measured temperature.
2. L_phys  — MSE of the PDE residual at collocation points.
3. L_bc    — MSE of the Dirichlet boundary condition violations.
4. L_ic    — MSE of the initial condition violations.

PDE residual (Physics loss)
----------------------------
The 1-D heat equation:
    ∂T/∂t = α ∂²T/∂x² + Q(x, t)

Residual form (should equal zero everywhere in the domain):
    f(x, t) = ∂T/∂t − α ∂²T/∂x² − Q(x, t)  = 0

Derivatives ∂T/∂t and ∂²T/∂x² are computed via PyTorch autograd,
treating x and t as leaf tensors with requires_grad=True.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Heat-source function (mirrors data_generation.py, but in PyTorch tensors)
# ─────────────────────────────────────────────────────────────────────────────

def heat_source_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Pytorch version of Q(x, t) — Joule heating.

    Q(x, t) = Q_peak * sin(π x / L) * (0.5 + 0.5 * sin(ω t / T_end))

    Uses the same formula as data_generation.heat_source so that the
    physics residual is consistent with the data-generating process.
    """
    spatial_profile  = torch.sin(cfg.OMEGA_X * x / cfg.L_CABLE)
    temporal_profile = 0.5 + 0.5 * torch.sin(cfg.OMEGA_T * t / cfg.T_END)
    return cfg.Q_PEAK * spatial_profile * temporal_profile


# ─────────────────────────────────────────────────────────────────────────────
# Autograd helper: first-order derivative
# ─────────────────────────────────────────────────────────────────────────────

def grad(output: torch.Tensor,
         input_:  torch.Tensor,
         create_graph: bool = True) -> torch.Tensor:
    """
    Compute ∂output/∂input_ using torch.autograd.grad.

    Parameters
    ----------
    output       : scalar or tensor (the function to differentiate)
    input_       : tensor w.r.t. which we differentiate (must have requires_grad=True)
    create_graph : keep computational graph for higher-order derivatives

    Returns
    -------
    Gradient tensor of same shape as input_.
    """
    return torch.autograd.grad(
        outputs=output,
        inputs=input_,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ─────────────────────────────────────────────────────────────────────────────

def loss_data(model, x_d: torch.Tensor, t_d: torch.Tensor,
              T_d: torch.Tensor) -> torch.Tensor:
    """
    L_data = (1/N) Σ (T_θ(x_i, t_i) − T_measured_i)²

    Fits the PINN to the noisy sensor observations.
    This term anchors the solution to measured reality.
    """
    T_pred = model(x_d, t_d)
    return torch.mean((T_pred - T_d) ** 2)


def loss_physics(model, x_col: torch.Tensor,
                 t_col: torch.Tensor) -> torch.Tensor:
    """
    L_phys = (1/N) Σ f(x_i, t_i)²

    where the PDE residual is:
        f = ∂T/∂t − α ∂²T/∂x² − Q(x, t)

    Steps:
    1. Forward pass to get T = T_θ(x, t).
    2. Compute ∂T/∂t via autograd.
    3. Compute ∂T/∂x via autograd.
    4. Compute ∂²T/∂x² by differentiating ∂T/∂x w.r.t. x again.
    5. Evaluate heat source Q(x, t).
    6. Form residual and return its MSE.
    """
    # Step 1: forward pass
    # x_col and t_col must have requires_grad=True (set in dataset.py)
    T_pred = model(x_col, t_col)   # (N, 1)

    # Step 2: ∂T/∂t
    dT_dt = grad(T_pred, t_col)    # (N, 1)

    # Step 3: ∂T/∂x  (first spatial derivative)
    dT_dx = grad(T_pred, x_col)    # (N, 1)

    # Step 4: ∂²T/∂x²  (second spatial derivative via chain rule)
    d2T_dx2 = grad(dT_dx, x_col)  # (N, 1)

    # Step 5: heat source Q(x, t) at collocation points
    Q = heat_source_torch(x_col, t_col)   # (N, 1)

    # Step 6: PDE residual   f = ∂T/∂t − α ∂²T/∂x² − Q
    residual = dT_dt - cfg.ALPHA * d2T_dx2 - Q

    return torch.mean(residual ** 2)


def loss_bc(model,
            bc_left:  tuple,
            bc_right: tuple) -> torch.Tensor:
    """
    L_bc = MSE at left boundary + MSE at right boundary

    Dirichlet BCs:
        T(0, t) = T_AMBIENT
        T(L, t) = T_AMBIENT

    Both cable terminations are held at ambient temperature (cooling
    by the cable connectors / heat sinks at the joints).
    """
    x_l, t_l, T_l_target = bc_left
    x_r, t_r, T_r_target = bc_right

    T_pred_left  = model(x_l, t_l)
    T_pred_right = model(x_r, t_r)

    mse_left  = torch.mean((T_pred_left  - T_l_target) ** 2)
    mse_right = torch.mean((T_pred_right - T_r_target) ** 2)

    return mse_left + mse_right


def loss_ic(model,
            x_ic:  torch.Tensor,
            t_ic:  torch.Tensor,
            T_ic:  torch.Tensor) -> torch.Tensor:
    """
    L_ic = (1/N) Σ (T_θ(x_i, 0) − T_AMBIENT)²

    Initial condition:
        T(x, 0) = T_AMBIENT   (cable at ambient temperature before energisation)
    """
    T_pred = model(x_ic, t_ic)
    return torch.mean((T_pred - T_ic) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

def total_loss(model, training_data: dict) -> tuple:
    """
    Compute and return the weighted total loss and its components.

    L = λ_data * L_data
      + λ_phys * L_phys
      + λ_bc   * L_bc
      + λ_ic   * L_ic

    Returns
    -------
    total  : scalar tensor (differentiable, used for .backward())
    components : dict of float values for logging
    """
    # ── Data loss ──────────────────────────────────────────────────────────
    L_d = loss_data(
        model,
        training_data["x_data"],
        training_data["t_data"],
        training_data["T_data"],
    )

    # ── Physics (PDE) loss ─────────────────────────────────────────────────
    L_p = loss_physics(
        model,
        training_data["x_col"],
        training_data["t_col"],
    )

    # ── Boundary condition loss ────────────────────────────────────────────
    L_bc = loss_bc(
        model,
        training_data["bc_left"],
        training_data["bc_right"],
    )

    # ── Initial condition loss ─────────────────────────────────────────────
    L_ic = loss_ic(
        model,
        training_data["x_ic"],
        training_data["t_ic"],
        training_data["T_ic"],
    )

    # ── Weighted sum ───────────────────────────────────────────────────────
    total = (
        cfg.LAMBDA_DATA  * L_d
        + cfg.LAMBDA_PHYS * L_p
        + cfg.LAMBDA_BC   * L_bc
        + cfg.LAMBDA_IC   * L_ic
    )

    components = {
        "total":   total.item(),
        "data":    L_d.item(),
        "physics": L_p.item(),
        "bc":      L_bc.item(),
        "ic":      L_ic.item(),
    }

    return total, components
