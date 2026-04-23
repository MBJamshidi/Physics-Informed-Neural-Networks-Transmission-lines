"""
src/evaluate.py
===============
Quantitative evaluation of the trained PINN against the FDM reference
solution.

Metrics computed
----------------
* Mean Absolute Error (MAE)  [°C]
* Root Mean Squared Error (RMSE)  [°C]
* Maximum Absolute Error  [°C]
* Relative L² error  (dimensionless)

All metrics are computed:
  (a) over the full spatio-temporal domain
  (b) at the hottest spatial location (cable midpoint) as a function of t
  (c) at the final time snapshot t = T_END as a function of x
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from src.model import HeatPINN


# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: HeatPINN, data_dict: dict) -> dict:
    """
    Run a comprehensive evaluation.

    Parameters
    ----------
    model     : trained HeatPINN
    data_dict : dict from data_generation.load_dataset()

    Returns
    -------
    results : dict with metric values and prediction arrays
    """
    x_grid  = data_dict["x"]        # (nx,)
    t_grid  = data_dict["t"]        # (nt,)
    T_true  = data_dict["T_true"]   # (nt, nx)  — clean FDM solution

    # ── PINN predictions on the full FDM grid ─────────────────────────────
    T_pinn = model.predict_grid(x_grid, t_grid)   # (nt, nx)

    # ── Global error metrics ───────────────────────────────────────────────
    abs_err = np.abs(T_pinn - T_true)

    mae     = abs_err.mean()
    rmse    = np.sqrt(((T_pinn - T_true) ** 2).mean())
    max_err = abs_err.max()

    # Relative L² error
    rel_l2  = (np.linalg.norm(T_pinn - T_true) /
               np.linalg.norm(T_true - cfg.T_AMBIENT) + 1e-12)

    # ── Spatial profile at final time t = T_END ────────────────────────────
    T_true_final = T_true[-1, :]     # (nx,)
    T_pinn_final = T_pinn[-1, :]     # (nx,)

    # ── Temporal profile at midpoint x = L/2 ──────────────────────────────
    mid_idx = len(x_grid) // 2
    T_true_mid = T_true[:, mid_idx]   # (nt,)
    T_pinn_mid = T_pinn[:, mid_idx]   # (nt,)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(f"  Domain MAE        : {mae:.4f} °C")
    print(f"  Domain RMSE       : {rmse:.4f} °C")
    print(f"  Max absolute err  : {max_err:.4f} °C")
    print(f"  Relative L² err   : {rel_l2:.4f}")
    print(f"  T_true range      : [{T_true.min():.2f}, {T_true.max():.2f}] °C")
    print(f"  T_pinn range      : [{T_pinn.min():.2f}, {T_pinn.max():.2f}] °C")
    print("=" * 55)

    return {
        # Full fields
        "x_grid":       x_grid,
        "t_grid":       t_grid,
        "T_true":       T_true,
        "T_pinn":       T_pinn,
        "abs_error":    abs_err,
        # Global scalars
        "mae":          mae,
        "rmse":         rmse,
        "max_err":      max_err,
        "rel_l2":       rel_l2,
        # 1-D slices
        "T_true_final": T_true_final,
        "T_pinn_final": T_pinn_final,
        "T_true_mid":   T_true_mid,
        "T_pinn_mid":   T_pinn_mid,
    }


def compute_pde_residual_on_grid(model: HeatPINN, data_dict: dict,
                                  subsample: int = 50) -> np.ndarray:
    """
    Compute the PDE residual |∂T/∂t − α ∂²T/∂x² − Q| on a coarse sub-grid.

    This visualises where the trained PINN satisfies the physics.

    Parameters
    ----------
    subsample : keep every `subsample`-th grid point in each direction.

    Returns
    -------
    residual_field : (nt_sub, nx_sub) array
    x_sub, t_sub   : sub-sampled grid vectors
    """
    x_grid = data_dict["x"]
    t_grid = data_dict["t"]

    x_sub = x_grid[::subsample]
    t_sub = t_grid[::subsample]

    X, T_m = np.meshgrid(x_sub, t_sub)
    x_flat = X.ravel().astype(np.float32)
    t_flat = T_m.ravel().astype(np.float32)

    x_ten = torch.tensor(x_flat, requires_grad=True).unsqueeze(1)
    t_ten = torch.tensor(t_flat, requires_grad=True).unsqueeze(1)

    model.eval()
    T_pred = model(x_ten, t_ten)

    # ∂T/∂t
    dT_dt = torch.autograd.grad(
        T_pred, t_ten, grad_outputs=torch.ones_like(T_pred),
        create_graph=True, retain_graph=True)[0]
    # ∂T/∂x
    dT_dx = torch.autograd.grad(
        T_pred, x_ten, grad_outputs=torch.ones_like(T_pred),
        create_graph=True, retain_graph=True)[0]
    # ∂²T/∂x²
    d2T_dx2 = torch.autograd.grad(
        dT_dx, x_ten, grad_outputs=torch.ones_like(dT_dx),
        create_graph=False, retain_graph=False)[0]

    import torch
    Q = cfg.Q_PEAK * torch.sin(
        cfg.OMEGA_X * x_ten / cfg.L_CABLE
    ) * (0.5 + 0.5 * torch.sin(cfg.OMEGA_T * t_ten / cfg.T_END))

    residual = (dT_dt - cfg.ALPHA * d2T_dx2 - Q).detach().numpy().flatten()

    return np.abs(residual).reshape(len(t_sub), len(x_sub)), x_sub, t_sub
