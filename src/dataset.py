"""
src/dataset.py
==============
PyTorch dataset wrappers and collocation-point samplers for the PINN.

Three kinds of training points are used:

1. **Data points** — (x, t, T_measured) from the noisy FDM solution.
   These ground the PINN to reality.

2. **Collocation points** — randomly sampled (x, t) inside the domain.
   The PDE residual is evaluated here (no temperature label needed).

3. **Boundary / initial condition points** — (x=0 or L, t) and (x, t=0).
   Used to enforce BCs/IC.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a tensor that requires grad for autograd differentiation
# ─────────────────────────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    """Convert a numpy array to a float32 CUDA/CPU tensor."""
    t = torch.tensor(arr, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: observed data points
# ─────────────────────────────────────────────────────────────────────────────

class ThermalDataset(Dataset):
    """
    Randomly sub-samples N_DATA_POINTS (x, t, T) triples from the FDM grid.

    The noisy temperature is used as the target so the network learns
    to de-noise while fitting the physics.
    """

    def __init__(self, data_dict: dict, n_points: int = cfg.N_DATA_POINTS):
        x_grid = data_dict["x"]          # (nx,)
        t_grid = data_dict["t"]          # (nt,)
        T_noisy = data_dict["T_noisy"]   # (nt, nx)

        nx = len(x_grid)
        nt = len(t_grid)

        # Build full meshgrid
        X, T_mesh = np.meshgrid(x_grid, t_grid)    # both (nt, nx)
        X_flat = X.ravel()                          # (nt*nx,)
        T_flat = T_mesh.ravel()
        Temp_flat = T_noisy.ravel()

        # Random sub-sample
        rng = np.random.default_rng(cfg.SEED)
        idx = rng.choice(len(X_flat), size=n_points, replace=False)

        self.x_data   = X_flat[idx].reshape(-1, 1).astype(np.float32)
        self.t_data   = T_flat[idx].reshape(-1, 1).astype(np.float32)
        self.T_target = Temp_flat[idx].reshape(-1, 1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_data[idx]),
            torch.from_numpy(self.t_data[idx]),
            torch.from_numpy(self.T_target[idx]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: collocation points for PDE residual
# ─────────────────────────────────────────────────────────────────────────────

def sample_collocation_points(n: int = cfg.N_COLLOCATION,
                               seed: int = cfg.SEED) -> tuple:
    """
    Uniformly sample (x, t) pairs inside the spatio-temporal domain:
        x ∈ [0, L_CABLE],  t ∈ [0, T_END]

    Returns tensors that require gradients so that PyTorch can differentiate
    the network output T(x,t) with respect to x and t (needed for PDE loss).

    Returns
    -------
    x_col, t_col : tensors of shape (n, 1), requires_grad=True
    """
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(0.0, cfg.L_CABLE, (n, 1)).astype(np.float32)
    t_np = rng.uniform(0.0, cfg.T_END,   (n, 1)).astype(np.float32)

    x_col = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    t_col = torch.tensor(t_np, dtype=torch.float32, requires_grad=True)
    return x_col, t_col


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3: boundary condition points
# ─────────────────────────────────────────────────────────────────────────────

def sample_bc_points(n_per_boundary: int = cfg.N_BC_POINTS,
                     seed: int = cfg.SEED) -> tuple:
    """
    Sample points on the two Dirichlet boundaries:
        Left  BC:  x = 0,        t ∈ [0, T_END]
        Right BC:  x = L_CABLE,  t ∈ [0, T_END]

    At both boundaries the temperature equals T_AMBIENT.

    Returns
    -------
    (x_left, t_left, T_left_target),
    (x_right, t_right, T_right_target)
    Each tensor has shape (n_per_boundary, 1).
    """
    rng = np.random.default_rng(seed + 1)

    # Left boundary x = 0
    t_left_np = rng.uniform(0.0, cfg.T_END, (n_per_boundary, 1)).astype(np.float32)
    x_left_np = np.zeros_like(t_left_np)
    T_bc_np   = np.full_like(t_left_np, cfg.T_AMBIENT)

    x_left   = torch.tensor(x_left_np, requires_grad=True)
    t_left   = torch.tensor(t_left_np, requires_grad=True)
    T_bc_left = torch.tensor(T_bc_np)

    # Right boundary x = L
    t_right_np = rng.uniform(0.0, cfg.T_END, (n_per_boundary, 1)).astype(np.float32)
    x_right_np = np.full_like(t_right_np, cfg.L_CABLE)

    x_right    = torch.tensor(x_right_np, requires_grad=True)
    t_right    = torch.tensor(t_right_np, requires_grad=True)
    T_bc_right = torch.tensor(T_bc_np.copy())

    return (x_left, t_left, T_bc_left), (x_right, t_right, T_bc_right)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 4: initial condition points
# ─────────────────────────────────────────────────────────────────────────────

def sample_ic_points(n: int = cfg.N_IC_POINTS,
                     seed: int = cfg.SEED) -> tuple:
    """
    Sample points on the initial time slice t = 0:
        x ∈ [0, L_CABLE],  t = 0

    At t = 0 the cable is at uniform ambient temperature.

    Returns
    -------
    x_ic, t_ic, T_ic_target  — each shape (n, 1)
    """
    rng = np.random.default_rng(seed + 2)
    x_np  = rng.uniform(0.0, cfg.L_CABLE, (n, 1)).astype(np.float32)
    t_np  = np.zeros((n, 1), dtype=np.float32)
    T_np  = np.full((n, 1), cfg.T_AMBIENT, dtype=np.float32)

    x_ic = torch.tensor(x_np, requires_grad=True)
    t_ic = torch.tensor(t_np, requires_grad=True)
    T_ic = torch.tensor(T_np)
    return x_ic, t_ic, T_ic


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build all training data at once
# ─────────────────────────────────────────────────────────────────────────────

def build_all_training_data(data_dict: dict) -> dict:
    """
    Assemble every category of training point into a single dictionary
    for easy use in the training loop.
    """
    dataset = ThermalDataset(data_dict)

    # Stack data tensors
    x_d = torch.from_numpy(dataset.x_data)
    t_d = torch.from_numpy(dataset.t_data)
    T_d = torch.from_numpy(dataset.T_target)

    # Collocation (PDE) points
    x_col, t_col = sample_collocation_points()

    # BC points
    bc_left, bc_right = sample_bc_points()

    # IC points
    x_ic, t_ic, T_ic = sample_ic_points()

    return {
        # Observed data
        "x_data":  x_d,
        "t_data":  t_d,
        "T_data":  T_d,
        # PDE collocation
        "x_col":   x_col,
        "t_col":   t_col,
        # Boundary conditions
        "bc_left":  bc_left,   # (x_left, t_left, T_target)
        "bc_right": bc_right,  # (x_right, t_right, T_target)
        # Initial condition
        "x_ic": x_ic,
        "t_ic": t_ic,
        "T_ic": T_ic,
    }
