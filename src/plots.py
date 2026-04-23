"""
src/plots.py
============
Comprehensive visualisation suite for the Power Cable Thermal PINN.

Figures generated
-----------------
1. fdm_solution.png       — FDM temperature field T(x, t) as a heatmap
2. pinn_vs_fdm.png        — Side-by-side: FDM | PINN | absolute error
3. spatial_profile.png    — T(x) at t = T_END for FDM vs PINN
4. temporal_profile.png   — T(t) at x = L/2 for FDM vs PINN
5. loss_curves.png        — Training loss (total + components) vs epoch
6. heat_source.png        — Q(x, t) field used in the simulation
7. pde_residual.png       — |PDE residual| on the evaluation grid
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works on any OS)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

# Global style
plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "figure.dpi":      120,
    "savefig.bbox":    "tight",
    "savefig.dpi":     150,
})

os.makedirs(cfg.FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str):
    path = os.path.join(cfg.FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FDM reference solution
# ─────────────────────────────────────────────────────────────────────────────

def plot_fdm_solution(data_dict: dict):
    x = data_dict["x"]
    t = data_dict["t"]
    T = data_dict["T_true"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(x, t, T, cmap="inferno", shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Temperature [°C]")
    ax.set_xlabel("Position x  [normalised]")
    ax.set_ylabel("Time t  [normalised]")
    ax.set_title("FDM Reference Solution  T(x, t)\n"
                 "Power Cable — 1-D Heat Equation")
    _save(fig, "fdm_solution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PINN vs FDM comparison (3-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pinn_vs_fdm(results: dict):
    x       = results["x_grid"]
    t       = results["t_grid"]
    T_true  = results["T_true"]
    T_pinn  = results["T_pinn"]
    abs_err = results["abs_error"]

    vmin = min(T_true.min(), T_pinn.min())
    vmax = max(T_true.max(), T_pinn.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: FDM
    im0 = axes[0].pcolormesh(x, t, T_true, cmap="inferno",
                               vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(im0, ax=axes[0]).set_label("°C")
    axes[0].set_title("FDM (Ground Truth)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

    # Panel 2: PINN
    im1 = axes[1].pcolormesh(x, t, T_pinn, cmap="inferno",
                               vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(im1, ax=axes[1]).set_label("°C")
    axes[1].set_title("PINN Prediction")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

    # Panel 3: Absolute error
    im2 = axes[2].pcolormesh(x, t, abs_err, cmap="RdYlGn_r",
                               norm=mcolors.LogNorm(
                                   vmin=max(abs_err.min(), 1e-4),
                                   vmax=abs_err.max()),
                               shading="auto")
    fig.colorbar(im2, ax=axes[2]).set_label("°C  (log scale)")
    axes[2].set_title(f"Absolute Error\n"
                       f"MAE={results['mae']:.3f}°C, "
                       f"MaxErr={results['max_err']:.3f}°C")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("t")

    fig.suptitle("Physics-Informed Neural Network vs Finite-Difference Solution",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "pinn_vs_fdm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial profile at t = T_END
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_profile(results: dict):
    x = results["x_grid"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, results["T_true_final"], "b-",  lw=2,   label="FDM (reference)")
    ax.plot(x, results["T_pinn_final"], "r--", lw=2,   label="PINN")
    ax.fill_between(x,
                    results["T_true_final"] - results["abs_error"][-1, :],
                    results["T_true_final"] + results["abs_error"][-1, :],
                    alpha=0.2, color="red", label="±error band")
    ax.set_xlabel("Position x  [normalised]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Spatial Temperature Profile at t = T_END\n"
                 "(final time snapshot)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "spatial_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temporal profile at x = L/2
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_profile(results: dict):
    t = results["t_grid"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, results["T_true_mid"], "b-",  lw=2,  label="FDM (reference)")
    ax.plot(t, results["T_pinn_mid"], "r--", lw=2,  label="PINN")
    ax.set_xlabel("Time t  [normalised]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Temperature vs Time at Cable Midpoint (x = L/2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "temporal_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict):
    epochs = np.arange(1, len(history["total"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: total loss on log scale
    axes[0].semilogy(epochs, history["total"], "k-", lw=1.5, label="Total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Total Training Loss")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # Right: individual components
    colours = {"data": "blue", "physics": "green", "bc": "orange", "ic": "red"}
    labels  = {"data": "L_data", "physics": "L_physics", "bc": "L_bc", "ic": "L_ic"}
    for key, colour in colours.items():
        axes[1].semilogy(epochs, history[key], color=colour,
                          lw=1.2, label=labels[key])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_title("Loss Components")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("PINN Training Loss History", fontsize=13, fontweight="bold")
    _save(fig, "loss_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Heat source field Q(x, t)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heat_source(data_dict: dict):
    x = data_dict["x"]
    t = data_dict["t"]
    Q = data_dict["Q"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(x, t, Q, cmap="hot", shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Q(x, t)  [°C / t-unit]")
    ax.set_xlabel("Position x  [normalised]")
    ax.set_ylabel("Time t  [normalised]")
    ax.set_title("Joule-Heating Source Term Q(x, t)\n"
                 "Represents I²R Losses in the Power Cable Conductor")
    _save(fig, "heat_source.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PDE residual field
# ─────────────────────────────────────────────────────────────────────────────

def plot_pde_residual(residual_field: np.ndarray,
                      x_sub:          np.ndarray,
                      t_sub:          np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(x_sub, t_sub, residual_field,
                        cmap="viridis", shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|PDE residual|  [°C / t-unit]")
    ax.set_xlabel("Position x  [normalised]")
    ax.set_ylabel("Time t  [normalised]")
    ax.set_title("|∂T/∂t − α ∂²T/∂x² − Q|  (PDE residual)\n"
                 "Lower = better physics satisfaction")
    _save(fig, "pde_residual.png")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(data_dict: dict, results: dict, history: dict,
                        residual_field: np.ndarray = None,
                        x_sub: np.ndarray = None,
                        t_sub: np.ndarray = None):
    print("\n[Plots] Generating all figures …")
    plot_fdm_solution(data_dict)
    plot_pinn_vs_fdm(results)
    plot_spatial_profile(results)
    plot_temporal_profile(results)
    plot_loss_curves(history)
    plot_heat_source(data_dict)
    if residual_field is not None:
        plot_pde_residual(residual_field, x_sub, t_sub)
    print(f"[Plots] All figures saved to {cfg.FIGURES_DIR}")
