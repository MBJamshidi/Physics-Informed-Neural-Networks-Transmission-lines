"""
src/data_generation.py
======================
Generates synthetic temperature data for a power cable using an explicit
finite-difference method (FDM) to solve the 1D heat equation:

    ∂T/∂t = α ∂²T/∂x² + Q(x, t)

Physical interpretation
-----------------------
* The cable is modelled as a 1-D conductor of normalised length L = 1.
* Both ends are held at ambient temperature (Dirichlet BCs) — this
  represents cooling at cable terminations / joints.
* The initial condition is uniform ambient temperature (cold start).
* The heat source Q(x, t) represents Joule (I²R) losses.  A sinusoidal
  current load causes heating that peaks near midspan and varies in time.

Numerical scheme
----------------
Explicit (forward-time, centred-space) FTCS scheme:

    T_i^{n+1} = T_i^n + r*(T_{i+1}^n - 2*T_i^n + T_{i-1}^n) + Δt*Q_i^n

where r = α*Δt/Δx² is the Fourier number.
Stability requires r ≤ 0.5.
"""

import numpy as np
import os
import sys

# Allow imports from the project root when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Heat-source definition
# ─────────────────────────────────────────────────────────────────────────────

def heat_source(x: np.ndarray, t: float) -> np.ndarray:
    """
    Joule-heating source term Q(x, t)  [°C / normalised-time].

    Physical model:
        Q = Q_peak * sin(π x / L) * (0.5 + 0.5 * sin(ω t))

    * sin(π x / L): spatial profile — maximum heating at cable midpoint,
      zero at terminations (where current enters/exits via thick bus bars
      that do not overheat).
    * (0.5 + 0.5 * sin(ω t)): temporal load cycle — load fluctuates
      sinusoidally (e.g., daily demand cycle).  The offset 0.5 ensures
      Q ≥ 0 at all times (physical: no negative heating).

    Parameters
    ----------
    x : array of spatial positions in [0, L_CABLE]
    t : scalar time value

    Returns
    -------
    Q : array of same shape as x
    """
    spatial_profile  = np.sin(cfg.OMEGA_X * x / cfg.L_CABLE)
    temporal_profile = 0.5 + 0.5 * np.sin(cfg.OMEGA_T * t / cfg.T_END)
    return cfg.Q_PEAK * spatial_profile * temporal_profile


# ─────────────────────────────────────────────────────────────────────────────
# Finite-difference solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_heat_equation_fdm(nx: int = cfg.NX,
                             nt: int = cfg.NT) -> dict:
    """
    Solve the 1D heat equation with the explicit FTCS finite-difference scheme.

    Grid
    ----
    x_grid : nx points uniformly spaced in [0, L_CABLE]
    t_grid : nt points uniformly spaced in [0, T_END]

    Boundary conditions (Dirichlet)
    --------------------------------
    T(0, t) = T_AMBIENT   (left termination held at ambient)
    T(L, t) = T_AMBIENT   (right termination held at ambient)

    Initial condition
    -----------------
    T(x, 0) = T_AMBIENT   (cable starts at ambient temperature)

    Returns
    -------
    dict with keys:
        'x'      : (nx,)    spatial grid [m, normalised]
        't'      : (nt,)    temporal grid [s, normalised]
        'T'      : (nt, nx) temperature field  [°C]
        'Q'      : (nt, nx) heat source field  [°C/t-unit]
        'dx'     : float    spatial step
        'dt'     : float    temporal step
        'r'      : float    Fourier number (stability indicator)
    """
    # ── Grid ──────────────────────────────────────────────────────────────
    dx = cfg.L_CABLE / (nx - 1)
    dt = cfg.T_END   / (nt - 1)
    x  = np.linspace(0.0, cfg.L_CABLE, nx)   # shape (nx,)
    t  = np.linspace(0.0, cfg.T_END,   nt)   # shape (nt,)

    # ── Stability check ────────────────────────────────────────────────────
    r = cfg.ALPHA * dt / dx**2
    if r > 0.5:
        raise ValueError(
            f"FDM stability violated: r = {r:.4f} > 0.5.  "
            f"Increase NT or decrease NX in config.py."
        )
    print(f"[FDM] Grid: {nx} × {nt},  Δx={dx:.5f},  Δt={dt:.6f},  r={r:.4f}  (stable ✓)")

    # ── Initialise temperature field ───────────────────────────────────────
    T = np.full((nt, nx), cfg.T_AMBIENT, dtype=np.float64)
    Q_field = np.zeros((nt, nx), dtype=np.float64)

    # Apply initial condition explicitly
    T[0, :] = cfg.T_AMBIENT   # IC: uniform ambient temperature

    # ── Time-stepping loop ─────────────────────────────────────────────────
    for n in range(nt - 1):
        t_n = t[n]

        # Evaluate heat source at current time
        Q_n = heat_source(x, t_n)
        Q_field[n, :] = Q_n

        # FTCS update for interior points i = 1 … nx-2
        # T_i^{n+1} = T_i^n + r*(T_{i+1}^n - 2*T_i^n + T_{i-1}^n) + dt*Q_i^n
        T[n+1, 1:-1] = (
            T[n, 1:-1]
            + r * (T[n, 2:] - 2.0 * T[n, 1:-1] + T[n, :-2])
            + dt * Q_n[1:-1]
        )

        # Enforce Dirichlet BCs at every time step
        T[n+1,  0] = cfg.T_AMBIENT   # left end
        T[n+1, -1] = cfg.T_AMBIENT   # right end

    # Store Q at the final time step
    Q_field[-1, :] = heat_source(x, t[-1])

    return {"x": x, "t": t, "T": T, "Q": Q_field, "dx": dx, "dt": dt, "r": r}


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(data: dict, path: str = cfg.DATA_FILE) -> None:
    """Persist the FDM solution and noisy observations to a compressed NPZ file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Add Gaussian noise to simulate sensor measurements
    rng = np.random.default_rng(cfg.SEED)
    T_noisy = data["T"] + rng.normal(0.0, cfg.NOISE_STD, size=data["T"].shape)

    np.savez_compressed(
        path,
        x       = data["x"],
        t       = data["t"],
        T_true  = data["T"],       # clean FDM solution
        T_noisy = T_noisy,         # noisy "sensor" readings
        Q       = data["Q"],
        dx      = np.array(data["dx"]),
        dt      = np.array(data["dt"]),
        r       = np.array(data["r"]),
    )
    print(f"[FDM] Dataset saved → {path}")


def load_dataset(path: str = cfg.DATA_FILE) -> dict:
    """Load a previously saved NPZ dataset."""
    raw = np.load(path)
    return {k: raw[k] for k in raw.files}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: run this module directly to regenerate data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(cfg.SEED)
    print("=" * 60)
    print("  Power Cable Thermal FDM Solver")
    print("=" * 60)
    data = solve_heat_equation_fdm()
    print(f"[FDM] T range: {data['T'].min():.2f} – {data['T'].max():.2f} °C")
    save_dataset(data)
    print("[FDM] Done.")
