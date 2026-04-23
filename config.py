"""
config.py
=========
Central configuration for the Power Cable Thermal PINN project.

Physical problem:
    1D heat equation along a power cable:
        ∂T/∂t = α ∂²T/∂x² + Q(x,t)

All physical quantities use SI units unless noted.
"""

import os

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────
# PHYSICAL PARAMETERS  (realistic underground XLPE cable)
# ─────────────────────────────────────────────────────────────

# Cable length [m] — 100-metre segment of a distribution cable
L_CABLE = 1.0          # normalised to 1.0 (represents 100 m after scaling)

# Total simulation time [s] — 1 hour of thermal transient
T_END   = 1.0          # normalised to 1.0 (represents 3600 s after scaling)

# Thermal diffusivity [m²/s] — typical XLPE insulation + copper composite
# Real value ~5e-7 m²/s; here we use a normalised equivalent so the
# PDE is numerically well-conditioned on the unit domain.
ALPHA = 0.01           # [normalised units]

# Ambient / boundary temperature [°C]
T_AMBIENT = 20.0

# Peak temperature rise above ambient due to Joule heating [°C]
# Models I²R losses in a loaded conductor (typical peak ~50 °C rise)
Q_PEAK = 50.0

# Heat-source angular frequency — current load cycles over the day
# Here 2π means one full sinusoidal load cycle over T_END
OMEGA_T = 2.0 * 3.14159265358979  # temporal frequency of load variation
OMEGA_X = 3.14159265358979        # spatial variation (hotspot near midpoint)

# Noise standard deviation added to synthetic data to simulate sensors [°C]
NOISE_STD = 0.5

# ─────────────────────────────────────────────────────────────
# FINITE-DIFFERENCE SOLVER SETTINGS  (data generation)
# ─────────────────────────────────────────────────────────────
NX  = 101    # spatial grid points  (Δx = L/100)
NT  = 1001   # temporal grid points (Δt = T_END/1000)

# ─────────────────────────────────────────────────────────────
# PINN / NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────
INPUT_DIM   = 2          # (x, t)
OUTPUT_DIM  = 1          # T(x, t)
HIDDEN_DIMS = [64, 64, 64, 64]   # four hidden layers, 64 neurons each
ACTIVATION  = "tanh"

# ─────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
LEARNING_RATE    = 1e-3
N_EPOCHS         = 8000          # total training epochs
LR_DECAY_STEP    = 2000          # halve LR every N epochs
LR_DECAY_GAMMA   = 0.5

# Collocation points for physics residual (sampled inside the domain)
N_COLLOCATION = 5000

# Data sub-sample: use this many randomly chosen (x,t) pairs from FDM
N_DATA_POINTS = 2000

# ─────────────────────────────────────────────────────────────
# LOSS WEIGHTS  (λ values in total loss)
# ─────────────────────────────────────────────────────────────
LAMBDA_DATA  = 1.0    # data fidelity
LAMBDA_PHYS  = 1.0    # PDE residual
LAMBDA_BC    = 10.0   # boundary conditions (strongly enforced)
LAMBDA_IC    = 10.0   # initial condition   (strongly enforced)

# BC / IC collocation counts
N_BC_POINTS  = 200    # per boundary (x=0 and x=L)
N_IC_POINTS  = 200    # along initial time slice t=0

# ─────────────────────────────────────────────────────────────
# LOGGING & CHECKPOINTING
# ─────────────────────────────────────────────────────────────
LOG_INTERVAL   = 100    # print loss every N epochs
SAVE_INTERVAL  = 2000   # save checkpoint every N epochs

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR   = os.path.join(OUTPUTS_DIR, "figures")
MODELS_DIR    = os.path.join(OUTPUTS_DIR, "models")
LOGS_DIR      = os.path.join(OUTPUTS_DIR, "logs")

DATA_FILE     = os.path.join(DATA_DIR, "thermal_data.npz")
MODEL_CKPT    = os.path.join(MODELS_DIR, "pinn_best.pt")
LOSS_LOG_FILE = os.path.join(LOGS_DIR,  "loss_history.npz")
