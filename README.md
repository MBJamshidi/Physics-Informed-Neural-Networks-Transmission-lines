# Physics-Informed Neural Networks for Power Cable Thermal Dynamics

> **Solve the 1-D heat equation governing temperature evolution in a loaded underground power cable — using a neural network that learns from both sensor data *and* physics.**

A complete, beginner-friendly implementation of a **Physics-Informed Neural Network (PINN)** applied to a real power-systems problem: predicting conductor temperature along a high-voltage underground cable under varying load.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [The Physical Problem](#the-physical-problem)
4. [What is a PINN?](#what-is-a-pinn)
5. [The Governing PDE](#the-governing-pde)
6. [Loss Function Explained](#loss-function-explained)
7. [Neural Network Architecture](#neural-network-architecture)
8. [Dataset Generation](#dataset-generation)
9. [Project Structure](#project-structure)
10. [Configuration Reference](#configuration-reference)
11. [Understanding the Outputs](#understanding-the-outputs)
12. [Why This Matters in Power Systems](#why-this-matters-in-power-systems)
13. [Physical Parameter Glossary](#physical-parameter-glossary)
14. [Frequently Asked Questions](#frequently-asked-questions)

---

## What This Project Does

This project trains a neural network to predict the temperature `T(x, t)` at every point `x` along a 100-metre power cable at every moment `t` during a 1-hour transient, given:

- A **physical model** (the 1-D heat equation with Joule heating)
- A small set of **noisy sensor observations**

The trained PINN produces a smooth, physics-consistent temperature field — even in regions with no sensors — because it is simultaneously constrained by the governing PDE.

**Key result:** The PINN matches the reference finite-difference solution with an error typically below 1 °C, using only 2,000 noisy data points out of a 101 × 1,001 grid.

---

## Quick Start (5 minutes)

### Requirements

- Python **3.9 – 3.12**
- No GPU required (CPU training in ~10–20 min)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MBJamshidi/Physics-Informed-Neural-Networks-Transmission-lines.git
cd Physics-Informed-Neural-Networks-Transmission-lines

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python main.py
```

That single command runs all 6 steps automatically:

| Step | What happens |
|------|-------------|
| 1 | Generates synthetic temperature data using a finite-difference solver |
| 2 | Builds training tensors (data, collocation, BC, IC points) |
| 3 | Instantiates the PINN (~17 k parameters) |
| 4 | Trains for 8,000 epochs with Adam + LR decay |
| 5 | Evaluates PINN accuracy vs. the reference FDM solution |
| 6 | Saves all figures, the best model checkpoint, and loss logs |

### Expected console output

```
╔══════════════════════════════════════════════╗
║   Power Cable Thermal PINN                   ║
╚══════════════════════════════════════════════╝

[Step 1/6]  Generating synthetic thermal data via FDM …
  Grid shape: (1001, 101)  (nt × nx)
  T range: [20.00, 67.43] °C

[Step 2/6]  Assembling PINN training tensors …
...
[Step 4/6]  Training …
  Epoch    0 | Total: 4.2136e+02 | Data: 1.32e+02 | Phys: 7.60e+01 | BC: 2.61e+01 | IC: 1.87e+01
  Epoch  100 | Total: 3.5721e+01 | Data: 8.41e+00 | Phys: 5.20e+00 | BC: 1.20e+00 | IC: 9.40e-01
  ...
══════════════════════════════════════════════════════════════
  Pipeline complete!
  MAE          : 0.2341 °C
  RMSE         : 0.3102 °C
  Max error    : 1.1847 °C
  Rel L² error : 0.0063
══════════════════════════════════════════════════════════════
```

### Run individual modules

```bash
# Re-generate the FDM dataset only
python src/data_generation.py

# Test the neural network forward pass (and autograd)
python src/model.py
```

---

## The Physical Problem

An underground power cable carrying current generates heat through **Joule (I²R) losses** in the conductor. This heat must diffuse outward through the cable insulation and surrounding soil to the ambient environment.

```
           Heat sinks (terminations)
     ←─────────────────────────────────→
     T = T_amb                   T = T_amb
     |                               |
  ───●───────────────────────────────●───
     |         100-m cable           |
     └── Joule heating Q(x, t) ──────┘
```

**The engineering question:** What is the maximum current this cable can safely carry without exceeding its thermal limit (e.g. 90 °C for XLPE insulation)?

To answer this, we need an accurate model of `T(x, t)` — which is exactly what the PINN provides.

---

## What is a PINN?

A **Physics-Informed Neural Network** is a neural network trained to satisfy three objectives simultaneously:

```
                    ┌─────────────────────────────────┐
                    │     Neural Network  T_θ(x, t)   │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
       Fit sensor data     Satisfy the PDE      Honour BC and IC
       (L_data)            (L_phys)             (L_bc, L_ic)
```

**Why not just use a regular neural network?**

A regular data-driven network needs measurements everywhere to interpolate accurately. In practice, sensors are sparse and noisy. The PDE residual term acts as a physics-based regulariser that constrains the solution in regions with no measurements — making the model physically consistent even far from sensors.

**Why not just solve the PDE numerically (FDM/FEM)?**

Numerical solvers need dense grids and struggle with noisy/incomplete data. PINNs can assimilate real sensor observations and naturally handle irregular measurement locations. They also produce a smooth, continuously differentiable solution — useful for gradient-based control.

---

## The Governing PDE

The **1-D heat diffusion equation** with a volumetric source term:

```
∂T/∂t = α · ∂²T/∂x² + Q(x, t)
```

| Symbol | Meaning | Unit |
|--------|---------|------|
| `T(x,t)` | Temperature along the cable | °C |
| `x` | Position along the cable axis | m (normalised → [0, 1]) |
| `t` | Time | s (normalised → [0, 1]) |
| `α` | Thermal diffusivity | m²/s (normalised = 0.01) |
| `Q(x,t)` | Volumetric heat source (Joule losses) | °C / time-unit |

### Heat source model

```
Q(x, t) = Q_peak · sin(π x / L) · [0.5 + 0.5 · sin(ω t / T_end)]
```

- **Spatial term** `sin(π x)`: heating peaks at the cable midpoint and falls to zero at the cooled terminations (thick copper bus-bars absorb heat at both ends).
- **Temporal term** `0.5 + 0.5 sin(ωt)`: sinusoidal approximation of a daily load cycle (always ≥ 0).

### Boundary conditions (Dirichlet)

```
T(0, t) = T_ambient    ∀ t          ← left termination
T(L, t) = T_ambient    ∀ t          ← right termination
```

### Initial condition

```
T(x, 0) = T_ambient    ∀ x          ← cable starts cold
```

---

## Loss Function Explained

The total loss combines four physics-grounded terms:

```
L_total = λ_data · L_data  +  λ_phys · L_phys  +  λ_bc · L_bc  +  λ_ic · L_ic
```

### Term 1 — Data fidelity `L_data`

```
L_data = (1/N_d) · Σ [T_θ(x_i, t_i) − T_sensor_i]²
```

Matches the PINN output to noisy sensor observations at `N_d = 2,000` randomly sampled points. This anchors the solution to measured reality.

### Term 2 — PDE residual `L_phys`

```
f(x, t) = ∂T/∂t − α · ∂²T/∂x² − Q(x, t)     (should be zero everywhere)

L_phys  = (1/N_c) · Σ f(x_j, t_j)²
```

Evaluated at `N_c = 5,000` **collocation points** randomly scattered throughout the domain. These are *not* sensor locations — they are free points where we enforce the physics.

The derivatives `∂T/∂t` and `∂²T/∂x²` are computed by **automatic differentiation** through the network (PyTorch `autograd`), giving exact partial derivatives — no finite-difference approximation.

### Term 3 — Boundary condition loss `L_bc`

```
L_bc = MSE[T_θ(0, t) − T_ambient] + MSE[T_θ(L, t) − T_ambient]
```

Soft enforcement of the Dirichlet BCs. The high weight `λ_bc = 10` ensures near-exact adherence.

### Term 4 — Initial condition loss `L_ic`

```
L_ic = (1/N_ic) · Σ [T_θ(x_k, 0) − T_ambient]²
```

Enforces the cold-start condition. High weight `λ_ic = 10` ensures the network correctly represents the initial state.

### Default weights

| Weight | Value | Why |
|--------|-------|-----|
| `λ_data` | 1.0 | Balances data fit with physics |
| `λ_phys` | 1.0 | Core physics enforcement |
| `λ_bc`   | 10.0 | Boundaries must be nearly exact |
| `λ_ic`   | 10.0 | Initial state must be nearly exact |

---

## Neural Network Architecture

```
Input (2)  ──▶  [Linear 2→64, Tanh]  ──▶  [Linear 64→64, Tanh]
           ──▶  [Linear 64→64, Tanh]  ──▶  [Linear 64→64, Tanh]  ──▶  Output (1)
```

| Property | Value |
|----------|-------|
| Inputs | `x` (position), `t` (time) |
| Hidden layers | 4 layers × 64 neurons |
| Activation | **Tanh** |
| Output | `T(x, t)` — temperature [°C] |
| Initialisation | Xavier (Glorot) normal |
| Trainable parameters | ~16,961 |
| Optimizer | Adam, lr = 1e-3 |
| LR schedule | Halved every 2,000 epochs |

**Why Tanh and not ReLU?**

The PDE loss requires computing `∂²T/∂x²` through the network. ReLU's second derivative is zero almost everywhere, making the physics loss meaningless. Tanh is smooth (infinitely differentiable), so autograd correctly propagates through second-order derivatives.

---

## Dataset Generation

Since real sensor data is not available, a synthetic dataset is generated using the **Forward-Time Centred-Space (FTCS)** finite-difference method — a classical, well-validated numerical solver.

```
T_i^{n+1} = T_i^n + r · (T_{i+1}^n − 2·T_i^n + T_{i-1}^n) + Δt · Q_i^n

r = α·Δt/Δx²   (Fourier number — must be ≤ 0.5 for numerical stability)
```

**Grid:** 101 spatial points × 1,001 time steps.

Gaussian noise (σ = 0.5 °C) is added to simulate realistic sensor readings.

The dataset is saved as `data/thermal_data.npz` and contains:
- `x`, `t` — grid coordinate vectors
- `T_true` — clean FDM solution (ground truth)
- `T_noisy` — sensor-noisy version (used for training)
- `Q` — heat source field

---

## Project Structure

```
PowerSystemPINN_PDE/
│
├── README.md               ← You are here
├── requirements.txt        ← Python dependencies
├── main.py                 ← Full pipeline entry point (run this)
├── config.py               ← All parameters (physics + training)
│
├── src/
│   ├── __init__.py
│   ├── data_generation.py  ← FDM solver + heat source Q(x,t)
│   ├── dataset.py          ← PyTorch Datasets + collocation samplers
│   ├── model.py            ← PINN neural network (HeatPINN)
│   ├── losses.py           ← Data / physics / BC / IC loss terms
│   ├── train.py            ← Training loop, LR schedule, checkpointing
│   ├── evaluate.py         ← MAE, RMSE, relative L² error vs. FDM
│   ├── plots.py            ← All visualisation routines
│   └── utils.py            ← Seeds, timers, device selection, directories
│
├── data/
│   └── thermal_data.npz    ← Generated dataset (auto-created on first run)
│
└── outputs/
    ├── figures/            ← PNG plots saved after training
    ├── models/             ← PyTorch model checkpoints (.pt)
    └── logs/               ← Loss history (.npz)
```

**Start reading the code in this order:**
1. `config.py` — understand all parameters
2. `src/data_generation.py` — see how the FDM reference solution is built
3. `src/model.py` — the neural network
4. `src/losses.py` — how physics is enforced
5. `src/train.py` — the training loop
6. `main.py` — how everything connects

---

## Configuration Reference

All parameters live in `config.py`. You can change them without touching any other file.

```python
# ── Physical parameters ──────────────────────────────────────────
ALPHA     = 0.01    # thermal diffusivity (normalised)
Q_PEAK    = 50.0    # peak Joule heating [°C]
T_AMBIENT = 20.0    # boundary / initial temperature [°C]
L_CABLE   = 1.0     # cable length (normalised; represents 100 m)
T_END     = 1.0     # simulation duration (normalised; represents 1 hour)

# ── Network architecture ─────────────────────────────────────────
HIDDEN_DIMS = [64, 64, 64, 64]   # 4 hidden layers, 64 neurons each
ACTIVATION  = "tanh"

# ── Training hyperparameters ─────────────────────────────────────
N_EPOCHS         = 8000
LEARNING_RATE    = 1e-3
LR_DECAY_STEP    = 2000   # halve LR every N epochs
N_COLLOCATION    = 5000   # PDE collocation points
N_DATA_POINTS    = 2000   # noisy sensor observations used for training

# ── Loss weights ─────────────────────────────────────────────────
LAMBDA_DATA  = 1.0
LAMBDA_PHYS  = 1.0
LAMBDA_BC    = 10.0
LAMBDA_IC    = 10.0

# ── Logging / checkpointing ──────────────────────────────────────
LOG_INTERVAL   = 100    # print loss every N epochs
SAVE_INTERVAL  = 2000   # save checkpoint every N epochs
```

**Tips for experimentation:**

| Change | Effect |
|--------|--------|
| Increase `N_COLLOCATION` to 10,000 | Better physics enforcement, slower training |
| Decrease `N_DATA_POINTS` to 500 | Tests how well physics compensates for sparse data |
| Increase `LAMBDA_PHYS` to 5.0 | Stronger physics regularisation |
| Add more layers to `HIDDEN_DIMS` | More capacity, may overfit or slow training |
| Change `ACTIVATION` to `"silu"` | Often trains faster than Tanh on modern hardware |

---

## Understanding the Outputs

After training, all outputs are in the `outputs/` directory.

| File | What it shows |
|------|---------------|
| `figures/fdm_solution.png` | Reference FDM temperature field T(x,t) — the ground truth |
| `figures/heat_source.png` | Joule heating field Q(x,t) — the input forcing |
| `figures/pinn_vs_fdm.png` | Side-by-side: PINN prediction vs. FDM + absolute error |
| `figures/spatial_profile.png` | Temperature along the cable at the final time step |
| `figures/temporal_profile.png` | Temperature history at the cable midpoint |
| `figures/loss_curves.png` | Training loss (total + each component) vs. epoch |
| `figures/pde_residual.png` | Where the PINN violates the PDE (should be near-zero) |
| `models/pinn_best.pt` | Best model checkpoint (lowest validation loss) |
| `logs/loss_history.npz` | Epoch-by-epoch loss values for post-processing |

**How to interpret the loss curves:**
- All four loss components should decrease monotonically.
- If `L_bc` or `L_ic` stays high, increase `LAMBDA_BC` / `LAMBDA_IC`.
- If `L_phys` stays high, increase `N_COLLOCATION` or `LAMBDA_PHYS`.
- If `L_data` decreases but `L_phys` spikes, the network is overfitting to data and ignoring physics — increase `LAMBDA_PHYS`.

---

## Why This Matters in Power Systems

| Challenge | How this model helps |
|-----------|---------------------|
| **Ampacity (current-carrying capacity)** | Defined by the maximum allowable conductor temperature (90 °C for XLPE). The thermal model gives ampacity under real loading conditions. |
| **Dynamic Thermal Rating (DTR)** | Instead of a conservative fixed ampacity, DTR monitors real-time temperature to permit higher current when conditions allow — potentially unlocking 10–40 % extra capacity from existing cables. |
| **Overload management** | Short-duration overloads are safe if the thermal time constant allows recovery. The PDE model determines safe overload windows. |
| **Fault detection** | Localized heating from damaged insulation creates spatial anomalies in the thermal field that can be detected by comparing sensor data to the PINN prediction. |
| **Asset lifetime estimation** | Cumulative high-temperature exposure degrades XLPE insulation (Arrhenius degradation law). Accurate thermal history enables better asset management. |

---

## Physical Parameter Glossary

| Term | Symbol | Meaning |
|------|--------|---------|
| Thermal diffusivity | α | Rate at which heat spreads through the material. High α → fast thermal response. |
| Joule heating | Q = I²R | Heat generated per unit volume by current flowing through conductor resistance. |
| Ampacity | — | Maximum continuous current the cable can carry without exceeding its temperature rating. |
| Dynamic Thermal Rating | DTR | Real-time ampacity calculated from current environmental and loading conditions. |
| XLPE | — | Cross-linked polyethylene insulation — standard for modern power cables, rated to 90 °C. |
| Fourier number | r = α·Δt/Δx² | Dimensionless group governing FDM stability; must be ≤ 0.5 for the explicit scheme. |
| Collocation points | — | Points in the (x, t) domain where the PDE residual is evaluated (not necessarily sensor locations). |
| PDE residual | f(x,t) | The amount by which the neural network output fails to satisfy the PDE — trained to be zero. |

---

## Frequently Asked Questions

**Q: Do I need a GPU?**  
No. The network is intentionally small (~17 k parameters). On a modern CPU, 8,000 epochs takes 10–20 minutes.

**Q: Can I use this with real cable sensor data?**  
Yes. Replace the noisy FDM data in `src/data_generation.py` with your real sensor readings. Adjust `N_DATA_POINTS` and ensure your `x`, `t` values are normalised to [0, 1].

**Q: Why is the domain normalised to [0, 1]?**  
Neural networks train more stably when inputs are in a comparable range. `x = 1.0` represents the physical cable length (100 m); `t = 1.0` represents the physical simulation time (3,600 s). The thermal diffusivity `α` is scaled accordingly.

**Q: What happens if I remove the physics loss?**  
Try setting `LAMBDA_PHYS = 0`. The network becomes a pure data-interpolator — it will fit the training points but produce physically inconsistent predictions between them, especially in data-sparse regions.

**Q: Why are boundary/IC weights so much larger (10×)?**  
Boundary and initial conditions are hard constraints. Without the higher weight, the network might satisfy the PDE interior well but drift from the correct boundary values, producing an unphysical solution.

**Q: Can I extend this to 2-D (cable cross-section)?**  
Yes — extend the input from `(x, t)` to `(x, y, t)` and update the PDE residual in `src/losses.py` to include `∂²T/∂y²`. This increases training time significantly.

---

*University of Technology Sydney (UTS) — Power Systems Research*  
*Contact: [bmj.jmd@gmail.com](mailto:bmj.jmd@gmail.com)*
