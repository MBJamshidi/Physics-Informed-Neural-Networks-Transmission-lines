"""
main.py
=======
End-to-end pipeline for the Power Cable Thermal PINN project.

Execution order:
  1. Setup        — seeds, directories, banner
  2. Data         — FDM numerical solution + noisy observations
  3. Dataset      — collocation, BC, IC points for PINN training
  4. Model        — instantiate the neural network
  5. Training     — PINN optimisation
  6. Evaluation   — PINN vs FDM metrics
  7. Plots        — all figures saved to outputs/figures/

Run from the project root:
    python main.py

All outputs land in:
    data/           — thermal_data.npz
    outputs/figures — PNG plots
    outputs/models  — checkpoints
    outputs/logs    — loss history
"""

import os
import sys

# ── Ensure the project root is on the path ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as cfg
from src.utils         import set_all_seeds, ensure_directories, print_banner, Timer, load_loss_history
from src.data_generation import solve_heat_equation_fdm, save_dataset, load_dataset
from src.dataset         import build_all_training_data
from src.model           import HeatPINN
from src.train           import train, load_best_model
from src.evaluate        import evaluate, compute_pde_residual_on_grid
from src.plots           import generate_all_plots


def main():
    # ── 1. Setup ─────────────────────────────────────────────────────────
    print_banner()
    set_all_seeds(cfg.SEED)
    ensure_directories()

    # ── 2. Data generation ───────────────────────────────────────────────
    print("\n[Step 1/6]  Generating synthetic thermal data via FDM …")
    with Timer("FDM solver"):
        if os.path.exists(cfg.DATA_FILE):
            print(f"  Found existing dataset at {cfg.DATA_FILE} — loading.")
            data_dict = load_dataset(cfg.DATA_FILE)
        else:
            fdm_result = solve_heat_equation_fdm()
            save_dataset(fdm_result)
            data_dict  = load_dataset(cfg.DATA_FILE)

    print(f"  Grid shape: {data_dict['T_true'].shape}  (nt × nx)")
    print(f"  T range: [{data_dict['T_true'].min():.2f}, "
          f"{data_dict['T_true'].max():.2f}] °C")

    # ── 3. Build training data ────────────────────────────────────────────
    print("\n[Step 2/6]  Assembling PINN training tensors …")
    with Timer("Dataset"):
        training_data = build_all_training_data(data_dict)

    print(f"  Data points      : {len(training_data['x_data'])}")
    print(f"  Colloc. points   : {len(training_data['x_col'])}")
    print(f"  BC points (each) : {len(training_data['bc_left'][0])}")
    print(f"  IC points        : {len(training_data['x_ic'])}")

    # ── 4. Instantiate model ─────────────────────────────────────────────
    print("\n[Step 3/6]  Building PINN …")
    model = HeatPINN()
    print(model)

    # ── 5. Training ───────────────────────────────────────────────────────
    print("\n[Step 4/6]  Training …")
    with Timer("Training"):
        history = train(model, training_data, n_epochs=cfg.N_EPOCHS)

    # Load the best checkpoint for evaluation
    model = load_best_model(model)

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    print("\n[Step 5/6]  Evaluating PINN vs FDM …")
    with Timer("Evaluation"):
        results = evaluate(model, data_dict)

    # PDE residual on a coarse sub-grid
    print("  Computing PDE residual field …")
    residual_field, x_sub, t_sub = compute_pde_residual_on_grid(
        model, data_dict, subsample=20
    )

    # ── 7. Plots ──────────────────────────────────────────────────────────
    print("\n[Step 6/6]  Generating plots …")
    with Timer("Plotting"):
        generate_all_plots(
            data_dict, results, history,
            residual_field=residual_field, x_sub=x_sub, t_sub=t_sub
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Pipeline complete!")
    print("═" * 60)
    print(f"  Dataset      : {cfg.DATA_FILE}")
    print(f"  Best model   : {cfg.MODEL_CKPT}")
    print(f"  Figures      : {cfg.FIGURES_DIR}")
    print(f"  Loss log     : {cfg.LOSS_LOG_FILE}")
    print("─" * 60)
    print(f"  MAE          : {results['mae']:.4f} °C")
    print(f"  RMSE         : {results['rmse']:.4f} °C")
    print(f"  Max error    : {results['max_err']:.4f} °C")
    print(f"  Rel L² error : {results['rel_l2']:.4f}")
    print("═" * 60)


if __name__ == "__main__":
    main()
