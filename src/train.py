"""
src/train.py
============
Training loop for the Physics-Informed Neural Network.

Strategy
--------
* Adam optimiser — adaptive learning rate, works well for PINNs.
* StepLR scheduler — halve the learning rate every LR_DECAY_STEP epochs
  to fine-tune in later training.
* All training data (collocation + BC + IC + observed data) is loaded
  once into memory — no batching needed since the point sets are small.
* The best model (lowest total loss) is saved as a checkpoint.
* Loss history is saved to disk for post-hoc plotting.
"""

import torch
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from src.model  import HeatPINN
from src.losses import total_loss


# ─────────────────────────────────────────────────────────────────────────────

def train(model: HeatPINN,
          training_data: dict,
          n_epochs: int  = cfg.N_EPOCHS) -> dict:
    """
    Train the PINN.

    Parameters
    ----------
    model         : HeatPINN instance (freshly initialised or pre-loaded)
    training_data : dict produced by dataset.build_all_training_data()
    n_epochs      : number of gradient-descent steps

    Returns
    -------
    history : dict with lists 'total', 'data', 'physics', 'bc', 'ic'
              (one entry per epoch)
    """
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR,   exist_ok=True)

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # ── Learning rate scheduler ───────────────────────────────────────────
    # Multiplies lr by GAMMA every STEP epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser,
        step_size = cfg.LR_DECAY_STEP,
        gamma     = cfg.LR_DECAY_GAMMA,
    )

    # ── History buffers ────────────────────────────────────────────────────
    history = {"total": [], "data": [], "physics": [], "bc": [], "ic": []}

    best_loss   = float("inf")
    t0          = time.time()

    print("=" * 65)
    print("  PINN Training — Power Cable Thermal Dynamics")
    print("=" * 65)
    print(f"  Epochs          : {n_epochs}")
    print(f"  Learning rate   : {cfg.LEARNING_RATE}")
    print(f"  LR decay        : ×{cfg.LR_DECAY_GAMMA} every {cfg.LR_DECAY_STEP} epochs")
    print(f"  Loss weights    : data={cfg.LAMBDA_DATA}, phys={cfg.LAMBDA_PHYS}, "
          f"bc={cfg.LAMBDA_BC}, ic={cfg.LAMBDA_IC}")
    print(f"  Colloc. points  : {cfg.N_COLLOCATION}")
    print(f"  Data points     : {cfg.N_DATA_POINTS}")
    print(f"  Parameters      : {model.count_parameters():,}")
    print("=" * 65)

    for epoch in range(1, n_epochs + 1):

        model.train()
        optimiser.zero_grad()

        # ── Forward pass + loss ────────────────────────────────────────────
        loss, comps = total_loss(model, training_data)

        # ── Backward pass ──────────────────────────────────────────────────
        loss.backward()

        # Gradient clipping prevents occasional exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        scheduler.step()

        # ── Record history ─────────────────────────────────────────────────
        for key in history:
            history[key].append(comps[key] if key != "total" else comps["total"])

        # ── Checkpoint: save best model ────────────────────────────────────
        if comps["total"] < best_loss:
            best_loss = comps["total"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimiser":   optimiser.state_dict(),
                "best_loss":   best_loss,
            }, cfg.MODEL_CKPT)

        # ── Periodic save ──────────────────────────────────────────────────
        if epoch % cfg.SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(
                cfg.MODELS_DIR, f"pinn_epoch_{epoch:05d}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)

        # ── Logging ────────────────────────────────────────────────────────
        if epoch % cfg.LOG_INTERVAL == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now  = optimiser.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d}/{n_epochs} | "
                f"LR={lr_now:.2e} | "
                f"L_total={comps['total']:10.4f} | "
                f"L_data={comps['data']:8.4f} | "
                f"L_phys={comps['physics']:8.4f} | "
                f"L_bc={comps['bc']:8.4f} | "
                f"L_ic={comps['ic']:8.4f} | "
                f"t={elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print("=" * 65)
    print(f"  Training complete in {total_time:.1f}s  |  Best loss: {best_loss:.6f}")
    print(f"  Best model saved → {cfg.MODEL_CKPT}")
    print("=" * 65)

    # ── Save loss history ──────────────────────────────────────────────────
    np.savez_compressed(
        cfg.LOSS_LOG_FILE,
        total   = np.array(history["total"]),
        data    = np.array(history["data"]),
        physics = np.array(history["physics"]),
        bc      = np.array(history["bc"]),
        ic      = np.array(history["ic"]),
    )
    print(f"  Loss history saved → {cfg.LOSS_LOG_FILE}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load best checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_best_model(model: HeatPINN) -> HeatPINN:
    """Load the best-loss checkpoint into `model` in-place and return it."""
    ckpt = torch.load(cfg.MODEL_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    print(f"[Load] Best model (epoch {ckpt['epoch']}, "
          f"loss={ckpt['best_loss']:.6f}) loaded from {cfg.MODEL_CKPT}")
    return model
