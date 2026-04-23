"""
src/utils.py
============
Shared utility functions used across the project.
"""

import os
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int = cfg.SEED):
    """
    Fix all random seeds for full reproducibility.
    Covers: Python built-in, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN algorithms (if GPU is used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────

def ensure_directories():
    """Create all required output directories if they do not exist."""
    for path in [cfg.DATA_DIR, cfg.FIGURES_DIR, cfg.MODELS_DIR, cfg.LOGS_DIR]:
        os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Simple timing context manager
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager that prints elapsed time on exit."""

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._t0
        print(f"  [{self.label}] Done in {elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return CUDA device if available, else CPU.
    This project is designed to run on CPU, but CUDA is used if present.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Device] Using: {device}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print project banner
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║   Physics-Informed Neural Network                           ║
║   Power Cable Thermal Dynamics                              ║
║   PDE: ∂T/∂t = α ∂²T/∂x² + Q(x,t)                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ─────────────────────────────────────────────────────────────────────────────
# Load loss history from disk
# ─────────────────────────────────────────────────────────────────────────────

def load_loss_history(path: str = cfg.LOSS_LOG_FILE) -> dict:
    """Load the NPZ loss log saved during training."""
    raw = np.load(path)
    return {k: raw[k].tolist() for k in raw.files}


# ─────────────────────────────────────────────────────────────────────────────
# Normalise / denormalise helpers (for future extension)
# ─────────────────────────────────────────────────────────────────────────────

def normalise(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map arr from [vmin, vmax] → [0, 1]."""
    return (arr - vmin) / (vmax - vmin + 1e-12)


def denormalise(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map arr from [0, 1] → [vmin, vmax]."""
    return arr * (vmax - vmin) + vmin
