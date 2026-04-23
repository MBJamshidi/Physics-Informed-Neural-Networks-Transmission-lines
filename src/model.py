"""
src/model.py
============
Physics-Informed Neural Network architecture for the power cable
thermal problem.

The network maps (x, t) → T(x, t):

    Input  layer : 2 neurons  — spatial position x, time t
    Hidden layers: 4 × 64 neurons with Tanh activation
    Output layer : 1 neuron   — predicted temperature T

Design rationale
----------------
* **Tanh** is chosen because:
  - it is smooth (infinitely differentiable), which is required for the
    automatic differentiation of ∂T/∂t and ∂²T/∂x² in the PDE loss.
  - it maps R → (−1, 1), giving bounded activations that aid stability
    when gradients are back-propagated through the physics loss.

* **Xavier initialisation** keeps the initial activations in the linear
  region of Tanh, which accelerates early training.

* The network is intentionally small (CPU-trainable in minutes).
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


class HeatPINN(nn.Module):
    """
    Fully-connected neural network T_θ(x, t) for the 1-D heat equation.

    Parameters
    ----------
    input_dim   : number of input features (2: x and t)
    hidden_dims : list of hidden-layer widths
    output_dim  : number of outputs (1: temperature)
    activation  : activation function name ("tanh" supported)
    """

    def __init__(
        self,
        input_dim:   int  = cfg.INPUT_DIM,
        hidden_dims: list = cfg.HIDDEN_DIMS,
        output_dim:  int  = cfg.OUTPUT_DIM,
        activation:  str  = cfg.ACTIVATION,
    ):
        super().__init__()

        # ── Build the layer sequence ───────────────────────────────────────
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

        layers = []
        in_features = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_features, width))
            layers.append(act_fn())
            in_features = width
        layers.append(nn.Linear(in_features, output_dim))   # no activation on output

        self.net = nn.Sequential(*layers)

        # ── Xavier (Glorot) initialisation ─────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (N, 1) tensor — spatial positions in [0, L_CABLE]
        t : (N, 1) tensor — time values in [0, T_END]

        Returns
        -------
        T : (N, 1) tensor — predicted temperature [°C]
        """
        # Concatenate inputs along the feature dimension → (N, 2)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    # ── Convenience methods ────────────────────────────────────────────────

    def predict_grid(self, x_np, t_np) -> "np.ndarray":
        """
        Evaluate T on a numpy meshgrid.  Returns a (nt, nx) numpy array.
        Used for evaluation and plotting — no gradient tracking.
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            X, T_m = np.meshgrid(x_np, t_np)         # (nt, nx)
            x_flat = X.ravel().astype(np.float32)
            t_flat = T_m.ravel().astype(np.float32)
            x_ten = torch.tensor(x_flat).unsqueeze(1)
            t_ten = torch.tensor(t_flat).unsqueeze(1)
            T_pred = self.forward(x_ten, t_ten).squeeze(1).numpy()
        return T_pred.reshape(len(t_np), len(x_np))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        n = self.count_parameters()
        return (
            super().__repr__()
            + f"\n  ↳ Trainable parameters: {n:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = HeatPINN()
    print(model)

    # Dummy forward pass
    x_test = torch.rand(8, 1, requires_grad=True)
    t_test = torch.rand(8, 1, requires_grad=True)
    T_out  = model(x_test, t_test)
    print(f"Output shape: {T_out.shape}  (expected [8, 1])")

    # Verify that autograd works through the network
    dT_dx = torch.autograd.grad(T_out.sum(), x_test, create_graph=True)[0]
    print(f"∂T/∂x shape: {dT_dx.shape}  (autograd OK)")
