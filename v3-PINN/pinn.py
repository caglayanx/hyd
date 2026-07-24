"""Fourier-feature PINN surrogate for the V3 hydrogen transport model.

Maps coordinates and physics inputs to the lattice hydrogen concentration:

    (x, y, t, P, T, sigma_h) -> c_L

Architecture: a Fourier-feature MLP (random Fourier features followed by a
multi-layer perceptron). PyTorch is imported lazily so this module imports
cleanly without the ML stack.

Loss (see :mod:`train`):
    L = L_data + lambda_bc L_bc + lambda_pde L_pde
    L_data : MSE against V2 c_L samples
    L_bc   : MSE enforcing C_s(P, T) on the notch-surface mask
    L_pde  : autograd residual of dC_L/dt = -div(J) (V2 transport equation)
"""
from __future__ import annotations

import numpy as np


def _import_torch():
    import torch  # type: ignore[import-not-found]
    return torch


class HydrogenPINN:
    """Fourier-feature MLP surrogate ``(x,y,t,P,T,sigma_h) -> c_L``.

    Parameters
    ----------
    in_features : int
        Number of physical inputs (default 6: x, y, t, P, T, sigma_h).
    hidden : int
        Hidden width of the MLP.
    n_layers : int
        Number of hidden linear layers.
    fourier_modes : int
        Number of random Fourier features (0 disables the Fourier mapping).
    sigma_scale : float
        Standard deviation of the random Fourier-feature projection.
    """

    def __init__(self, in_features: int = 6, hidden: int = 128, n_layers: int = 6,
                 fourier_modes: int = 32, sigma_scale: float = 1.0, seed: int = 0):
        torch = _import_torch()
        self._torch = torch
        self.in_features = in_features
        self.fourier_modes = fourier_modes
        self.sigma_scale = sigma_scale
        g = torch.Generator().manual_seed(int(seed))
        if fourier_modes > 0:
            self.B = torch.randn(fourier_modes, in_features, generator=g) * sigma_scale
        else:
            self.B = torch.zeros(0, in_features)
        self.net = self._build_net(hidden, n_layers)

    def _build_net(self, hidden, n_layers):
        torch = self._torch
        nn = torch.nn
        feat_dim = self.in_features if self.fourier_modes == 0 else 2 * self.fourier_modes
        layers = [nn.Linear(feat_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        return nn.Sequential(*layers)

    def features(self, inputs):
        """Random Fourier features ``[sin(2 pi B x), cos(2 pi B x)]`` (or pass-through)."""
        torch = self._torch
        if self.fourier_modes == 0:
            return inputs
        proj = torch.matmul(inputs, self.B.t())
        return torch.cat([torch.sin(2.0 * np.pi * proj), torch.cos(2.0 * np.pi * proj)], dim=-1)

    def __call__(self, inputs):
        return self.net(self.features(inputs))

    def parameters(self):
        return self.net.parameters()


__all__ = ("HydrogenPINN",)
