"""Fourier-feature MLP surrogate for the V3 hydrogen transport model.

Maps the spatio-temporal and physical inputs to the normalised lattice
hydrogen concentration:

    (x, y, t, P, T, sigma_h) -> c_L / C_s

Architecture:
    * Inputs are first standardised by a fixed affine map ``(x - mean)/std``
      (registered as non-trainable buffers) so the raw physical scales
      (coords ~1e-4 m, t ~1e-3 s, P ~1e7 Pa, T ~3e2 K, sigma_h ~1e10 Pa) do NOT
      saturate the ``tanh`` activations. Because the standardisation lives inside
      the forward pass, ``torch.autograd.grad`` still yields correct derivatives
      with respect to the *physical* inputs (the chain rule is automatic).
    * A random Fourier-feature mapping is applied ONLY to the standardised
      spatio-temporal inputs (x, y, t) to resolve the sharp gradients near the
      notch tip:

          gamma(x,y,t) = [sin(2 pi B z), cos(2 pi B z)] ,  z = (x, y, t),

      with B in R^{m x 3} drawn from N(0, sigma_scale^2).
    * The standardised physical parameters (P, T, sigma_h) are concatenated to
      the Fourier features (NOT Fourier-mapped) before the hidden MLP layers.

``HydrogenPINN`` subclasses ``torch.nn.Module`` so ``.to(device)``,
``.parameters()``, ``state_dict`` and ``train()/eval()`` all work natively.
PyTorch is imported lazily at construction time.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Spatio-temporal inputs that receive the Fourier mapping (x, y, t).
N_ST_INPUTS: int = 3
# Physical inputs concatenated after the Fourier features (P, T, sigma_h).
N_PHYS_INPUTS: int = 3
IN_FEATURES: int = N_ST_INPUTS + N_PHYS_INPUTS  # 6

# Sensible physical default standardisation (mean, std) per input column.
DEFAULT_INPUT_MEAN = (0.0, 0.0, 5e-3, 35e6, 3e2, 5e9)
DEFAULT_INPUT_STD = (1e-4, 1e-4, 1e-2, 1e7, 5e1, 1e10)


def _import_torch():
    import torch  # type: ignore[import-not-found]
    return torch


class HydrogenPINN:
    """Fourier-feature MLP surrogate ``(x,y,t,P,T,sigma_h) -> c_L``.

    Parameters
    ----------
    hidden : int, optional
        Hidden width of the MLP (default 128).
    n_layers : int, optional
        Number of hidden linear layers (default 6).
    fourier_modes : int, optional
        Number of random Fourier features (0 disables the Fourier mapping and
        passes the raw standardised spatio-temporal inputs through).
    sigma_scale : float, optional
        Standard deviation of the random Fourier-feature projection ``B``.
    input_mean, input_std : sequence of float, optional
        Per-input standardisation statistics (length 6). Defaults are sensible
        physical scales; for best results pass statistics computed from the
        dataset (see :meth:`ZarrPINNDataset.input_stats`).
    seed : int, optional
        RNG seed for the Fourier-feature initialisation.
    """

    def __init__(self, hidden: int = 128, n_layers: int = 6,
                 fourier_modes: int = 32, sigma_scale: float = 1.0,
                 input_mean: Sequence[float] = DEFAULT_INPUT_MEAN,
                 input_std: Sequence[float] = DEFAULT_INPUT_STD,
                 seed: int = 0):
        torch = _import_torch()
        nn = torch.nn
        self._torch = torch
        self.fourier_modes = int(fourier_modes)
        self.sigma_scale = float(sigma_scale)

        if len(input_mean) != IN_FEATURES or len(input_std) != IN_FEATURES:
            raise ValueError(f"input_mean/std must have length {IN_FEATURES}.")
        mean_t = torch.as_tensor(np.asarray(input_mean, dtype=np.float32))
        std_t = torch.as_tensor(np.asarray(input_std, dtype=np.float32))
        std_t = torch.clamp(std_t, min=1e-12)  # guard against divide-by-zero

        rng = np.random.default_rng(seed)
        if self.fourier_modes > 0:
            B = rng.normal(0.0, self.sigma_scale, size=(self.fourier_modes, N_ST_INPUTS))
        else:
            B = np.zeros((0, N_ST_INPUTS), dtype=np.float32)
        B_tensor = torch.as_tensor(B, dtype=torch.float32)

        feat_dim = (2 * self.fourier_modes + N_PHYS_INPUTS) if self.fourier_modes > 0 \
            else (N_ST_INPUTS + N_PHYS_INPUTS)
        layers = [nn.Linear(feat_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]

        class _Net(nn.Module):
            def __init__(self_inner, mean, std, b, mlp):
                super().__init__()
                self_inner.register_buffer("input_mean", mean)
                self_inner.register_buffer("input_std", std)
                self_inner.register_buffer("B", b)
                self_inner.mlp = mlp

            def forward(self_inner, inputs):
                # Fixed affine standardisation (chain-ruled by autograd).
                z = (inputs - self_inner.input_mean) / self_inner.input_std
                zst = z[:, :N_ST_INPUTS]
                if self_inner.B.shape[0] == 0:
                    feats = zst
                else:
                    proj = torch.matmul(zst, self_inner.B.t())
                    feats = torch.cat([torch.sin(2.0 * np.pi * proj),
                                       torch.cos(2.0 * np.pi * proj)], dim=-1)
                phys = z[:, N_ST_INPUTS:]
                return self_inner.mlp(torch.cat([feats, phys], dim=-1))

        self.net = _Net(mean_t, std_t, B_tensor, nn.Sequential(*layers))

    # ------------------------------------------------------------------
    # Forward + nn.Module convenience proxies.
    # ------------------------------------------------------------------
    def __call__(self, inputs):
        return self.net(inputs)

    def parameters(self):
        return self.net.parameters()

    def to(self, *args, **kwargs):
        self.net = self.net.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def eval(self):
        self.net.eval()
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
        return self


__all__ = ("HydrogenPINN", "N_ST_INPUTS", "N_PHYS_INPUTS", "IN_FEATURES",
           "DEFAULT_INPUT_MEAN", "DEFAULT_INPUT_STD")
