"""Training loop for the V3 PINN surrogate.

Minimises the composite loss

    L = L_data + lambda_bc * L_bc + lambda_pde * L_pde

where
    L_data : MSE between the PINN prediction and the V2 c_L samples,
    L_bc   : MSE enforcing C_s(P, T) on the notch-surface mask,
    L_pde  : autograd residual of dC_L/dt = -div(J), with the stress-assisted
             flux J from the V2 transport equation.

PyTorch is imported lazily.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# V3 reuses the V2 thermodynamics (C_s boundary value, stress-assisted flux) by
# adding the V2 directory to sys.path. Heavy deps (torch, zarr) stay lazy.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2-data-generation"))

import constants as C
import hydrogen_thermo as ht


def _import_torch():
    import torch  # type: ignore[import-not-found]
    return torch


def data_loss(model, inputs, target):
    """MSE between the PINN prediction and the V2 c_L data."""
    torch = _import_torch()
    return torch.mean((model(inputs) - target) ** 2)


def boundary_loss(model, bc_inputs, P, T):
    """MSE enforcing ``C_L = C_s(P, T)`` on the notch-surface mask."""
    torch = _import_torch()
    c_s = float(ht.surface_concentration(T, P))
    return torch.mean((model(bc_inputs) - c_s) ** 2)


def pde_residual(model, inputs, grad_sigma_h, T, D=C.D_L, V_H=C.V_H):
    """Autograd PDE residual of ``dC_L/dt = -div(J)``.

    ``J = -D grad(C_L) + (D C_L V_H / (R T)) grad(sigma_h)``.
    """
    torch = _import_torch()
    inputs = inputs.clone().detach().requires_grad_(True)
    c = model(inputs)
    grad_c = torch.autograd.grad(c, inputs, grad_outputs=torch.ones_like(c),
                                 create_graph=True)[0]
    # inputs columns: (x, y, t, P, T, sigma_h) -> dC/dx, dC/dy, dC/dt
    dc_dx, dc_dy, dc_dt = grad_c[:, 0], grad_c[:, 1], grad_c[:, 2]
    div_J = -D * (dc_dx + dc_dy) + (D * c[:, 0] * V_H / (C.R * T)) * grad_sigma_h.sum(dim=1)
    return dc_dt + div_J


def train_pinn(model, zarr_store, epochs=5000, lr=1e-3, lambda_bc=1.0,
               lambda_pde=1.0, P=C.P_REF, T=C.T_REF):
    """Adam training loop for the PINN surrogate on a V2 Zarr v3 dataset."""
    torch = _import_torch()
    from dataset import ZarrPINNDataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.net = model.net.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = ZarrPINNDataset(zarr_store)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    history = []
    for epoch in range(int(epochs)):
        for inputs, target, grad_sigma_h in loader:
            inputs, target, grad_sigma_h = (inputs.to(device), target.to(device),
                                             grad_sigma_h.to(device))
            opt.zero_grad()
            l_data = data_loss(model, inputs, target)
            l_bc = boundary_loss(model, inputs, P, T)
            l_pde = torch.mean(pde_residual(model, inputs, grad_sigma_h, T) ** 2)
            loss = l_data + lambda_bc * l_bc + lambda_pde * l_pde
            loss.backward()
            opt.step()
        history.append(float(loss.item()))
    return history


__all__ = ("data_loss", "boundary_loss", "pde_residual", "train_pinn")
