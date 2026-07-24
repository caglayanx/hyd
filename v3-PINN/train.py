"""Training loop and loss functions for the V3 PINN surrogate.

Composite loss:

    L = L_data + lambda_bc * L_bc + lambda_pde * L_pde

    L_data : MSE between the PINN prediction and the normalised c_L data
             (c_L / C_s, computed in :class:`dataset.ZarrPINNDataset`).
    L_bc   : MSE enforcing the predicted concentration == 1.0 (the normalised
             Dirichlet value C_s) on the notch-surface DOFs (bc_mask == True).
    L_pde  : autograd residual of the stress-assisted diffusion equation

             R = dc/dt - D(T) laplacian(c)
                 + (D(T) V_H / (R T)) (grad(c) . grad(sigma_h)),

             where ``grad(sigma_h)`` is the exact nodal gradient provided by the
             dataset (``grad_sigma``), NOT a numerical derivative. ``D(T)`` is
             the Arrhenius lattice diffusivity from :mod:`transport`.

All derivatives of ``c`` (dc/dx, dc/dy, dc/dt, d2c/dx2, d2c/dy2) are computed
with ``torch.autograd.grad(..., create_graph=True)`` so the residual stays in
the autograd graph for backpropagation.

PyTorch is imported lazily.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# V3 reuses V2 thermodynamics (C_s normaliser, Arrhenius D(T), V_H) by adding
# the V2 directory to sys.path. Heavy deps (torch, zarr) stay lazy.
_V2 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v2-data-generation"))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

import constants as C  # type: ignore[import-not-found]
import hydrogen_thermo as ht  # type: ignore[import-not-found]
import transport as tr  # type: ignore[import-not-found]


def _import_torch():
    import torch  # type: ignore[import-not-found]
    return torch


# ===========================================================================
# Loss functions.
# ===========================================================================
def data_loss(model, inputs, target):
    """MSE between the PINN prediction and the normalised c_L data."""
    torch = _import_torch()
    return torch.mean((model(inputs) - target) ** 2)


def boundary_loss(model, inputs, bc_mask):
    """MSE enforcing ``c_L = 1.0`` (normalised C_s) on notch-surface DOFs.

    Only points with ``bc_mask == True`` contribute; if none are present in the
    batch the loss is zero (no NaN).
    """
    torch = _import_torch()
    if not bool(bc_mask.any()):
        return torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    bc_inputs = inputs[bc_mask]
    pred = model(bc_inputs)
    return torch.mean((pred - 1.0) ** 2)


def pde_residual(model, inputs, grad_sigma, T):
    """Autograd PDE residual of the stress-assisted diffusion equation.

        R = dc/dt - D(T) (d2c/dx2 + d2c/dy2)
            + (D(T) V_H / (R T)) (dc/dx * d sigma/dx + dc/dy * d sigma/dy)

    ``grad_sigma`` is the exact nodal gradient (d sigma/dx, d sigma/dy) from the
    dataset. The model output column layout is ``inputs = (x, y, t, P, T, sigma_h)``.
    """
    torch = _import_torch()
    x = inputs.clone().detach().requires_grad_(True)
    c = model(x)  # (N, 1)

    # First derivatives: dc/dx, dc/dy, dc/dt.
    grad_c = torch.autograd.grad(
        c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]  # (N, 6)
    dc_dx = grad_c[:, 0:1]
    dc_dy = grad_c[:, 1:2]
    dc_dt = grad_c[:, 2:3]

    # Second spatial derivatives: d2c/dx2, d2c/dy2.
    c_xx = torch.autograd.grad(
        dc_dx, x, grad_outputs=torch.ones_like(dc_dx), create_graph=True)[0][:, 0:1]
    c_yy = torch.autograd.grad(
        dc_dy, x, grad_outputs=torch.ones_like(dc_dy), create_graph=True)[0][:, 1:2]

    laplacian_c = c_xx + c_yy
    grad_c_xy = torch.cat([dc_dx, dc_dy], dim=1)  # (N, 2)

    # Temperature-dependent Arrhenius diffusivity D(T) (per-batch scalar field).
    D_T = tr.diffusion_coefficient(float(T))  # Python float -> constant tensor
    D_T = torch.as_tensor(D_T, dtype=x.dtype, device=x.device)
    RT = torch.as_tensor(C.R * float(T), dtype=x.dtype, device=x.device)
    V_H = torch.as_tensor(C.V_H, dtype=x.dtype, device=x.device)

    drift = (D_T * V_H / RT) * (grad_c_xy * grad_sigma).sum(dim=1, keepdim=True)
    R = dc_dt - D_T * laplacian_c + drift
    return R


def pde_loss(model, inputs, grad_sigma, T):
    """Mean-squared PDE residual ``mean(R^2)``."""
    torch = _import_torch()
    R = pde_residual(model, inputs, grad_sigma, T)
    return torch.mean(R ** 2)


# ===========================================================================
# Training loop.
# ===========================================================================
def train_pinn(model=None, zarr_store=None, *, epochs: int = 5000, batch_size: int = 4096,
               lr: float = 1e-3, lambda_bc: float = 1.0, lambda_pde: float = 1.0,
               P: float = C.P_REF, T: float = C.T_REF, device: str = None,
               num_workers: int = 0):
    """Adam training loop for the PINN surrogate on a V2 Zarr v3 dataset.

    Parameters
    ----------
    model : HydrogenPINN, optional
        Surrogate to train. If ``None``, a fresh model is built with
        standardisation statistics fitted to the dataset.
    zarr_store : str or zarr.Group
        V2 dataset root (required).
    epochs : int
        Number of epochs over the full flattened dataset.
    batch_size : int
        Mini-batch size (points per batch).
    lr : float
        Adam learning rate.
    lambda_bc, lambda_pde : float
        Weights of the BC and PDE residual losses.
    P, T : float
        Reference pressure/temperature (used only as a fallback for ``D(T)``
        when a batch carries no per-sample T; the dataset T attribute is
        preferred).
    device : str, optional
        ``"cuda"`` or ``"cpu"`` (auto-detected if None).
    num_workers : int
        DataLoader workers.
    """
    torch = _import_torch()
    from dataset import ZarrPINNDataset
    from pinn import HydrogenPINN
    from torch.utils.data import DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ZarrPINNDataset(zarr_store)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, collate_fn=ZarrPINNDataset.collate_fn,
                         drop_last=False)

    # If the caller passed a bare model, fit its standardisation to the dataset.
    if model is None:
        mean, std = dataset.input_stats()
        model = HydrogenPINN(input_mean=mean.tolist(), input_std=std.tolist())
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(int(epochs)):
        running = 0.0
        n_batches = 0
        for batch in loader:
            # Assemble the (N, 6) input tensor and move to device.
            coords = batch["coords"].to(device)            # (N, 2)
            t = batch["time"].to(device)                     # (N, 1)
            P_b = batch["P"].to(device)                     # (N, 1)
            T_b = batch["T"].to(device)                     # (N, 1)
            sigma_h = batch["sigma_h"].to(device)           # (N, 1)
            grad_sigma = batch["grad_sigma"].to(device)     # (N, 2)
            target = batch["c_lattice"].to(device)          # (N, 1) normalised
            bc_mask = batch["bc_mask"].to(device)           # (N, 1) bool

            inputs = torch.cat([coords, t, P_b, T_b, sigma_h], dim=1)  # (N, 6)

            opt.zero_grad()
            l_data = data_loss(model, inputs, target)
            l_bc = boundary_loss(model, inputs, bc_mask.squeeze(-1))
            # Use the per-sample T carried by the batch for D(T); fall back to T.
            T_eff = float(T_b.mean().item()) if T_b.numel() else float(T)
            l_pde = pde_loss(model, inputs, grad_sigma, T_eff)
            loss = l_data + lambda_bc * l_bc + lambda_pde * l_pde
            loss.backward()
            opt.step()

            running += float(loss.item())
            n_batches += 1
        history.append(running / max(n_batches, 1))
    return history


__all__ = ("data_loss", "boundary_loss", "pde_residual", "pde_loss", "train_pinn")
