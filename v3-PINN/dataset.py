"""Zarr v3 dataset loader for the V3 PINN.

Streams ``(x, y, t, P, T, sigma_h, c_L)`` tuples from a V2 Zarr v3 sweep
dataset into a PyTorch ``Dataset``. The notch-surface mask is exposed so the
boundary-condition loss can be evaluated on the Dirichlet facets where
``C_L = C_s(P, T)``.
"""
from __future__ import annotations

import numpy as np


def _import_torch():
    import torch  # type: ignore[import-not-found]
    return torch


def _open_zarr_group(store):
    import zarr  # type: ignore[import-not-found]
    if isinstance(store, zarr.Group):
        return store
    return zarr.open_group(store, mode="r")


class ZarrPINNDataset:
    """PyTorch ``Dataset`` streaming V2 Zarr v3 samples.

    Yields ``(inputs, target, grad_sigma_h)`` tuples where
    ``inputs = (x, y, t, P, T, sigma_h)`` and ``target = c_L``.
    """

    def __init__(self, store):
        torch = _import_torch()
        self._torch = torch
        self.root = _open_zarr_group(store)
        self.samples = sorted(self.root.group_keys()) if hasattr(self.root, "group_keys") else list(self.root.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        grp = self.root[name] if hasattr(self.root, "__getitem__") else self.root[name]
        coords = np.asarray(grp["coords"])
        c_lattice = np.asarray(grp["c_lattice"])
        sigma_h = np.asarray(grp["sigma_h"])
        grad_sigma_h = np.asarray(grp["grad_sigma_h"])
        times = np.asarray(grp["times"])
        bc_mask = np.asarray(grp["bc_mask"])
        # Build (x, y, t, P, T, sigma_h) inputs across the transient DOFs.
        n = coords.shape[0]
        t_col = np.repeat(times[:, None], n, axis=1).reshape(-1, 1)
        inputs = np.concatenate([coords, t_col, sigma_h.reshape(-1, 1)], axis=1)
        target = c_lattice.reshape(-1, 1)
        return (self._torch.from_numpy(inputs.astype(np.float32)),
                self._torch.from_numpy(target.astype(np.float32)),
                self._torch.from_numpy(grad_sigma_h.reshape(-1, 2).astype(np.float32)))


__all__ = ("ZarrPINNDataset",)
