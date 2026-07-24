"""Chunked Zarr v3 writer for the V2 transient field dataset.

Streams the per-step DOLFINx DOF fields (``coords``, ``c_L``, ``sigma_h``,
``grad(sigma_h)``, ``times``, ``bc_mask``) into a chunked Zarr v3 group, with
no VTK/XDMF intermediates. The lumped-mass formulation in ``solver.py``
guarantees non-negative ``c_L`` so the writer raises on negative samples.

Zarr is imported lazily so this module imports cleanly without the data stack.
"""
from __future__ import annotations

import numpy as np


def _import_zarr():
    import zarr  # type: ignore[import-not-found]
    return zarr


class ZarrDataWriter:
    """Chunked Zarr v3 writer for the V2 mech-hydrogen sweep dataset.

    Each sample is written as a subgroup recording the transient DOF tensors.
    """

    def __init__(self):
        self._zarr = _import_zarr()
        self.root = None

    def open_group(self, path: str):
        """Open (or create) a Zarr v3 group at ``path``."""
        self.root = self._zarr.open_group(path, mode="w")
        return self.root

    def write_sample(self, name: str, coords, c_lattice, sigma_h, grad_sigma_h,
                     times, bc_mask, *, check_nonnegative: bool = True):
        """Write one sweep sample as a subgroup of the dataset."""
        if self.root is None:
            raise RuntimeError("Call open_group(path) before write_sample().")
        if check_nonnegative:
            c = np.asarray(c_lattice)
            if np.nanmin(c) < -1e-12:
                raise RuntimeError(
                    f"Negative c_L (min={float(np.nanmin(c)):.3e}) in sample {name!r}; "
                    "lumped-mass formulation should prevent ringing."
                )
        grp = self.root.create_group(name)
        grp.create_array("coords", data=coords)
        grp.create_array("c_lattice", data=c_lattice)
        grp.create_array("sigma_h", data=sigma_h)
        grp.create_array("grad_sigma_h", data=grad_sigma_h)
        grp.create_array("times", data=times)
        grp.create_array("bc_mask", data=bc_mask)
        return grp


__all__ = ("ZarrDataWriter",)
