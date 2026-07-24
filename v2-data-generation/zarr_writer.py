"""Chunked Zarr v3 exporter for the V2 transient field dataset.

For each simulation the writer creates a subgroup (``sample_000``,
``sample_001``, ...) and records the transient DOF tensors required for V3
PINN training:

    coords             : (n_dof, 2)         float64   nodal coordinates
    bc_mask            : (n_dof,)           bool      notch-surface Dirichlet DOFs
    c_lattice          : (n_steps, n_dof)   float64   transient lattice H
    hydrostatic_stress : (n_dof,)           float64   prescribed sigma_h field
    grad_sigma         : (n_dof, 2)         float64   exact nodal grad(sigma_h)
    time_points        : (n_steps,)         float64   sample times

Per-sample scalar metadata are stored as group attributes (``.attrs``):

    P, T, K_I, r_notch, dislocation_type

Zarr is imported lazily so this module imports cleanly without the data stack.
The writer validates non-negativity of ``c_lattice`` (the lumped-mass solver
should never produce negative concentrations).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _import_zarr():
    """Lazily import Zarr v3."""
    try:
        import zarr  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "zarr>=3.0 is required to export the V2 dataset. Install it with "
            "`pip install zarr>=3.0`."
        ) from exc
    return zarr


class ZarrDataWriter:
    """Chunked Zarr v3 writer for the V2 mech-hydrogen sweep dataset.

    Parameters
    ----------
    store : str or zarr.Store, optional
        Path/URL/store for the root Zarr v3 group. If None, an in-memory store
        is used (handy for tests).
    chunk_dof : int, optional
        Chunk size along the DOF axis (default 4096).
    chunk_steps : int, optional
        Chunk size along the time axis (default 64).
    """

    def __init__(self, store: Optional[object] = None, *, chunk_dof: int = 4096,
                 chunk_steps: int = 64, mode: str = "w"):
        zarr = _import_zarr()
        self._zarr = zarr
        self.chunk_dof = int(chunk_dof)
        self.chunk_steps = int(chunk_steps)
        self.mode = str(mode)
        if store is None:
            store = zarr.storage.MemoryStore()
        self.root = zarr.open_group(store, mode=self.mode)
        self._count = 0

    def _create_array(self, grp, name, data, *, chunks):
        """Create a chunked array in ``grp`` from ``data`` (dtype inferred)."""
        return grp.create_array(name, data=data, chunks=chunks)


    # ------------------------------------------------------------------
    # Sample writing.
    # ------------------------------------------------------------------
    def write_sample(self, fields: Dict[str, np.ndarray], *, P: float, T: float,
                     K_I: float, r_notch: float, dislocation_type: str,
                     check_nonnegative: bool = True,
                     name: Optional[str] = None) -> str:
        """Write one simulation as a subgroup ``sample_NNN``.

        Parameters
        ----------
        fields : dict
            Must contain ``coords`` (n_dof, 2), ``bc_mask`` (n_dof,),
            ``c_lattice`` (n_steps, n_dof), ``hydrostatic_stress`` (n_dof,),
            ``grad_sigma`` (n_dof, 2), ``time_points`` (n_steps,).
        P, T : float
            Operating pressure (Pa) and temperature (K).
        K_I : float
            Stress-intensity scale (Pa sqrt(m)).
        r_notch : float
            Notch root radius (m).
        dislocation_type : str
            Dislocation family tag (e.g. ``"edge"``).
        check_nonnegative : bool, optional
            Raise if ``c_lattice`` has negative values (default True).
        name : str, optional
            Explicit subgroup name (e.g. ``"sample_042"``). If ``None`` the
            internal sequential counter ``sample_{count:03d}`` is used. Passing
            the deterministic grid index makes MPI/resume naming reproducible.
        """
        if name is None:
            name = f"sample_{self._count:03d}"

        coords = np.ascontiguousarray(fields["coords"], dtype=np.float64)
        bc_mask = np.asarray(fields["bc_mask"], dtype=bool)
        c_lattice = np.ascontiguousarray(fields["c_lattice"], dtype=np.float64)
        sigma_h = np.ascontiguousarray(fields["hydrostatic_stress"], dtype=np.float64)
        grad_sigma = np.ascontiguousarray(fields["grad_sigma"], dtype=np.float64)
        time_points = np.ascontiguousarray(fields["time_points"], dtype=np.float64)

        n_dof = coords.shape[0]
        n_steps = c_lattice.shape[0]
        if bc_mask.shape != (n_dof,):
            raise ValueError(f"bc_mask shape {bc_mask.shape} != ({n_dof},)")
        if c_lattice.shape != (n_steps, n_dof):
            raise ValueError(f"c_lattice shape {c_lattice.shape} != ({n_steps}, {n_dof})")
        if sigma_h.shape != (n_dof,):
            raise ValueError(f"hydrostatic_stress shape {sigma_h.shape} != ({n_dof},)")
        if grad_sigma.shape != (n_dof, 2):
            raise ValueError(f"grad_sigma shape {grad_sigma.shape} != ({n_dof}, 2)")
        if time_points.shape != (n_steps,):
            raise ValueError(f"time_points shape {time_points.shape} != ({n_steps},)")

        if check_nonnegative:
            cmin = float(np.nanmin(c_lattice))
            if cmin < -1e-12:
                raise RuntimeError(
                    f"Negative c_L (min={cmin:.3e}) in sample {name!r}; the "
                    "lumped-mass formulation should prevent ringing."
                )

        # Create the subgroup only after validation passes (no stale group on error).
        grp = self.root.create_group(name)

        cd, cs = self.chunk_dof, self.chunk_steps
        # Zarr v3 infers dtype from `data`; passing both `data` and `dtype` errors.
        self._create_array(grp, "coords", coords, chunks=(min(cd, n_dof), 2))
        self._create_array(grp, "bc_mask", bc_mask, chunks=(min(cd, n_dof),))
        self._create_array(grp, "c_lattice", c_lattice,
                           chunks=(min(cs, n_steps), min(cd, n_dof)))
        self._create_array(grp, "hydrostatic_stress", sigma_h, chunks=(min(cd, n_dof),))
        self._create_array(grp, "grad_sigma", grad_sigma, chunks=(min(cd, n_dof), 2))
        self._create_array(grp, "time_points", time_points, chunks=(min(cs, n_steps),))

        grp.attrs["P"] = float(P)
        grp.attrs["T"] = float(T)
        grp.attrs["K_I"] = float(K_I)
        grp.attrs["r_notch"] = float(r_notch)
        grp.attrs["dislocation_type"] = str(dislocation_type)

        self._count += 1
        return name

    # ------------------------------------------------------------------
    # Introspection.
    # ------------------------------------------------------------------
    def sample_names(self):
        """Sorted list of sample subgroup names written so far."""
        keys = list(self.root.group_keys()) if hasattr(self.root, "group_keys") else list(self.root.keys())
        return sorted(keys)

    def __len__(self):
        return self._count


__all__ = ("ZarrDataWriter",)
