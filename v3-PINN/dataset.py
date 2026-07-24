"""Zarr v3 multi-sample dataset loader for the V3 PINN.

Discovers every ``sample_*`` subgroup inside a V2 Zarr v3 store and maps the
full ``(sample, dof, time)`` index space into a flat, O(1)-indexable tensor
representation for ``torch.utils.data.DataLoader``.

Artifact mitigation (CRITICAL):
    SKIP_FIRST = 2  : drop the first 2 time steps of ``c_lattice`` (transient
                       ringing from the Dirichlet ramp-up).
    CLIP_NEG  = True : clip any remaining negative ``c_lattice`` to 0.0.
    Normalisation   : divide ``c_lattice`` by the theoretical ``C_s(P, T)``
                       from the Peng-Robinson EOS so the network learns the
                       normalised field ``c_L / C_s`` (BC value == 1.0).

Each item is a single ``(sample, dof, time)`` point yielding:
    coords (2,), time (1,), P (1,), T (1,), sigma_h (1,),
    grad_sigma (2,), c_lattice (1,) [normalised], bc_mask (1,) [bool].

PyTorch and Zarr are imported lazily.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# Default artifact-mitigation flags (overridable per-dataset).
SKIP_FIRST: int = 2
CLIP_NEG: bool = True


def _import_torch():
    import torch  # type: ignore[import-not-found]
    return torch


def _import_zarr():
    import zarr  # type: ignore[import-not-found]
    return zarr


def _open_group(store):
    zarr = _import_zarr()
    if isinstance(store, zarr.Group):
        return store
    return zarr.open_group(store, mode="r")


class ZarrPINNDataset:
    """Flat ``(sample, dof, time)`` view over a V2 Zarr v3 sweep dataset.

    Parameters
    ----------
    store : str or zarr.Group
        Path/URL/store/group object pointing at the V2 dataset root.
    skip_first : int, optional
        Number of leading time steps to drop from ``c_lattice`` (default 2).
    clip_neg : bool, optional
        Clip remaining negative ``c_lattice`` to 0.0 (default True).
    """

    def __init__(self, store, *, skip_first: int = SKIP_FIRST,
                 clip_neg: bool = CLIP_NEG):
        torch = _import_torch()
        self._torch = torch
        self.skip_first = int(skip_first)
        self.clip_neg = bool(clip_neg)

        # Import the V2 thermodynamics for C_s normalisation. The V2 directory
        # is a sibling of v3-PINN; add it to sys.path if needed.
        import os, sys
        _v2 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v2-data-generation"))
        if _v2 not in sys.path:
            sys.path.insert(0, _v2)
        import constants as C  # type: ignore[import-not-found]
        import hydrogen_thermo as ht  # type: ignore[import-not-found]
        self._C = C
        self._ht = ht

        self.root = _open_group(store)
        self.sample_names: List[str] = sorted(
            self.root.group_keys() if hasattr(self.root, "group_keys") else list(self.root.keys())
        )
        if not self.sample_names:
            raise ValueError(f"No sample_* subgroups found in {store!r}.")

        # Per-sample in-memory tensors (no cross-time replication of coords).
        self._coords: List[np.ndarray] = []
        self._sigma_h: List[np.ndarray] = []
        self._grad_sigma: List[np.ndarray] = []
        self._bc_mask: List[np.ndarray] = []
        self._c_lattice: List[np.ndarray] = []   # (n_steps', n_dof) normalised
        self._time_points: List[np.ndarray] = [] # (n_steps',)
        self._P: List[float] = []
        self._T: List[float] = []
        self._cs: List[float] = []               # normaliser C_s(P,T) per sample

        # Flat-index boundaries: cumulative (n_dof * n_steps') over samples.
        counts: List[int] = []
        for name in self.sample_names:
            grp = self.root[name]
            coords = np.asarray(grp["coords"], dtype=np.float64)        # (n_dof, 2)
            bc_mask = np.asarray(grp["bc_mask"], dtype=bool)             # (n_dof,)
            c_lattice = np.asarray(grp["c_lattice"], dtype=np.float64)  # (n_steps, n_dof)
            sigma_h = np.asarray(grp["hydrostatic_stress"], dtype=np.float64)  # (n_dof,)
            grad_sigma = np.asarray(grp["grad_sigma"], dtype=np.float64)       # (n_dof, 2)
            time_points = np.asarray(grp["time_points"], dtype=np.float64)     # (n_steps,)

            P = float(grp.attrs["P"])
            T = float(grp.attrs["T"])
            c_s = float(ht.surface_concentration(T, P))

            # Artifact mitigation.
            k = self.skip_first
            if 0 < k < c_lattice.shape[0]:
                c_lattice = c_lattice[k:]
                time_points = time_points[k:]
            if self.clip_neg:
                c_lattice = np.clip(c_lattice, 0.0, None)
            # Normalise by C_s(P, T); guard against C_s == 0.
            c_lattice = c_lattice / c_s if c_s > 0 else c_lattice

            self._coords.append(np.ascontiguousarray(coords))
            self._sigma_h.append(np.ascontiguousarray(sigma_h))
            self._grad_sigma.append(np.ascontiguousarray(grad_sigma))
            self._bc_mask.append(np.ascontiguousarray(bc_mask))
            self._c_lattice.append(np.ascontiguousarray(c_lattice))
            self._time_points.append(np.ascontiguousarray(time_points))
            self._P.append(P)
            self._T.append(T)
            self._cs.append(c_s)

            n_dof = coords.shape[0]
            n_steps = c_lattice.shape[0]
            counts.append(n_dof * n_steps)

        self._counts = np.asarray(counts, dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(self._counts)])
        self._n_total = int(self._offsets[-1])

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        if idx < 0:
            idx += self._n_total
        if not 0 <= idx < self._n_total:
            raise IndexError(idx)
        torch = self._torch
        # Locate the sample via cumulative offsets, then (time, dof) within it.
        # Flat layout is time-major: local = time * n_dof + dof.
        s = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local = idx - int(self._offsets[s])
        n_dof = self._coords[s].shape[0]
        time, dof = divmod(local, n_dof)

        f32 = torch.float32
        return {
            "coords": torch.as_tensor(self._coords[s][dof], dtype=f32),
            "time": torch.as_tensor(self._time_points[s][time], dtype=f32).reshape(1),
            "P": torch.as_tensor(self._P[s], dtype=f32).reshape(1),
            "T": torch.as_tensor(self._T[s], dtype=f32).reshape(1),
            "sigma_h": torch.as_tensor(self._sigma_h[s][dof], dtype=f32).reshape(1),
            "grad_sigma": torch.as_tensor(self._grad_sigma[s][dof], dtype=f32),
            "c_lattice": torch.as_tensor(self._c_lattice[s][time, dof], dtype=f32).reshape(1),
            "bc_mask": torch.as_tensor(self._bc_mask[s][dof], dtype=torch.bool).reshape(1),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        """Stack a list of point dicts into a batched dict of tensors."""
        torch = _import_torch()
        out = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            out[key] = torch.stack(vals, dim=0) if key != "bc_mask" \
                else torch.stack(vals, dim=0)
        return out

    # ------------------------------------------------------------------
    # Introspection helpers.
    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return len(self.sample_names)

    def cs(self, sample: int) -> float:
        """Theoretical ``C_s(P, T)`` normaliser for a given sample index."""
        return self._cs[int(sample)]

    def input_stats(self):
        """Per-input ``(mean, std)`` standardisation statistics (length 6).

        Computed over all samples' coords, time_points (post-skip), P, T and
        hydrostatic stress. Use these to construct a :class:`HydrogenPINN` whose
        internal standardisation matches the dataset, avoiding tanh saturation
        from raw physical scales.
        """
        import numpy as _np
        # coords (x, y): flatten over samples & dofs.
        xy = _np.concatenate([c.reshape(-1) for c in self._coords])  # (sum n_dof * 2,)
        xy = xy.reshape(-1, 2)
        # time: flatten over samples & kept steps.
        tt = _np.concatenate([tp.reshape(-1) for tp in self._time_points])
        # P, T: per-sample scalars.
        P = _np.asarray(self._P, dtype=_np.float64)
        T = _np.asarray(self._T, dtype=_np.float64)
        # sigma_h: flatten over samples & dofs.
        sh = _np.concatenate([s.reshape(-1) for s in self._sigma_h])
        mean = _np.array([xy[:, 0].mean(), xy[:, 1].mean(), tt.mean(),
                          P.mean(), T.mean(), sh.mean()], dtype=_np.float64)
        std = _np.array([xy[:, 0].std(), xy[:, 1].std(), tt.std(),
                         P.std(), T.std(), sh.std()], dtype=_np.float64)
        # Constant features (std ~ 0) -> std = 1 so they normalise to 0 cleanly
        # instead of dividing by a tiny epsilon.
        std[std < 1e-12] = 1.0
        return mean, std


__all__ = ("ZarrPINNDataset", "SKIP_FIRST", "CLIP_NEG")
