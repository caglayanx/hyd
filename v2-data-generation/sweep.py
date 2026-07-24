"""V2 data-sweep orchestrator: 1000-sample FEniCSx mech-hydrogen dataset.

Builds the parameter grid and runs :class:`solver.HydrogenDiffusionSolver` for
every combination, exporting the transient fields to a chunked Zarr v3 store
at ``v2-data-generation/data/dataset.zarr`` (path resolved relative to this
folder, independent of the current working directory).

Parameter grid (exactly 1000 samples = 4 x 2 x 5 x 5 x 5):
    * 4 dislocation types : arc, straight, edge, dipole
    * 2 external pressures: 35 MPa, 70 MPa
    * 5 notch radii      : 100..1000 um (log scale)
    * 5 stress intensities K_I : 1e5..1e7 Pa*sqrt(m) (log scale)
    * 5 temperatures     : 298.15..423.15 K (linear scale)

Each sample runs inside a try/except so a single failing sample is logged and
skipped without aborting the 1000-sample batch. The notch-surface Dirichlet
value ``C_s(P, T)`` is evaluated dynamically inside the solver from the
Peng-Robinson EOS (:mod:`hydrogen_thermo`).

DOLFINx / UFL / PETSc are imported lazily so this script imports cleanly without
the FEniCSx stack (e.g. for ``--grid`` inspection).

Usage::

    python v2-data-generation/sweep.py            # full 1000-sample sweep
    python v2-data-generation/sweep.py --limit 4  # smoke test (first 4 samples)
    python v2-data-generation/sweep.py --grid     # print the grid, do not solve
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import shutil
import sys
import time
import traceback
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

# Make this directory importable for sibling modules (constants, hydrogen_thermo,
# solver, zarr_writer). v2-data-generation/ is a flat, self-contained module set.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import constants as C
import hydrogen_thermo as ht
from zarr_writer import ZarrDataWriter

# ---------------------------------------------------------------------------
# Output location: resolved strictly relative to this folder.
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(HERE, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.zarr")

# ---------------------------------------------------------------------------
# Parameter grid (exact matrix from the task spec).
# ---------------------------------------------------------------------------
DISLOCATION_TYPES = ("arc", "straight", "edge", "dipole")
PRESSURES = (35.0e6, 70.0e6)                                   # 2 pressures (Pa)
NOTCH_RADII = np.logspace(np.log10(100.0e-6), np.log10(1000.0e-6), 5)   # 5 radii (m)
K_I_GRID = np.logspace(np.log10(1.0e5), np.log10(1.0e7), 5)             # 5 K_I (Pa sqrt(m))
TEMPERATURES = np.linspace(298.15, 423.15, 5)                  # 5 temperatures (K)

# Time-stepping defaults.
N_STEPS = 64
DT = 1.0e-3          # s

# The six per-sample arrays written to each ``sample_NNN`` subgroup (used by the
# shard merge step).
REQUIRED_ARRAYS = ("coords", "bc_mask", "c_lattice",
                    "hydrostatic_stress", "grad_sigma", "time_points")


def parameter_grid():
    """Yield every (dislocation_type, P, r_notch, K_I, T) tuple in the grid.

    Order: dislocation_type (slowest) -> P -> r_notch -> K_I -> T (fastest), so
    that the sample index is deterministic and reproducible.
    """
    for dtype in DISLOCATION_TYPES:
        for P in PRESSURES:
            for r_notch in NOTCH_RADII:
                for K_I in K_I_GRID:
                    for T in TEMPERATURES:
                        yield dtype, float(P), float(r_notch), float(K_I), float(T)


def grid_size() -> int:
    """Total number of samples in the grid (4*2*5*5*5 = 1000)."""
    return (len(DISLOCATION_TYPES) * len(PRESSURES) * len(NOTCH_RADII)
            * len(K_I_GRID) * len(TEMPERATURES))


# ---------------------------------------------------------------------------
# Mesh + stress-field builder (DOLFINx v0.11). Imported lazily.
# ---------------------------------------------------------------------------
def _import_dolfinx():
    try:
        import dolfinx  # type: ignore[import-not-found]
        import dolfinx.fem  # type: ignore[import-not-found]
        import dolfinx.mesh  # type: ignore[import-not-found]
        import ufl  # type: ignore[import-not-found]
        from petsc4py import PETSc  # type: ignore[import-not-found]
        from dolfinx import default_scalar_type  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only with the stack
        raise ImportError(
            "DOLFINx v0.11 is required to run the V2 sweep. Install it via "
            "conda (`conda install -c conda-forge fenics-dolfinx`) or apt."
        ) from exc
    return dolfinx, ufl, PETSc, default_scalar_type


def build_notched_domain(*, Lx: float = 2.0e-3, Ly: float = 1.0e-3,
                         n_cells_x: int = 48, n_cells_y: int = 24,
                         notch_id: int = 1):
    """Build a 2D rectangular notched specimen mesh and tag the notch surface.

    The notch surface is the left edge (``x = 0``); the notch root sits at
    mid-height ``(0, Ly/2)``. Returns ``(domain, facet_tags, notch_id, Lx, Ly)``.
    """
    dolfinx, _ufl, _PETSc, _dst = _import_dolfinx()
    from dolfinx import mesh as dmesh
    from mpi4py import MPI  # type: ignore[import-not-found]

    # Serial mesh (comm_self): the solver uses a self-comm GMRES KSP.
    domain = dmesh.create_rectangle(
        comm=MPI.comm_self, points=([0.0, 0.0], [Lx, Ly]),
        n=[n_cells_x, n_cells_y], cell_type=dmesh.CellType.triangle,
    )

    # Tag the left-edge facets (x = 0) as the notch surface (notch_id).
    tdim = domain.topology.dim
    fdim = tdim - 1
    left_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], 0.0))
    facet_tags = dmesh.meshtags(
        domain, fdim, left_facets,
        np.full(left_facets.shape, notch_id, dtype=np.int32))
    return domain, facet_tags, notch_id, Lx, Ly


def build_sigma_h_field(domain, *, K_I: float, r_notch: float,
                         dislocation_type: str, Ly: float):
    """Prescribed hydrostatic stress field ``sigma_h`` (Pa) as a DOLFINx Function.

    Combines a regularised mode-I near-tip K-field with a dislocation-type
    self-stress term, evaluated pointwise at the P1 DOF coordinates:

        sigma_h(x,y) = K_I / sqrt(2 pi (r + r_notch)) + sigma_self(x,y; type)

    where ``r`` is the distance from the notch root ``(0, Ly/2)`` and the
    dislocation-type term encodes the Mura self-stress family:

        straight : no curvature self-stress (bare K-field)
        edge     : + A / (|r - r_d| + r_c)            (single edge dislocation)
        arc      : + sigma_LT(kappa) * ln(R(kappa)/r_c),  kappa = 1/r_notch
        dipole   : + A/(|r-r_d1|+r_c) - A/(|r-r_d2|+r_c) (edge dipole)
    """
    dolfinx, _ufl, _PETSc, _dst = _import_dolfinx()

    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    sigma_h = dolfinx.fem.Function(V)
    coords = V.tabulate_dof_coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    tip = np.array([0.0, 0.5 * Ly])
    r = np.sqrt((x - tip[0]) ** 2 + (y - tip[1]) ** 2)
    # Regularised mode-I hydrostatic K-field (tensile).
    sh = K_I / np.sqrt(2.0 * np.pi * (r + r_notch))

    A = C.MU * C.BURGERS_B / (2.0 * np.pi * (1.0 - C.NU))   # edge-field amplitude
    r_c = C.CORE_RADIUS
    r_d = np.array([0.5e-3, 0.5 * Ly])                       # dislocation centre (m)

    if dislocation_type == "straight":
        pass  # bare K-field, no curvature self-stress
    elif dislocation_type == "edge":
        rd = np.sqrt((x - r_d[0]) ** 2 + (y - r_d[1]) ** 2)
        sh = sh + A / (rd + r_c)
    elif dislocation_type == "arc":
        kappa = 1.0 / r_notch                              # curvature R = r_notch
        s_lt = C.MU * C.BURGERS_B * kappa / (2.0 * (1.0 - C.NU))
        sh = sh + s_lt * np.log((1.0 / kappa) / r_c)        # Mura hydrostatic self-stress
    elif dislocation_type == "dipole":
        r_d2 = np.array([0.5e-3, 0.25 * Ly])
        rd1 = np.sqrt((x - r_d[0]) ** 2 + (y - r_d[1]) ** 2)
        rd2 = np.sqrt((x - r_d2[0]) ** 2 + (y - r_d2[1]) ** 2)
        sh = sh + A / (rd1 + r_c) - A / (rd2 + r_c)
    else:
        raise ValueError(f"Unknown dislocation_type {dislocation_type!r}.")

    sigma_h.x.array[:] = np.ascontiguousarray(sh, dtype=dolfinx.default_scalar_type)
    return sigma_h


# ---------------------------------------------------------------------------
# Single-sample run.
# ---------------------------------------------------------------------------
def run_one_sample(writer: ZarrDataWriter, *, dislocation_type: str, P: float,
                    r_notch: float, K_I: float, T: float,
                    n_steps: int = N_STEPS, dt: float = DT,
                    name: Optional[str] = None) -> str:
    """Build the mesh + stress field, run the solver, and write one Zarr sample.

    Returns the subgroup name (``sample_NNN``). Raises on failure (the caller
    wraps this in try/except so one failure does not abort the batch). ``name``
    fixes the subgroup name (deterministic grid index) for MPI/resume safety.
    """
    from solver import HydrogenDiffusionSolver

    domain, facet_tags, notch_id, _Lx, Ly = build_notched_domain()
    sigma_h = build_sigma_h_field(
        domain, K_I=K_I, r_notch=r_notch,
        dislocation_type=dislocation_type, Ly=Ly)

    solver = HydrogenDiffusionSolver(
        domain, facet_tags, notch_id, sigma_h, P=P, T=T)
    c_lattice, time_points = solver.run(n_steps=n_steps, dt=dt)
    fields = solver.export_fields()
    fields["c_lattice"] = c_lattice
    fields["time_points"] = time_points

    return writer.write_sample(
        fields, P=P, T=T, K_I=K_I, r_notch=r_notch,
        dislocation_type=dislocation_type, name=name)


# ---------------------------------------------------------------------------
# MPI + checkpoint/resume helpers.
# ---------------------------------------------------------------------------
def _get_mpi():
    """Return ``(comm, rank, size, mpi)``. Falls back to a 1-rank stub if mpi4py
    is unavailable, so the script runs both under ``mpirun`` and plain ``python``.
    ``mpi`` is the :mod:`mpi4py.MPI` module (or ``None`` in the stub fallback)."""
    try:
        from mpi4py import MPI  # type: ignore[import-not-found]
        return MPI.COMM_WORLD, MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, MPI
    except Exception:  # pragma: no cover - single-process fallback
        class _Stub:
            rank = 0
            size = 1
            def bcast(self, obj, root=0):
                return obj
            def barrier(self):
                pass
            def Abort(self, code=1):
                sys.exit(code)
        return _Stub(), 0, 1, None


def _existing_sample_indices(store: str) -> Set[int]:
    """Return the set of grid indices already present in ``store`` as
    ``sample_NNN`` groups (for checkpoint/resume). Empty if the store is absent."""
    try:
        import zarr  # type: ignore[import-not-found]
    except ImportError:
        return set()
    if not os.path.exists(store):
        return set()
    try:
        root = zarr.open_group(store, mode="r")
    except Exception:
        return set()
    keys = list(root.group_keys()) if hasattr(root, "group_keys") else list(root.keys())
    idx = set()
    for k in keys:
        if isinstance(k, str) and k.startswith("sample_"):
            try:
                idx.add(int(k.split("_", 1)[1]))
            except ValueError:
                continue
    return idx


def _remove_path(path: str) -> None:
    """Remove a file or directory tree if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _merge_shards(final_store: str, parts_dir: str, *, resume: bool) -> int:
    """Merge per-rank shard stores in ``parts_dir`` into the final ``final_store``.

    Each shard contains ``sample_NNN`` groups written by one MPI rank. The merge
    is performed serially by rank 0 into the final store (append mode on resume,
    fresh otherwise), skipping any group that already exists. Returns the number
    of groups merged.
    """
    import zarr  # type: ignore[import-not-found]
    mode = "a" if (resume and os.path.exists(final_store)) else "w"
    root = zarr.open_group(final_store, mode=mode)
    n_merged = 0
    shard_paths = sorted(glob.glob(os.path.join(parts_dir, "rank_*.zarr")))
    for shard_path in shard_paths:
        shard = zarr.open_group(shard_path, mode="r")
        keys = list(shard.group_keys()) if hasattr(shard, "group_keys") else list(shard.keys())
        for k in sorted(keys):
            if k in root:
                continue  # resume: keep existing
            src = shard[k]
            dst = root.create_group(k)
            for arr_name in REQUIRED_ARRAYS:
                a = np.asarray(src[arr_name])
                src_chunks = getattr(src[arr_name], "chunks", None)
                chunks = tuple(src_chunks) if src_chunks else tuple(a.shape)
                dst.create_array(arr_name, data=a, chunks=chunks)
            for ak, av in src.attrs.items():
                dst.attrs[ak] = av
            n_merged += 1
    return n_merged


# ---------------------------------------------------------------------------
# Batch driver (MPI-parallel, checkpoint/resume, rank-aware printing).
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V2 FEniCSx data sweep (MPI).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Run only the first N remaining samples (0 = all).")
    parser.add_argument("--grid", action="store_true",
                        help="Print the parameter grid and exit (no solve).")
    parser.add_argument("--out", default=DATASET_PATH,
                        help=f"Output Zarr path (default: {DATASET_PATH}).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sample_NNN groups already present in --out.")
    parser.add_argument("--keep-parts", action="store_true",
                        help="Keep per-rank shard stores after the merge.")
    args = parser.parse_args(argv)

    comm, rank, size, mpi = _get_mpi()

    # --- --grid: print only (rank 0), no solve --------------------------------
    if args.grid:
        if rank == 0:
            print(f"V2 parameter grid: {grid_size()} samples")
            print(f"  dislocation types: {DISLOCATION_TYPES}")
            print(f"  pressures (MPa) : {[P/1e6 for P in PRESSURES]}")
            print(f"  notch radii (um) : {[r*1e6 for r in NOTCH_RADII]}")
            print(f"  K_I (Pa sqrt m) : {list(K_I_GRID)}")
            print(f"  temperatures (K): {list(TEMPERATURES)}")
            print(f"  output path      : {args.out}")
            print(f"  MPI ranks         : {size}")
        return 0

    # --- Rank 0: compute the full grid + remaining indices (checkpoint) --------
    if rank == 0:
        full_grid = list(parameter_grid())              # list of (dtype,P,r,K,T)
        existing = _existing_sample_indices(args.out) if args.resume else set()
        if args.resume and existing:
            print(f"[rank 0] resume: {len(existing)} samples already present, "
                  f"skipping them.")
        remaining = [i for i in range(len(full_grid)) if i not in existing]
        if args.limit and args.limit > 0:
            remaining = remaining[:args.limit]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # Per-run shard directory (cleaned at start so stale shards don't leak in).
        parts_dir = args.out + ".parts"
        _remove_path(parts_dir)
        os.makedirs(parts_dir, exist_ok=True)
    else:
        full_grid = None
        remaining = None
        parts_dir = None

    # Broadcast the grid + remaining indices + parts_dir to every rank.
    full_grid = comm.bcast(full_grid, root=0)
    remaining = comm.bcast(remaining, root=0)
    parts_dir = comm.bcast(parts_dir, root=0)

    total = len(full_grid)
    n_remaining = len(remaining)

    if rank == 0:
        print("=" * 72)
        print(f"V2 data sweep: {n_remaining} remaining / {total} total samples "
              f"-> {args.out}  (MPI ranks = {size})")
        print("=" * 72)

    if n_remaining == 0:
        if rank == 0:
            print("[rank 0] Nothing to do: all samples already present.")
        comm.barrier()
        # Still merge any (empty) shards for consistency, then finish.
        if rank == 0:
            _merge_shards(args.out, parts_dir, resume=args.resume)
            if not args.keep_parts:
                _remove_path(parts_dir)
            print(f"[rank 0] Dataset: {args.out}")
        return 0

    # --- Strided distribution of remaining indices across ranks ----------------
    # remaining[rank], remaining[rank+size], ... -> good load balance for
    # similar-cost samples; each rank owns disjoint indices.
    my_indices = remaining[rank::size]

    # Each rank writes to its OWN shard store -> no concurrent-write races on a
    # single store; rank 0 merges all shards into the final store afterwards.
    shard_path = os.path.join(parts_dir, f"rank_{rank:04d}.zarr")
    writer = ZarrDataWriter(store=shard_path, mode="w")

    n_ok = n_fail = 0
    failures: List[Tuple[int, str, str]] = []
    t0 = time.time()
    for i in my_indices:
        dtype, P, r_notch, K_I, T = full_grid[i]
        tag = (f"sample {i:03d}/{total}: type={dtype} P={P/1e6:.0f}MPa "
               f"r={r_notch*1e6:.0f}um K_I={K_I:.1e} T={T:.1f}K")
        try:
            name = run_one_sample(writer, dislocation_type=dtype, P=P,
                                   r_notch=r_notch, K_I=K_I, T=T,
                                   name=f"sample_{i:03d}")
            n_ok += 1
            print(f"  [rank {rank}] [OK] {tag} -> {name}", flush=True)
        except Exception as exc:  # one failure must not crash the batch
            n_fail += 1
            failures.append((i, tag, repr(exc)))
            print(f"  [rank {rank}] [FAIL] {tag}: {exc!r}", flush=True)
            traceback.print_exc()

    # Gather per-rank ok/fail counts on rank 0 for the summary.
    local_counts = np.array([n_ok, n_fail], dtype=np.int64)
    counts = np.zeros(2, dtype=np.int64) if rank == 0 else None
    if mpi is not None and size > 1:
        comm.Reduce(local_counts, counts, op=mpi.SUM, root=0)
    else:
        counts = local_counts if rank == 0 else None

    # All ranks must finish writing before the merge.
    comm.barrier()

    if rank == 0:
        n_merged = _merge_shards(args.out, parts_dir, resume=args.resume)
        if not args.keep_parts:
            _remove_path(parts_dir)
        elapsed = time.time() - t0
        tot_ok = int(counts[0]) if counts is not None else n_ok
        tot_fail = int(counts[1]) if counts is not None else n_fail
        print("=" * 72)
        print(f"[rank 0] Sweep finished in {elapsed:.1f}s: {tot_ok} ok, "
              f"{tot_fail} failed (of {n_remaining} attempted); merged "
              f"{n_merged} groups into {args.out}.")
        print(f"[rank 0] Dataset: {args.out}")
        return 0 if tot_fail == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
