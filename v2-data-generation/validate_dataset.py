"""Rigorous QA validation gate for the V2 Zarr v3 sweep dataset.

Scans the generated Zarr store at ``v2-data-generation/data/dataset.zarr`` and
acts as a strict Quality-Assurance gate before V3 PINN training:

1. Structural checks  : all ``N`` ``sample_NNN`` groups exist and each contains
                        the 6 required arrays (``coords``, ``bc_mask``,
                        ``c_lattice``, ``hydrostatic_stress``, ``grad_sigma``,
                        ``time_points``) with no NaN/Inf and consistent shapes.
2. Physical admissibility : the negative-concentration fraction must be EXACTLY
                        0.0% (lumped-mass guarantee). Any ``c_lattice < 0`` is a
                        CRITICAL error that aborts the gate.
3. Thermodynamic variance : the boundary concentration (``c_lattice`` on
                        ``bc_mask`` nodes == the Dirichlet value ``C_s(P,T)``)
                        must scale with pressure and temperature. A 70 MPa /
                        423.15 K sample MUST have a higher boundary
                        concentration than a 35 MPa / 298.15 K sample, proving
                        the dynamic Peng-Robinson fugacity is wired in.
4. Report            : clean terminal report (Pass/Fail metrics, min/max
                        stress, mean array shapes).

Paths are resolved relative to the ``v2-data-generation`` folder. Zarr is
imported lazily.

Usage::

    python v2-data-generation/validate_dataset.py
    python v2-data-generation/validate_dataset.py --expected 1000
    python v2-data-generation/validate_dataset.py --store /path/to/dataset.zarr
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import constants as C
import hydrogen_thermo as ht

# Default store location, resolved relative to this folder.
DEFAULT_STORE = os.path.join(HERE, "data", "dataset.zarr")

# The six required per-sample arrays and their expected ndim.
REQUIRED_ARRAYS = {
    "coords": 2,            # (n_dof, 2)
    "bc_mask": 1,            # (n_dof,)
    "c_lattice": 2,         # (n_steps, n_dof)
    "hydrostatic_stress": 1,  # (n_dof,)
    "grad_sigma": 2,        # (n_dof, 2)
    "time_points": 1,       # (n_steps,)
}

# Reference (P, T) corners used by the thermodynamic-variance check.
CORNER_HIGH = (70.0e6, 423.15)   # high P, high T -> high C_s
CORNER_LOW = (35.0e6, 298.15)    # low P, low T  -> low C_s


def _import_zarr():
    try:
        import zarr  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "zarr>=3.0 is required to validate the V2 dataset. "
            "Install it with `pip install zarr>=3.0`."
        ) from exc
    return zarr


class ValidationError(RuntimeError):
    """Raised when the dataset fails a critical QA check."""


def _open_root(store):
    zarr = _import_zarr()
    if not os.path.exists(store) and not str(store).startswith(("http", "s3", "gs")):
        raise ValidationError(f"Store path does not exist: {store!r}")
    return zarr.open_group(store, mode="r")


def _sample_names(root):
    if hasattr(root, "group_keys"):
        return sorted(list(root.group_keys()))
    return sorted(list(root.keys()))


def _boundary_concentration(grp) -> float:
    """Max ``c_lattice`` on the notch-surface (``bc_mask``) DOFs.

    On Dirichlet DOFs the solver pins ``c_L = C_s(P, T)``, so this is the clean
    real-gas surface-concentration signal (free of stress-assisted enrichment).
    Returns ``nan`` if the sample has no notch DOFs.
    """
    c = np.asarray(grp["c_lattice"], dtype=np.float64)
    mask = np.asarray(grp["bc_mask"], dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.nanmax(c[:, mask]))


def validate(store: str, expected: int = 1000) -> dict:
    """Run the full QA gate. Returns a metrics dict; raises on critical failure."""
    root = _open_root(store)
    names = _sample_names(root)

    metrics = {
        "store": store, "expected": expected, "found": len(names),
        "structural_pass": False, "nonneg_pass": False, "thermo_pass": False,
        "neg_fraction": float("nan"), "min_stress": float("nan"),
        "max_stress": float("nan"), "mean_shapes": {}, "missing_arrays": [],
        "nan_inf_samples": [], "extra_samples": [], "missing_samples": [],
        "c_s_low": float("nan"), "c_s_high": float("nan"),
    }

    # --- 1. Structural: exact count + contiguous sample_NNN naming -------------
    expected_names = [f"sample_{i:03d}" for i in range(expected)]
    metrics["missing_samples"] = [n for n in expected_names if n not in names]
    metrics["extra_samples"] = [n for n in names if n not in expected_names]
    if metrics["missing_samples"] or metrics["extra_samples"] or len(names) != expected:
        raise ValidationError(
            f"Structural count mismatch: expected {expected} groups, found "
            f"{len(names)} (missing={metrics['missing_samples'][:5]}... "
            f"extra={metrics['extra_samples'][:5]}...)."
        )

    shape_counts = {k: {} for k in REQUIRED_ARRAYS}
    min_sigma = np.inf
    max_sigma = -np.inf
    total_neg = 0
    total_c = 0
    nan_inf_samples = []
    missing_arrays = []

    c_s_low = np.inf   # boundary C_s for the low  (35 MPa, 298.15 K) corner
    c_s_high = -np.inf  # boundary C_s for the high (70 MPa, 423.15 K) corner

    for name in expected_names:
        grp = root[name]
        # Required arrays present?
        for arr in REQUIRED_ARRAYS:
            if arr not in grp:
                missing_arrays.append((name, arr))
                continue
        if missing_arrays:
            continue  # shape/NaN checks skipped; reported below

        # Shape consistency + NaN/Inf scan.
        n_dof = int(np.asarray(grp["coords"], dtype=np.float64).shape[0])
        n_steps = int(np.asarray(grp["c_lattice"], dtype=np.float64).shape[0])
        shape_ok = True
        for arr, ndim in REQUIRED_ARRAYS.items():
            a = np.asarray(grp[arr])
            shape_counts[arr][tuple(a.shape)] = shape_counts[arr].get(tuple(a.shape), 0) + 1
            if a.ndim != ndim:
                shape_ok = False
        # Cross-array consistency.
        if (np.asarray(grp["bc_mask"]).shape != (n_dof,)
                or np.asarray(grp["hydrostatic_stress"]).shape != (n_dof,)
                or np.asarray(grp["grad_sigma"]).shape != (n_dof, 2)
                or np.asarray(grp["c_lattice"]).shape != (n_steps, n_dof)
                or np.asarray(grp["time_points"]).shape != (n_steps,)):
            shape_ok = False
        if not shape_ok:
            missing_arrays.append((name, "__shape__"))

        # NaN/Inf scan across all numeric arrays.
        for arr in ("coords", "c_lattice", "hydrostatic_stress", "grad_sigma", "time_points"):
            a = np.asarray(grp[arr], dtype=np.float64)
            if not np.all(np.isfinite(a)):
                nan_inf_samples.append(name)
                break

        # Stress extrema.
        sh = np.asarray(grp["hydrostatic_stress"], dtype=np.float64)
        min_sigma = min(min_sigma, float(np.nanmin(sh)))
        max_sigma = max(max_sigma, float(np.nanmax(sh)))

        # --- 2. Physical admissibility: zero negatives ------------------------
        c = np.asarray(grp["c_lattice"], dtype=np.float64)
        total_neg += int(np.sum(c < -1e-15))
        total_c += c.size

        # --- 3. Thermodynamic variance: boundary C_s vs (P, T) corners --------
        P = float(grp.attrs.get("P", np.nan))
        T = float(grp.attrs.get("T", np.nan))
        bc = _boundary_concentration(grp)
        if np.isfinite(bc):
            if abs(P - CORNER_LOW[0]) < 1.0 and abs(T - CORNER_LOW[1]) < 1.0:
                c_s_low = min(c_s_low, bc)
            if abs(P - CORNER_HIGH[0]) < 1.0 and abs(T - CORNER_HIGH[1]) < 1.0:
                c_s_high = max(c_s_high, bc)

    metrics["missing_arrays"] = missing_arrays
    metrics["nan_inf_samples"] = sorted(set(nan_inf_samples))
    metrics["min_stress"] = float(min_sigma) if np.isfinite(min_sigma) else float("nan")
    metrics["max_stress"] = float(max_sigma) if np.isfinite(max_sigma) else float("nan")
    # Mean (most common) shape per array.
    metrics["mean_shapes"] = {
        arr: max(counts, key=counts.get) if counts else None
        for arr, counts in shape_counts.items()
    }

    # Structural pass: no missing/extra, no missing arrays, no NaN/Inf, shapes ok.
    metrics["structural_pass"] = (
        not metrics["missing_arrays"] and not metrics["nan_inf_samples"]
        and len(metrics["missing_samples"]) == 0 and len(metrics["extra_samples"]) == 0
    )

    # --- 2. Non-negativity: EXACTLY 0.0% negative -----------------------------
    metrics["neg_fraction"] = (total_neg / total_c) if total_c else float("nan")
    metrics["nonneg_pass"] = (metrics["neg_fraction"] == 0.0)
    if not metrics["nonneg_pass"]:
        raise ValidationError(
            f"CRITICAL: negative c_lattice detected: {total_neg} negative "
            f"values (fraction={metrics['neg_fraction']*100:.6f}%). The "
            "lumped-mass formulation must yield EXACTLY 0.0% negatives."
        )

    # --- 3. Thermodynamic variance: high corner > low corner ------------------
    metrics["c_s_low"] = float(c_s_low) if np.isfinite(c_s_low) else float("nan")
    metrics["c_s_high"] = float(c_s_high) if np.isfinite(c_s_high) else float("nan")
    if not (np.isfinite(c_s_low) and np.isfinite(c_s_high)):
        metrics["thermo_pass"] = False
        metrics["thermo_msg"] = ("Could not find both (35 MPa, 298.15 K) and "
                                 "(70 MPa, 423.15 K) corner samples.")
    elif c_s_high > c_s_low:
        metrics["thermo_pass"] = True
        metrics["thermo_msg"] = (
            f"boundary C_s(70MPa,423K)={c_s_high:.4e} > "
            f"C_s(35MPa,298K)={c_s_low:.4e} mol/m^3")
    else:
        metrics["thermo_pass"] = False
        metrics["thermo_msg"] = (
            f"Thermodynamic variance FAILED: boundary C_s(70MPa,423K)="
            f"{c_s_high:.4e} is NOT higher than C_s(35MPa,298K)={c_s_low:.4e}. "
            "Dynamic Peng-Robinson fugacity is not scaling with P, T.")
        raise ValidationError(metrics["thermo_msg"])

    return metrics


# ---------------------------------------------------------------------------
# Terminal report.
# ---------------------------------------------------------------------------
def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report(metrics: dict) -> int:
    print("=" * 72)
    print("V2 dataset QA validation report")
    print("=" * 72)
    print(f"  Store               : {metrics['store']}")
    print(f"  Sample groups       : {metrics['found']} (expected {metrics['expected']})")
    print("-" * 72)
    print(f"  1. Structural       : {_pf(metrics['structural_pass'])}")
    if metrics["missing_samples"]:
        print(f"     missing samples  : {metrics['missing_samples'][:5]} ...")
    if metrics["extra_samples"]:
        print(f"     extra samples    : {metrics['extra_samples'][:5]} ...")
    if metrics["missing_arrays"]:
        print(f"     missing arrays   : {metrics['missing_arrays'][:5]} ...")
    if metrics["nan_inf_samples"]:
        print(f"     NaN/Inf samples  : {metrics['nan_inf_samples'][:5]} ...")
    print(f"     mean array shapes :")
    for arr, shp in metrics["mean_shapes"].items():
        print(f"        {arr:<20s}: {shp}")
    print("-" * 72)
    print(f"  2. Non-negativity    : {_pf(metrics['nonneg_pass'])}")
    print(f"     negative fraction : {metrics['neg_fraction']*100:.6f}% "
          f"(must be EXACTLY 0.000000%)")
    print("-" * 72)
    print(f"  3. Thermodynamic var.: {_pf(metrics['thermo_pass'])}")
    print(f"     {metrics.get('thermo_msg', '')}")
    print("-" * 72)
    print(f"  Hydrostatic stress   : min = {metrics['min_stress']:.4e} Pa, "
          f"max = {metrics['max_stress']:.4e} Pa")
    print("=" * 72)
    all_pass = (metrics["structural_pass"] and metrics["nonneg_pass"]
                 and metrics["thermo_pass"])
    print(f"  OVERALL QA GATE      : {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)
    return 0 if all_pass else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V2 Zarr dataset QA validator.")
    parser.add_argument("--store", default=DEFAULT_STORE,
                        help=f"Zarr store path (default: {DEFAULT_STORE}).")
    parser.add_argument("--expected", type=int, default=1000,
                        help="Expected number of sample groups (default 1000).")
    args = parser.parse_args(argv)

    try:
        metrics = validate(args.store, expected=args.expected)
    except ValidationError as exc:
        print("=" * 72)
        print(f"CRITICAL QA FAILURE: {exc}")
        print("=" * 72)
        return 1
    return report(metrics)


if __name__ == "__main__":
    raise SystemExit(main())
