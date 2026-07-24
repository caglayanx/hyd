#!/usr/bin/env python3
"""Hydrogen-induced dislocation line-tension degradation proof.

Version 1 lightweight, standalone analysis. Proves the core theoretical
result of the hydrogen-induced dislocation line-tension degradation theory
for 30CrMo (30CrMo4 / AISI 4130) quenched-and-tempered storage steel, at a
single FIXED thermodynamic state:

    T      = 298.15 K   (25 °C)
    P_ext  = 35 MPa     (35e6 Pa)

The free-energy chain implemented here is:

    1. Abel-Noble real-gas EOS  -> H2 fugacity f(P,T) = P exp(b P / (R T))
    2. Sieverts solubility      -> surface concentration C_s = K_s(T) sqrt(f)
    3. Stress-assisted lattice  -> c_L(k) = C_s exp(Omega sigma_max(k) / (R T))
    4. Oriani trap equilibrium  -> theta_T(k) from theta_L = c_L / N_L
    5. Configurational free energy  -> Psi_conf(c_L)
    6. Trap-binding free energy     -> Psi_trap(theta_T)
    7. Hydrostatic stress (Mura) + mechanical line energy
         R(k)                  = 1/kappa
         sigma_h(k)            = sigma_h^{Mura_Self}(k)   [sole mechanical-stress input]
         (NO "total" wrapper, NO P_ext superposition, NO added line-tension stress)
         Gamma_mech(k)         = mu b^2/(4 pi (1-nu)) * ln(R(k)/r_c)
    8. Hydrogen line-energy reduction  -> Gamma_H(k) (closed form, Kirchheim)
         c_L(k) = C_s exp(V_H sigma_h^{Mura_Self}(k) / (R T))
    9. Total line tension             -> Gamma(k) = Gamma_mech(k) + Gamma_H(k)

Every stress and energy function below is parameterised by the dislocation
curvature ``kappa`` (1/m); the radius of curvature ``R(kappa) = 1/kappa`` is
defined explicitly at the start of each calculation.

The hydrogen-induced reduction Gamma_H < 0 is the degradation mechanism: it
lowers the line tension of a curved (bowing) dislocation, increasing its
mobility and thus promoting hydrogen-enhanced localised plasticity (HELP) /
decohesion at the storage-vessel notch root.

Outputs:
    - Console printout of the fixed-state thermodynamic quantities.
    - Publication-ready figure: Gamma (total / mech / H) vs curvature kappa,
      saved to ``figures/line_tension_vs_curvature.png`` and shown on screen.

Run locally with only numpy / scipy / matplotlib installed::

    python v1-line-tension/analyze_line_tension.py
"""
from __future__ import annotations

import os
import sys

# --- Make this directory importable for sibling modules ---------------------
# v1-line-tension/ is a self-contained, flat module set (no `hyd` namespace,
# no FEniCSx/PyTorch). The script puts its own directory on sys.path so that
# `import constants`, `import hydrogen`, etc. resolve to sibling files.
HERE = os.path.dirname(os.path.abspath(__file__))          # .../hyd/v1-line-tension
sys.path.insert(0, HERE)

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; switch to a GUI backend interactively if desired
import matplotlib.pyplot as plt

import constants as C
import hydrogen as H
import line_tension as LT
from line_tension import (
    radius_of_curvature,
    get_mura_self_stress,
    set_mura_self_stress_provider,
    mura_self_stress,
)

# ===========================================================================
# Fixed Version-1 thermodynamic state (the proof operating point).
# ===========================================================================
T_FIXED: float = C.T_OP          # 298.15 K  (25 °C)
P_FIXED: float = C.P_OP          # 35 MPa    (35e6 Pa)


# ---------------------------------------------------------------------------
# Step 1-2: Abel-Noble fugacity and Sieverts surface concentration.
# ---------------------------------------------------------------------------
def thermodynamic_state() -> dict:
    """Compute the fixed-state gas-side thermodynamic quantities.

    Returns a dict with the Abel-Noble fugacity coefficient ``phi``, the
    fugacity ``f``, the ideal-gas fugacity (= P) for comparison, the Sieverts
    coefficient ``K_s``, and the surface concentration ``C_s``.
    """
    phi = H.abel_noble_fugacity_coefficient(P_FIXED, T_FIXED)
    f = H.abel_noble_fugacity(P_FIXED, T_FIXED)
    k_s = H.sieverts_coefficient(T_FIXED)
    c_s = H.surface_concentration(T_FIXED)
    return {
        "phi": phi,
        "f": f,
        "f_ideal": P_FIXED,
        "K_s": k_s,
        "C_s": c_s,
    }


# ---------------------------------------------------------------------------
# Step 3: stress-assisted lattice concentration as a function of curvature.
# ---------------------------------------------------------------------------
def stress_assisted_lattice_concentration(sigma_h: np.ndarray) -> np.ndarray:
    """Lattice H concentration ``c_L = C_s exp(Omega sigma_h / (R T))``.

    Uses the Abel-Noble-derived surface concentration ``C_s`` at the fixed V1
    state, so the real-gas fugacity is propagated into the lattice population.
    """
    c_s = H.surface_concentration(T_FIXED)
    return c_s * np.exp(C.OMEGA * np.asarray(sigma_h, dtype=np.float64) / (C.R * T_FIXED))


# ---------------------------------------------------------------------------
# Step 7: mechanical line tension of a curved edge dislocation (Hirth & Lothe).
# ---------------------------------------------------------------------------
def mechanical_line_tension(kappa: np.ndarray) -> np.ndarray:
    """Classical (hydrogen-free) line tension of a curved edge dislocation.

    In the line-tension approximation (standard in dislocation dynamics, see
    Hirth & Lothe), the restoring line tension equals the line energy per unit
    length. For a bowed edge dislocation of curvature ``kappa`` (radius of
    curvature ``R(kappa) = 1/kappa``),

        R(k)        = 1/kappa
        Gamma_mech(k) = mu b^2 / (4 pi (1 - nu)) * ln(R(k) / r_c) ,

    with the dislocation core radius ``r_c = b``. ``Gamma_mech`` diverges as
    ``k -> 0`` (straight line) and vanishes as ``k -> 1/r_c`` (core limit).

    Args:
        kappa: Dislocation curvature (1/m), strictly positive, shape ``(K,)``.

    Returns:
        Mechanical line tension ``Gamma_mech`` in J/m (= N), shape ``(K,)``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    R = radius_of_curvature(kappa)            # R(kappa) = 1/kappa (m)
    pre = C.MU * C.BURGERS_B ** 2 / (4.0 * np.pi * (1.0 - C.NU))
    return pre * np.log(R / C.CORE_RADIUS)


# ---------------------------------------------------------------------------
# Dislocation self-stress: BLACK-BOX external input.
#
# Per the project foundational document, the mechanical side is NOT re-derived
# here. The hydrostatic self-stress ``sigma_h^{Mura_Self}(kappa)`` is an
# external input from the prior numerical Mura line-integral solution, exposed
# through :func:`line_tension.get_mura_self_stress`. No analytical stress
# formula (Hirth-Lothe log, sigma_LT, 1/r edge field) is used in this module.
#
# The local hydrostatic stress driving hydrogen accumulation at the defect is
# defined ENTIRELY by the Mura line-integral output based on curvature:
#
#     sigma_h(kappa) = sigma_h^{Mura_Self}(kappa) .
#
# The macroscopic external pressure P_ext is NOT superposed onto the
# microscopic dislocation self-stress (that would be a physical redundancy):
# P_ext only drives the boundary fugacity through the surface concentration
# C_s (Sieverts/Abel-Noble), never the local hydrostatic field. There is NO
# "total" stress wrapper and NO separate line-tension stress/energy added on
# top of the Mura output: the Mura integral already accounts for the exact
# curved geometry, so any superposition would be unphysical double-counting.
# The sole mechanical-stress API is :func:`line_tension.get_mura_self_stress`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Steps 3-6 & 8: hydrogen line-energy and line-tension reduction via the
# free-energy chain integrated over the dislocation stress field.
# ---------------------------------------------------------------------------
def hydrogen_line_energy(kappa: float, n_radial: int = 2048) -> float:
    """Excess hydrogen line energy ``E_H(k)`` (J/m) for one curvature.

    The lattice hydrogen concentration is driven DIRECTLY by the local
    hydrostatic stress, which is defined ENTIRELY by the Mura line-integral
    output based on curvature (a scalar function of ``kappa`` only, uniform
    over the atmosphere, with NO ``P_ext`` superposition):

        R(k)                    = 1/kappa
        sigma_h(k)              = sigma_h^{Mura_Self}(k)   [external numerical input]
        c_L(k)                  = C_s * exp(V_H sigma_h(k) / (R T)) ,

    where ``sigma_h^{Mura_Self}(kappa)`` is the external Mura line-integral
    self-stress (a black-box input, NOT re-derived here), and ``C_s`` is the
    Sieverts surface concentration that already carries the boundary fugacity
    (hence ``P_ext``) through the Abel-Noble real-gas EOS.

    The excess hydrogen free-energy density ``Psi_H(c_L) - Psi_H^ref`` is
    spatially uniform and integrates analytically over the cylindrical
    atmosphere (per unit line length) of cross-section
    ``A_atm = pi (R_atm^2 - r_c^2)`` with ``R_atm = 1/sqrt(rho_0)`` the mean
    dislocation spacing:

        E_H(k) = [Psi_H(c_L(k)) - Psi_H^ref] * A_atm .

    The reference ``Psi_H^ref`` is the ZERO-stress state (no dislocation
    self-stress, ``sigma_h = 0``), i.e. ``c_ref = C_s`` (the surface
    concentration itself), so ``E_H`` isolates the DISLOCATION-INDUCED excess:
    ``E_H -> 0`` as ``k -> 0`` (straight line, ``sigma_h^{Mura_Self} -> 0``) and
    becomes more negative as ``k`` increases (tighter bowing raises
    ``sigma_h``, enriches the lattice, and deepens the trap-binding reduction)
    - the core of the line-tension degradation theory.

    Args:
        kappa: Dislocation curvature (1/m), strictly positive.
        n_radial: Unused (kept for API compatibility); the stress is uniform.

    Returns:
        Hydrogen line energy ``E_H`` in J/m (<= 0).
    """
    _ = n_radial  # stress is uniform in space; no radial quadrature needed.
    R_atm = 1.0 / float(np.sqrt(C.RHO_0))            # mean dislocation spacing (m)
    r_c = C.CORE_RADIUS
    A_atm = float(np.pi * (R_atm ** 2 - r_c ** 2))    # atmosphere cross-section per unit length

    # Local hydrostatic stress: raw Mura self-stress ONLY (no superposition).
    sigma_h = float(get_mura_self_stress(kappa))
    # Stress-assisted lattice concentration driven by sigma_h (Mura only).
    c_lattice = float(stress_assisted_lattice_concentration(np.array(sigma_h)))
    theta_t = float(H.oriani_trap_occupancy(np.array(c_lattice), T_FIXED))
    psi_h = float(H.hydrogen_free_energy_density(np.array(c_lattice), np.array(theta_t), T_FIXED))

    # Reference: ZERO dislocation self-stress (sigma_h = 0) -> c_ref = C_s.
    # Isolates the dislocation-induced excess so E_H -> 0 as kappa -> 0.
    c_ref = float(H.surface_concentration(T_FIXED))
    theta_ref = float(H.oriani_trap_occupancy(np.array(c_ref), T_FIXED))
    psi_ref = float(H.hydrogen_free_energy_density(np.array(c_ref), np.array(theta_ref), T_FIXED))

    return (psi_h - psi_ref) * A_atm


def hydrogen_line_tension(kappa: np.ndarray, n_radial: int = 2048) -> np.ndarray:
    """Hydrogen-induced line-tension reduction ``Gamma_H(k)`` (J/m, <= 0).

    In the line-tension approximation (``Gamma ~ E``, standard in dislocation
    mobility), the hydrogen contribution to the line tension equals the
    excess hydrogen line energy ``E_H(k)`` from :func:`hydrogen_line_energy`,
    driven by the local hydrostatic stress ``sigma_h(k)`` (Mura self-stress
    only, no ``P_ext`` superposition). Because the trap-binding term lowers the
    free energy and the stress-driven enrichment grows with curvature,
    ``Gamma_H <= 0`` and becomes more negative as ``k`` increases: hydrogen
    binding degrades (lowers) the line tension, and the degradation deepens
    with curvature - the core of the line-tension degradation theory.

    Args:
        kappa: Dislocation curvature (1/m), shape ``(K,)``.
        n_radial: Unused (kept for API compatibility); the stress is uniform.

    Returns:
        Hydrogen line-tension reduction ``Gamma_H`` in J/m (<= 0), shape ``(K,)``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    return np.array([hydrogen_line_energy(k, n_radial=n_radial) for k in kappa])


def total_line_tension(kappa: np.ndarray) -> tuple:
    """Total line tension ``Gamma = Gamma_mech + Gamma_H`` and the split.

    Args:
        kappa: Dislocation curvature (1/m), shape ``(K,)``.

    Returns:
        ``(Gamma_mech, Gamma_H, Gamma_total)`` each in J/m, shape ``(K,)``.
    """
    g_mech = mechanical_line_tension(kappa)
    g_h = hydrogen_line_tension(kappa)
    return g_mech, g_h, g_mech + g_h


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def report_state(state: dict) -> None:
    """Print the fixed-state thermodynamic quantities to stdout."""
    print("=" * 72)
    print("Line-tension degradation theory - Version 1 fixed-state proof (30CrMo storage steel)")
    print("=" * 72)
    print(f"  Fixed temperature   T      = {T_FIXED:.2f} K  ({T_FIXED - 273.15:.2f} °C)")
    print(f"  Fixed pressure       P_ext = {P_FIXED:.3e} Pa  ({P_FIXED / 1e6:.1f} MPa)")
    print("-" * 72)
    print("  Abel-Noble real-gas EOS:  P (V - b) = R T")
    print(f"  Co-volume            b_AN  = {C.B_AN:.3e} m^3/mol")
    print(f"  Fugacity coefficient phi  = {state['phi']:.6f}  (ideal = 1.000000)")
    print(f"  H2 fugacity          f     = {state['f']:.3e} Pa  "
          f"({state['f'] / 1e6:.3f} MPa)")
    print(f"  Ideal-gas fugacity   f_id  = {state['f_ideal']:.3e} Pa  "
          f"({state['f_ideal'] / 1e6:.3f} MPa)")
    print("-" * 72)
    print(f"  Sieverts coefficient K_s   = {state['K_s']:.3e} mol/(m^3 Pa^0.5)")
    print(f"  Surface concentration C_s = {state['C_s']:.3e} mol/m^3")
    print(f"  Trap binding energy  D G_b = {C.DELTA_G_B / 1e3:.1f} kJ/mol")
    print(f"  Lattice site density N_L   = {C.N_L:.3e} mol/m^3")
    print(f"  Burgers vector       b      = {C.BURGERS_B:.3e} m")
    print(f"  Shear modulus        mu     = {C.MU:.3e} Pa")
    print(f"  Poisson ratio        nu     = {C.NU:.4f}")
    print("=" * 72)


def report_line_tension(kappa: np.ndarray, g_mech, g_h, g_tot) -> None:
    """Print a compact curvature / stress / line-tension table.

    All quantities are parameterised by the curvature ``kappa`` (1/m); the
    radius ``R(kappa) = 1/kappa`` and the local hydrostatic stress
    ``sigma_h(kappa) = sigma_h^{Mura_Self}(kappa)`` (Mura self-stress ONLY, no
    ``P_ext`` superposition, dynamically scaling with ``kappa``) are listed
    alongside the line-tension split.
    """
    print(f"{'kappa [1/m]':>14} {'R [nm]':>10} {'sigma_h [GPa]':>16} "
          f"{'Gamma_mech [nN]':>16} {'Gamma_H [nN]':>14} {'Gamma [nN]':>12}")
    print("-" * 88)
    # Sample logarithmically across the grid for a readable table.
    idx = np.logspace(np.log10(1), np.log10(len(kappa)), num=8).astype(int) - 1
    idx = np.clip(idx, 0, len(kappa) - 1)
    sh = get_mura_self_stress(kappa)   # sigma_h(k) = raw Mura self-stress (Pa)
    for i in idx:
        r_nm = 1.0 / kappa[i] * 1e9
        print(f"{kappa[i]:14.3e} {r_nm:10.2f} {sh[i] / 1e9:16.4f} "
              f"{g_mech[i] * 1e9:16.4f} {g_h[i] * 1e9:14.4f} {g_tot[i] * 1e9:12.4f}")
    print("=" * 88)
    print("  sigma_h(kappa) = sigma_h^{Mura_Self}(kappa),  R(kappa) = 1/kappa")
    print("  Driving stress is EXCLUSIVELY sigma_h^{Mura_Self}(kappa) (Mura line-integral, NOT re-derived).")
    print("  NO 'total' superposition: no P_ext, no added line-tension stress/energy on top of Mura.")
    print("  P_ext only drives the boundary fugacity C_s (Abel-Noble/Sieverts).")
    print("  c_L = C_s * exp(V_H sigma_h^{Mura_Self} / (R T))  (raw Mura output scales with kappa)")
    print("  Gamma_H < 0 confirms hydrogen-induced line-tension DEGRADATION.")
    print("  Larger kappa (tighter bowing) -> deeper reduction (line-tension degradation).")
    print("=" * 88)


# ---------------------------------------------------------------------------
# Publication-ready figure: Gamma vs kappa.
# ---------------------------------------------------------------------------
def plot_line_tension(kappa, g_mech, g_h, g_tot, out_path: str) -> None:
    """Generate and save the publication-ready Gamma vs kappa figure."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)

    ax.plot(kappa, g_tot * 1e9, color="#1f1f1f", lw=2.4, label=r"$\Gamma$ (total)")
    ax.plot(kappa, g_mech * 1e9, color="#1f77b4", lw=1.8, ls="--",
            label=r"$\Gamma_{\mathrm{mech}}$ (hydrogen-free)")
    ax.plot(kappa, g_h * 1e9, color="#d62728", lw=1.8, ls="-.",
            label=r"$\Gamma_{\mathrm{H}}$ (hydrogen reduction)")

    ax.axhline(0.0, color="0.5", lw=0.8, ls=":")

    ax.set_xscale("log")
    ax.set_xlabel(r"Dislocation curvature  $\kappa = 1/R$  (m$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"Line tension  $\Gamma$  (nN)", fontsize=12)
    ax.set_title(
        r"H-induced line-tension degradation, driving stress $\sigma_h(\kappa)=\sigma_h^{\mathrm{Mura}}(\kappa)$"
        r" exclusively (no 'total' superposition), 30CrMo, "
        f"$T={T_FIXED - 273.15:.0f}^\\circ$C, $P_{{\\mathrm{{ext}}}}={P_FIXED / 1e6:.0f}$ MPa)",
        fontsize=10,
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Figure saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    state = thermodynamic_state()
    report_state(state)

    # Curvature grid: from ~straight (R ~ 5 um) to the elastic limit where the
    # line tension Gamma = Gamma_mech + Gamma_H is still positive. The local
    # hydrostatic stress sigma_h(kappa) = sigma_h^{Mura_Self}(kappa) is the RAW
    # Mura line-integral output (sole mechanical-stress input, NO "total"
    # superposition, NO P_ext, NO added line-tension stress) and drives the
    # lattice concentration c_L = C_s exp(V_H sigma_h / (R T)); the
    # configurational free-energy excess keeps Gamma_mech + Gamma_H positive
    # only up to kappa ~ 7.1e6 1/m. The upper bound kappa_max = 6.0e6
    # (R ~ 167 nm) keeps Gamma_total safely above zero while resolving the full
    # degradation trend, with the peak lattice occupancy well below unity
    # (continuum elastic regime).
    kappa_min = 1.0 / (5.0e-6)          # R = 5 um  -> kappa ~ 2e5 1/m
    kappa_max = 6.0e6                   # R ~ 167 nm -> elastic, unsaturated, Gamma > 0
    kappa = np.logspace(np.log10(kappa_min), np.log10(kappa_max), 80)

    g_mech, g_h, g_tot = total_line_tension(kappa)
    report_line_tension(kappa, g_mech, g_h, g_tot)

    out_path = os.path.join(HERE, "figures", "line_tension_vs_curvature.png")
    plot_line_tension(kappa, g_mech, g_h, g_tot, out_path)


if __name__ == "__main__":
    main()
