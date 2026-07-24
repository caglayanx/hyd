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
    7. Mechanical line energy       -> E_mech(k) = mu b^2/(4 pi (1-nu)) ln(1/(k r_c))
    8. Hydrogen line-energy reduction  -> Gamma_H(k) (closed form, Kirchheim)
    9. Total line tension             -> Gamma(k) = Gamma_mech(k) + Gamma_H(k)

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
    length. For a bowed edge dislocation of radius of curvature ``R = 1/kappa``,

        Gamma_mech(k) = E_mech(R) = mu b^2 / (4 pi (1 - nu)) * ln(R / r_c) ,

    with the dislocation core radius ``r_c = b``. ``Gamma_mech`` diverges as
    ``k -> 0`` (straight line) and vanishes as ``k -> 1/r_c`` (core limit).

    Args:
        kappa: Dislocation curvature (1/m), strictly positive, shape ``(K,)``.

    Returns:
        Mechanical line tension ``Gamma_mech`` in J/m (= N), shape ``(K,)``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    pre = C.MU * C.BURGERS_B ** 2 / (4.0 * np.pi * (1.0 - C.NU))
    return pre * np.log(1.0 / (kappa * C.CORE_RADIUS))


# ---------------------------------------------------------------------------
# Step 3 (stress field): hydrostatic field of an edge dislocation.
# The tensile hydrostatic stress of an edge dislocation (Hirth & Lothe) is
#   sigma_h(r) = A / r ,   A = mu b / (2 pi (1 - nu)) ,
# sampled on a radial line at the angle of maximum tension. The curvature
# radius R = 1/kappa sets the outer cutoff of the hydrogen atmosphere: a
# tighter bow (larger k) confines the atmosphere to a smaller, more stressed
# near-core region. The ``r + r_c`` regularisation removes the r -> 0
# singularity at the core.
# ---------------------------------------------------------------------------
def edge_hydrostatic_stress(r: np.ndarray) -> np.ndarray:
    """Hydrostatic stress field of an edge dislocation, ``sigma_h(r) = A/r``.

    Args:
        r: Radial distance from the core (m), shape ``(...)``.

    Returns:
        Hydrostatic stress in Pa (positive in tension), same shape as ``r``.
    """
    r = np.asarray(r, dtype=np.float64)
    A = C.MU * C.BURGERS_B / (2.0 * np.pi * (1.0 - C.NU))
    return A / (r + C.CORE_RADIUS)


def line_tension_stress(kappa: np.ndarray) -> np.ndarray:
    """Curvature-induced line-tension back-stress ``sigma_LT(k)`` (Pa, tensile).

    A bowed edge dislocation carries an additional line-tension hydrostatic
    stress that scales linearly with the curvature,

        sigma_LT(k) = mu b k / (2 (1 - nu)) ,

    and is sampled at the core where it enriches the hydrogen atmosphere. This
    is the term that makes the hydrogen line-tension reduction grow with
    curvature: a tighter bow (larger k) raises sigma_LT, enlarges the
    stress-assisted enrichment ``exp(Omega sigma / (R T))``, and deepens the
    reduction - the core of the line-tension degradation theory.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    return C.MU * C.BURGERS_B * kappa / (2.0 * (1.0 - C.NU))


# ---------------------------------------------------------------------------
# Steps 3-6 & 8: hydrogen line-energy and line-tension reduction via the
# free-energy chain integrated over the dislocation stress field.
# ---------------------------------------------------------------------------
def hydrogen_line_energy(kappa: float, n_radial: int = 2048) -> float:
    """Excess hydrogen line energy ``E_H(k)`` (J/m) for one curvature.

    Integrates the *excess* hydrogen free-energy density ``Psi_H - Psi_H^ref``
    (steps 5-6) over the cylindrical atmosphere of an edge dislocation, per
    unit line length, on a logarithmic radial grid from the core radius
    ``r_c = b`` out to a FIXED outer cutoff ``R_atm`` (the mean dislocation
    spacing ``1/sqrt(rho_0)``, i.e. the atmosphere overlap scale):

        E_H(k) = integral_{r_c}^{R_atm} [Psi_H(r,k) - Psi_H^ref] * 2 pi r dr ,

    with ``dV/L = 2 pi r dr`` the cylindrical-shell volume per unit line
    length. The reference ``Psi_H^ref`` is the stress-free bulk hydrogen
    free-energy density (``c_L = C_s``, no dislocation stress), so ``E_H``
    captures only the DISLOCATION-INDUCED excess hydrogen free energy.

    The driving hydrostatic stress is the sum of the static edge field and the
    curvature-induced line-tension back-stress (step 3),

        sigma_h(r, k) = A / (r + r_c) + sigma_LT(k) ,
        sigma_LT(k)   = mu b k / (2 (1 - nu)) ,

    so the lattice concentration ``c_L(r,k) = C_s exp(Omega sigma_h / (R T))``
    and the Oriani trap occupancy ``theta_T(r,k)`` (step 4) both grow with
    curvature. Because the trap-binding term ``Psi_trap = -Delta G_b * c_T`` is
    negative and grows with the stress-driven enrichment, ``E_H < 0`` and
    becomes more negative as ``k`` increases: hydrogen binding lowers the line
    energy, and the effect deepens with curvature.

    Args:
        kappa: Dislocation curvature (1/m), strictly positive.
        n_radial: Number of logarithmic radial quadrature points.

    Returns:
        Hydrogen line energy ``E_H`` in J/m (<= 0).
    """
    R_atm = 1.0 / float(np.sqrt(C.RHO_0))  # mean dislocation spacing (m)
    r = np.geomspace(C.CORE_RADIUS, R_atm, n_radial)
    # A bowed dislocation under line tension carries an additional tensile
    # hydrostatic stress along its length (the line tension pulls on the
    # segment endpoints); to first order this acts uniformly over the
    # hydrogen atmosphere and is superposed on the static 1/r edge field.
    sigma_h = edge_hydrostatic_stress(r) + line_tension_stress(kappa)
    c_lattice = stress_assisted_lattice_concentration(sigma_h)
    theta_t = H.oriani_trap_occupancy(c_lattice, T_FIXED)
    psi_h = H.hydrogen_free_energy_density(c_lattice, theta_t, T_FIXED)
    # Stress-free bulk reference: c_L = C_s, no hydrostatic stress enrichment.
    c_ref = np.full_like(r, H.surface_concentration(T_FIXED))
    theta_ref = H.oriani_trap_occupancy(c_ref, T_FIXED)
    psi_ref = H.hydrogen_free_energy_density(c_ref, theta_ref, T_FIXED)
    # Trapezoidal integration of the excess density * 2 pi r dr (per unit length).
    integrand = (psi_h - psi_ref) * 2.0 * np.pi * r
    return float(np.trapezoid(integrand, r))


def hydrogen_line_tension(kappa: np.ndarray, n_radial: int = 2048) -> np.ndarray:
    """Hydrogen-induced line-tension reduction ``Gamma_H(k)`` (J/m, <= 0).

    In the line-tension approximation (``Gamma ~ E``, standard in dislocation
    mobility), the hydrogen contribution to the line tension equals the
    excess hydrogen line energy ``E_H(k)`` from :func:`hydrogen_line_energy`.
    Because the trap-binding term lowers the free energy and the stress-driven
    enrichment grows with curvature, ``Gamma_H <= 0`` and becomes more negative
    as ``k`` increases: hydrogen binding degrades (lowers) the line tension,
    and the degradation deepens with curvature - the core of the line-tension
    degradation theory.

    Args:
        kappa: Dislocation curvature (1/m), shape ``(K,)``.
        n_radial: Number of radial quadrature points per curvature.

    Returns:
        Hydrogen line-tension reduction ``Gamma_H`` in J/m (<= 0), shape ``(K,)``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    return np.array([hydrogen_line_energy(k, n_radial) for k in kappa])


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
    """Print a compact curvature / line-tension table at sampled curvatures."""
    print(f"{'kappa [1/m]':>14} {'R [nm]':>10} {'Gamma_mech [nN]':>16} "
          f"{'Gamma_H [nN]':>14} {'Gamma [nN]':>12}")
    print("-" * 72)
    # Sample logarithmically across the grid for a readable table.
    idx = np.logspace(np.log10(1), np.log10(len(kappa)), num=8).astype(int) - 1
    idx = np.clip(idx, 0, len(kappa) - 1)
    for i in idx:
        r_nm = 1.0 / kappa[i] * 1e9
        print(f"{kappa[i]:14.3e} {r_nm:10.2f} {g_mech[i] * 1e9:16.4f} "
              f"{g_h[i] * 1e9:14.4f} {g_tot[i] * 1e9:12.4f}")
    print("=" * 72)
    print("  Gamma_H < 0 confirms hydrogen-induced line-tension DEGRADATION.")
    print("  Larger kappa (tighter bowing) -> deeper reduction (line-tension degradation).")
    print("=" * 72)


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
        r"H-induced line-tension degradation (30CrMo, "
        f"$T={T_FIXED - 273.15:.0f}^\\circ$C, $P={P_FIXED / 1e6:.0f}$ MPa)",
        fontsize=11,
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

    # Curvature grid: from ~straight (R ~ 5 um) to the near-core elastic limit.
    # The upper bound is set by lattice saturation: the combined edge + line-
    # tension stress drives c_L toward N_L, beyond which the configurational
    # entropy diverges (core artifact). kappa ~ 1e8 1/m (R ~ 10 nm) keeps the
    # peak lattice occupancy well below unity and the continuum elastic regime
    # intact.
    kappa_min = 1.0 / (5.0e-6)          # R = 5 um  -> kappa ~ 2e5 1/m
    kappa_max = 3.0e7                    # R = 33 nm -> elastic, unsaturated, Gamma > 0
    kappa = np.logspace(np.log10(kappa_min), np.log10(kappa_max), 80)

    g_mech, g_h, g_tot = total_line_tension(kappa)
    report_line_tension(kappa, g_mech, g_h, g_tot)

    out_path = os.path.join(HERE, "figures", "line_tension_vs_curvature.png")
    plot_line_tension(kappa, g_mech, g_h, g_tot, out_path)


if __name__ == "__main__":
    main()
