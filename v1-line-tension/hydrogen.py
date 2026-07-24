from __future__ import annotations

"""Chemical potential, hydrostatic stress, and trapping kinetics.

Couples the mechanical state (hydrostatic stress ``sigma_h``) to the hydrogen
transport problem through the stress-assisted chemical potential

    μ_H = μ_H^0 + R·T·ln(C_L / N_L) − V̄_H · σ_h

and the Oriani / Krom equilibrium between lattice and trapped populations.

All kinetics follow the Sofronis-McMeeking (1989) and Krom et al. (1999)
formulations with the 30CrMo storage-steel parameters in
:mod:`constants` (High-Pressure Hydrogen Storage Vessel context).
"""
from typing import Tuple

import numpy as np

import constants as C


def hydrostatic_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """Return the hydrostatic (trace) component of a stress tensor.

    Args:
        stress_tensor: Cauchy stress of shape ``(..., 3, 3)`` in Pascals.

    Returns:
        Hydrostatic stress ``σ_h = tr(σ)/3`` of shape ``(...,)`` in Pascals.
    """
    stress_tensor = np.asarray(stress_tensor, dtype=np.float64)
    return np.trace(stress_tensor, axis1=-2, axis2=-1) / 3.0


def chemical_potential(
    c_lattice: np.ndarray,
    sigma_h: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Stress-assisted lattice chemical potential of hydrogen.

    Args:
        c_lattice: Lattice hydrogen concentration in mol/m^3.
        sigma_h: Hydrostatic stress in Pa (positive in tension).
        temperature: Absolute temperature in Kelvin.

    Returns:
        Chemical potential ``μ_H`` in J/mol.
    """
    c_lattice = np.asarray(c_lattice, dtype=np.float64)
    sigma_h = np.asarray(sigma_h, dtype=np.float64)
    safe_c = np.maximum(c_lattice, C.EPS)
    return C.R * temperature * np.log(safe_c / C.N_L) - C.V_BAR_H * sigma_h


def oriani_equilibrium(
    c_lattice: np.ndarray,
    trap_density: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Oriani equilibrium trapped concentration for dislocation traps.

    Args:
        c_lattice: Lattice hydrogen concentration in mol/m^3.
        trap_density: Dislocation trap density in sites/m^3.
        temperature: Absolute temperature in Kelvin.

    Returns:
        Trapped hydrogen concentration ``C_T`` in mol/m^3.

    Raises:
        NotImplementedError: Trapping kinetics are delivered in a later task.
    """
    _ = (c_lattice, trap_density, temperature, C.E_B_DISLOC, C.ETA_D)
    raise NotImplementedError("Oriani trapping kinetics not implemented yet.")


# ---------------------------------------------------------------------------
# 9-step free-energy chain (Sieverts -> stress-assisted c_L -> Oriani trap ->
# Psi_H -> W_total -> E(kappa) -> line tension Gamma).
# ---------------------------------------------------------------------------
def sieverts_coefficient(temperature: float = C.T_REF) -> float:
    """Temperature-dependent Sieverts solubility coefficient ``K_s(T)``.

    ``K_s(T) = K_S_0 * exp(-H_S / (R*T))`` in ``mol / (m^3 * Pa^0.5)``.

    Args:
        temperature: Absolute temperature in Kelvin.

    Returns:
        Sieverts coefficient ``K_s(T)``.
    """
    return C.K_S_0 * float(np.exp(-C.H_S / (C.R * float(temperature))))


# ---------------------------------------------------------------------------
# Abel-Noble real-gas fugacity (fixed V1 state).
# ---------------------------------------------------------------------------
# The Abel-Noble equation of state for a dense gas with molar co-volume b is
#
#     P (V - b) = R T   ->   Z = P V / (R T) = 1 + b P / (R T).
#
# The fugacity coefficient follows from the standard relation
# ln(phi) = integral_0^P (Z - 1)/P' dP', which integrates analytically to
#
#     phi(P, T) = exp(b P / (R T)) ,   f(P, T) = phi * P.
#
# Because b > 0, phi > 1: the real-gas fugacity of high-pressure H2 exceeds the
# pressure, raising the Sieverts surface concentration C_s = K_s(T) sqrt(f)
# relative to the ideal-gas estimate C_s = K_s(T) sqrt(P).
# ---------------------------------------------------------------------------
def abel_noble_fugacity_coefficient(
    pressure: float = C.P_OP,
    temperature: float = C.T_OP,
) -> float:
    """Abel-Noble fugacity coefficient ``phi(P,T) = exp(b*P/(R*T))``.

    Args:
        pressure: H2 gas pressure in Pa (default: fixed V1 operating pressure).
        temperature: Absolute temperature in K (default: fixed V1 temperature).

    Returns:
        Dimensionless fugacity coefficient ``phi`` (>= 1 for b > 0).
    """
    rt = C.R * float(temperature)
    return float(np.exp(C.B_AN * float(pressure) / rt))


def abel_noble_fugacity(
    pressure: float = C.P_OP,
    temperature: float = C.T_OP,
) -> float:
    """Abel-Noble H2 fugacity ``f(P,T) = phi(P,T) * P`` in Pa.

    Args:
        pressure: H2 gas pressure in Pa (default: fixed V1 operating pressure).
        temperature: Absolute temperature in K (default: fixed V1 temperature).

    Returns:
        H2 fugacity ``f`` in Pa.
    """
    return abel_noble_fugacity_coefficient(pressure, temperature) * float(pressure)


def surface_concentration(temperature: float = C.T_REF) -> float:
    """Sieverts surface (lattice-interface) hydrogen concentration ``C_s``.

    ``C_s = K_s(T) * sqrt(f)`` in ``mol / m^3``, with the H2 fugacity ``f`` taken
    from the Abel-Noble real-gas equation evaluated at the FIXED V1 operating
    state (P = 35 MPa, T = 25 °C). This couples the real-gas thermodynamics of
    the storage vessel to the lattice solubility through Sieverts' law.

    Args:
        temperature: Absolute temperature in Kelvin (kept for API symmetry; the
            fugacity is evaluated at the fixed V1 pressure/temperature).

    Returns:
        Surface lattice hydrogen concentration ``C_s`` in mol/m^3.
    """
    f = abel_noble_fugacity(C.P_OP, C.T_OP)
    return sieverts_coefficient(temperature) * float(np.sqrt(f))


def stress_assisted_lattice_concentration(
    sigma_h: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Stress-assisted lattice hydrogen concentration ``c_L(kappa)``.

    ``c_L(kappa) = C_s * exp(Omega * sigma_h(kappa) / (R*T))`` in mol/m^3,
    where ``C_s`` is the Sieverts surface concentration and ``Omega`` is the
    hydrogen partial molar volume (positive in tension -> hydrostatic tension
    enriches the lattice).

    Args:
        sigma_h: Hydrostatic stress field ``sigma_h = tr(sigma)/3`` in Pa
            (positive in tension), array of arbitrary shape. This may be the
            output of :func:`radial_hydrostatic_stress` so that ``c_L`` is
            resolved as a spatial function ``c_L(R)``.
        temperature: Absolute temperature in Kelvin.

    Returns:
        Lattice hydrogen concentration ``c_L`` in mol/m^3, same shape as
        ``sigma_h``.
    """
    sigma_h = np.asarray(sigma_h, dtype=np.float64)
    c_s = surface_concentration(temperature)
    return c_s * np.exp(C.OMEGA * sigma_h / (C.R * float(temperature)))


def radial_hydrostatic_stress(
    R: np.ndarray,
    form: str = "kfield",
    amplitude: float = 1.0e6,
    eps: float = C.SIGMA_H_EPS,
) -> np.ndarray:
    """Spatial hydrostatic stress field ``sigma_h(R)`` (NumPy counterpart).

    Mirrors the symbolic UFL ``sigma_h(R)`` built in
    :class:`HydrogenDiffusionSolver` so that the free-energy chain can consume
    a spatially varying stress field rather than a single scalar. The
    ``R + eps`` regularisation removes the ``R -> 0`` singularity at the
    notch/dislocation centre.

    Args:
        R: Radial distance from the notch/dislocation centre (m), array of
            arbitrary shape.
        form: ``"zero"`` (pure Fickian), ``"kfield"`` (regularised near-tip
            ``A / sqrt(R + eps)``, amplitude in Pa*sqrt(m)), or
            ``"dislocation"`` (regularised ``A / (R + eps)``, amplitude in
            Pa*m).
        amplitude: Amplitude ``A`` of the chosen form.
        eps: ``R + eps`` regularisation length (m).

    Returns:
        Hydrostatic stress ``sigma_h(R)`` in Pa, same shape as ``R``.
    """
    R = np.asarray(R, dtype=np.float64)
    safe_r = R + float(eps)
    if form == "zero":
        return np.zeros_like(R)
    if form == "kfield":
        return float(amplitude) / np.sqrt(safe_r)
    if form == "dislocation":
        return float(amplitude) / safe_r
    raise ValueError(
        f"form must be 'zero', 'kfield', or 'dislocation', got {form!r}."
    )


def trap_equilibrium_constant(temperature: float = C.T_REF) -> float:
    """Oriani trap equilibrium constant ``K_T = exp(-Delta G_b / (R*T))``."""
    return float(np.exp(-C.DELTA_G_B / (C.R * float(temperature))))


def oriani_trap_occupancy(
    c_lattice: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Single-dislocation Oriani trap occupancy ``theta_T``.

    Solves the Oriani equilibrium

        ``theta_T / (1 - theta_T) = K_T * theta_L / (1 - theta_L)``

    with ``theta_L = c_L / N_L`` and ``K_T = exp(-Delta G_b / (R*T))``. The
    closed-form solution is ``theta_T = rhs / (1 + rhs)`` with
    ``rhs = K_T * theta_L / (1 - theta_L)``.

    Args:
        c_lattice: Lattice hydrogen concentration in mol/m^3 (same units as
            ``N_L``), array of arbitrary shape.
        temperature: Absolute temperature in Kelvin.

    Returns:
        Trap occupancy ``theta_T`` in ``[0, 1)``, same shape as ``c_lattice``.
    """
    c_lattice = np.asarray(c_lattice, dtype=np.float64)
    theta_l = np.clip(c_lattice / C.N_L_MOLAR, 0.0, 1.0 - 1.0e-12)
    k_t = trap_equilibrium_constant(temperature)
    rhs = k_t * theta_l / (1.0 - theta_l)
    return rhs / (1.0 + rhs)


def trap_line_concentration(theta_t: np.ndarray) -> np.ndarray:
    """Line-trapped hydrogen concentration ``C_T^line = theta_T * n_T``.

    ``n_T = 1 / b`` is the number of trap sites per unit line length (sites/m),
    so ``C_T^line`` is in sites/m.

    Args:
        theta_t: Trap occupancy ``theta_T`` (dimensionless).

    Returns:
        Line-trapped concentration ``C_T^line`` in sites/m.
    """
    theta_t = np.asarray(theta_t, dtype=np.float64)
    return theta_t * C.ETA_D


def trap_volume_concentration(theta_t: np.ndarray) -> np.ndarray:
    """Volume-trapped hydrogen concentration ``c_T^vol = C_T^line / A_eff``.

    Args:
        theta_t: Trap occupancy ``theta_T`` (dimensionless).

    Returns:
        Volume-trapped concentration ``c_T^vol`` in sites/m^3.
    """
    return trap_line_concentration(theta_t) / C.A_EFF


def configurational_free_energy_density(
    c_lattice: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Configurational (mixing-entropy) free-energy density ``Psi_conf(c_L)``.

    ``Psi_conf = R*T * [ c_L*ln(c_L/N_L) + (N_L - c_L)*ln(1 - c_L/N_L) ]`` in
    J/m^3, with the standard lattice-gas mixing entropy over ``N_L`` sites.

    Args:
        c_lattice: Lattice hydrogen concentration in mol/m^3 (same units as
            ``N_L``), array of arbitrary shape.
        temperature: Absolute temperature in Kelvin.

    Returns:
        Configurational free-energy density ``Psi_conf`` in J/m^3.
    """
    c_lattice = np.asarray(c_lattice, dtype=np.float64)
    rt = C.R * float(temperature)
    n_l = C.N_L_MOLAR
    theta = c_lattice / n_l
    # Guard the 0 * log(0) endpoints (c_L = 0 and c_L = N_L) against NaN; the
    # errstate silences the log(0) that numpy still evaluates inside where.
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(c_lattice > 0.0, c_lattice * np.log(theta), 0.0)
        term2 = np.where(c_lattice < n_l, (n_l - c_lattice) * np.log(1.0 - theta), 0.0)
    return rt * (term1 + term2)


def trap_free_energy_density(
    theta_t: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Trap-binding free-energy density ``Psi_trap = -Delta G_b * c_T^vol``.

    Args:
        theta_t: Trap occupancy ``theta_T`` (dimensionless).
        temperature: Absolute temperature in Kelvin (unused; binding energy is
            isothermal here, kept for API symmetry).

    Returns:
        Trap free-energy density ``Psi_trap`` in J/m^3 (negative: binding
        lowers the free energy).
    """
    _ = temperature
    # c_T^vol is in sites/m^3 (step 3); convert to mol/m^3 for a J/m^3 density
    # when combined with the molar binding free energy Delta G_b (J/mol).
    c_t_vol_molar = trap_volume_concentration(theta_t) / C.N_AVOG
    return -C.DELTA_G_B * c_t_vol_molar


def hydrogen_free_energy_density(
    c_lattice: np.ndarray,
    theta_t: np.ndarray,
    temperature: float = C.T_REF,
) -> np.ndarray:
    """Total hydrogen free-energy density ``Psi_H = Psi_conf + Psi_trap``.

    Args:
        c_lattice: Lattice hydrogen concentration in mol/m^3.
        theta_t: Trap occupancy ``theta_T`` (dimensionless).
        temperature: Absolute temperature in Kelvin.

    Returns:
        Hydrogen free-energy density ``Psi_H`` in J/m^3.
    """
    return (
        configurational_free_energy_density(c_lattice, temperature)
        + trap_free_energy_density(theta_t, temperature)
    )


__all__: Tuple[str, ...] = (
    "hydrostatic_stress",
    "chemical_potential",
    "oriani_equilibrium",
    "sieverts_coefficient",
    "abel_noble_fugacity_coefficient",
    "abel_noble_fugacity",
    "surface_concentration",
    "stress_assisted_lattice_concentration",
    "radial_hydrostatic_stress",
    "trap_equilibrium_constant",
    "oriani_trap_occupancy",
    "trap_line_concentration",
    "trap_volume_concentration",
    "configurational_free_energy_density",
    "trap_free_energy_density",
    "hydrogen_free_energy_density",
)
