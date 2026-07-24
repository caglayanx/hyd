"""Hydrogen thermodynamics for the V2 coupled mech-hydrogen simulator.

Implements the hydrogen side of the combined mech-hydrogen line-tension model
for 30CrMo storage steel, strictly following the project physical-model
document:

    P, T -> PR-EOS -> Z -> phi -> f -> C_s -> C_L(P,T,sigma_h)
          -> theta_T -> C_T -> Psi_H

The module is intentionally solver-agnostic: it contains only the
thermodynamic building blocks (Peng-Robinson EOS, Sieverts solubility,
stress-assisted lattice concentration, Oriani trapping, and the hydrogen
free-energy density). It depends on numpy only; FEniCSx is NOT imported here.

Notation (symbol clashes are avoided by design):
    b          : Burgers vector (mechanical side).
    b_PR       : Peng-Robinson co-volume (hydrogen side).
    kappa_PR   : Peng-Robinson correlation parameter (NOT dislocation curvature).
    V_H        : Partial molar volume of hydrogen.
    sigma_h    : Hydrostatic stress.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

import constants as C

# A scalar or a NumPy array of pressures / temperatures.
ArrayLike = Union[float, np.ndarray]

_SQRT2 = float(np.sqrt(2.0))


# ---------------------------------------------------------------------------
# 1. Peng-Robinson EOS parameters (a_PR(T), alpha(T), b_PR).
# ---------------------------------------------------------------------------
def kappa_pr(omega: float = C.OMEGA_H2) -> float:
    """PR acentric correlation parameter ``kappa_PR``.

    ``kappa_PR = 0.37464 + 1.54226*omega - 0.26992*omega^2``.
    """
    return 0.37464 + 1.54226 * float(omega) - 0.26992 * float(omega) ** 2


def alpha(T: ArrayLike, T_c: float = C.T_C_H2, omega: float = C.OMEGA_H2) -> ArrayLike:
    """PR temperature-dependent alpha factor.

    ``alpha(T) = [1 + kappa_PR (1 - sqrt(T/T_c))]^2``.
    """
    T = np.asarray(T, dtype=np.float64)
    k = kappa_pr(omega)
    return (1.0 + k * (1.0 - np.sqrt(T / float(T_c)))) ** 2


def a_c(T_c: float = C.T_C_H2, P_c: float = C.P_C_H2) -> float:
    """PR attractive constant ``a_c = 0.45724 R^2 T_c^2 / P_c`` (Pa m^6 / mol^2)."""
    return 0.45724 * C.R ** 2 * float(T_c) ** 2 / float(P_c)


def b_pr(T_c: float = C.T_C_H2, P_c: float = C.P_C_H2) -> float:
    """PR co-volume ``b_PR = 0.07780 R T_c / P_c`` (m^3 / mol).

    Named ``b_PR`` to avoid clashing with the Burgers vector ``b``.
    """
    return 0.07780 * C.R * float(T_c) / float(P_c)


def a_pr(T: ArrayLike, T_c: float = C.T_C_H2, P_c: float = C.P_C_H2,
         omega: float = C.OMEGA_H2) -> ArrayLike:
    """PR temperature-dependent attractive parameter ``a_PR(T) = a_c * alpha(T)``."""
    return a_c(T_c, P_c) * alpha(T, T_c, omega)


# ---------------------------------------------------------------------------
# 2. Dimensionless parameters and compressibility factor Z(T,P).
# ---------------------------------------------------------------------------
def A_dim(T: ArrayLike, P: ArrayLike, T_c: float = C.T_C_H2,
          P_c: float = C.P_C_H2, omega: float = C.OMEGA_H2) -> ArrayLike:
    """Dimensionless PR attraction parameter ``A = a_PR(T) P / (R^2 T^2)``."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    return a_pr(T, T_c, P_c, omega) * P / (C.R ** 2 * T ** 2)


def B_dim(T: ArrayLike, P: ArrayLike, T_c: float = C.T_C_H2,
          P_c: float = C.P_C_H2) -> ArrayLike:
    """Dimensionless PR co-volume parameter ``B = b_PR P / (R T)``."""
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    return b_pr(T_c, P_c) * P / (C.R * T)


def _solve_z_scalar(a: float, b: float) -> float:
    """Largest real root of the PR cubic (gas-phase compressibility).

    ``Z^3 - (1-B) Z^2 + (A - 3 B^2 - 2 B) Z - (A B - B^2 - B^3) = 0``.

    For supercritical H2 (T >> T_c) the cubic has a single real root; in the
    two-phase region the largest real root is taken as the gas-phase Z.
    """
    coeffs = [1.0, -(1.0 - b), (a - 3.0 * b ** 2 - 2.0 * b), -(a * b - b ** 2 - b ** 3)]
    roots = np.roots(coeffs)
    real = roots[np.isreal(roots)].real
    if real.size == 0:
        # Fall back to the root with the largest real part (numerical guard).
        return float(np.max(roots.real))
    return float(np.max(real))


def compressibility(T: ArrayLike, P: ArrayLike, T_c: float = C.T_C_H2,
                     P_c: float = C.P_C_H2, omega: float = C.OMEGA_H2) -> ArrayLike:
    """Compressibility factor ``Z(T,P)`` (dimensionless).

    Accepts scalar or array T, P (broadcast together). Returns an array of the
    same broadcasted shape holding the largest real root of the PR cubic.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    P_arr = np.asarray(P, dtype=np.float64)
    a_arr = np.broadcast_to(A_dim(T_arr, P_arr, T_c, P_c, omega), np.broadcast_shapes(T_arr.shape, P_arr.shape))
    b_arr = np.broadcast_to(B_dim(T_arr, P_arr, T_c, P_c), np.broadcast_shapes(T_arr.shape, P_arr.shape))
    out = np.array([_solve_z_scalar(float(a_), float(b_)) for a_, b_ in zip(a_arr.ravel(), b_arr.ravel())])
    return out.reshape(a_arr.shape)


def fugacity_coefficient(T: ArrayLike, P: ArrayLike, T_c: float = C.T_C_H2,
                         P_c: float = C.P_C_H2, omega: float = C.OMEGA_H2) -> ArrayLike:
    """PR fugacity coefficient ``phi(T,P)`` (dimensionless).

    ``ln(phi) = Z - 1 - ln(Z - B) - [A/(2 sqrt(2) B)] * ln( (Z + (1+sqrt2) B) / (Z + (1-sqrt2) B) )``.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    P_arr = np.asarray(P, dtype=np.float64)
    Z = np.asarray(compressibility(T_arr, P_arr, T_c, P_c, omega), dtype=np.float64)
    a = A_dim(T_arr, P_arr, T_c, P_c, omega)
    b = B_dim(T_arr, P_arr, T_c, P_c)
    # Guard B -> 0 (ideal-gas limit): phi -> 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(
            b > 0.0,
            (a / (2.0 * _SQRT2 * b))
            * np.log((Z + (1.0 + _SQRT2) * b) / (Z + (1.0 - _SQRT2) * b)),
            0.0,
        )
        ln_phi = (Z - 1.0) - np.log(Z - b) - term
    return np.exp(ln_phi)


def fugacity(T: ArrayLike, P: ArrayLike, T_c: float = C.T_C_H2,
             P_c: float = C.P_C_H2, omega: float = C.OMEGA_H2) -> ArrayLike:
    """H2 fugacity ``f(T,P) = phi(T,P) * P`` in Pa."""
    return fugacity_coefficient(T, P, T_c, P_c, omega) * np.asarray(P, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. Sieverts surface concentration C_s(P,T).
# ---------------------------------------------------------------------------
def sieverts_coefficient(T: ArrayLike, K_0: float = C.K_0,
                         H_s: float = C.H_S) -> ArrayLike:
    """Sieverts solubility coefficient ``K_s(T) = K_0 exp(-H_s/(R T))``.

    Units: ``mol / (m^3 Pa^0.5)``.
    """
    T = np.asarray(T, dtype=np.float64)
    return K_0 * np.exp(-H_s / (C.R * T))


def surface_concentration(T: ArrayLike, P: ArrayLike, T_c: float = C.T_C_H2,
                          P_c: float = C.P_C_H2, omega: float = C.OMEGA_H2,
                          K_0: float = C.K_0, H_s: float = C.H_S) -> ArrayLike:
    """Sieverts surface concentration ``C_s = K_s(T) sqrt(f)`` (mol/m^3).

    ``C_s = K_0 exp(-H_s/(R T)) sqrt(phi * P)``.
    """
    f = fugacity(T, P, T_c, P_c, omega)
    return sieverts_coefficient(T, K_0, H_s) * np.sqrt(f)


# ---------------------------------------------------------------------------
# 4. Stress-assisted lattice concentration C_L(P,T,sigma_h).
# ---------------------------------------------------------------------------
def lattice_concentration(T: ArrayLike, P: ArrayLike, sigma_h: ArrayLike,
                          T_c: float = C.T_C_H2, P_c: float = C.P_C_H2,
                          omega: float = C.OMEGA_H2, V_H: float = C.V_H) -> ArrayLike:
    """Stress-assisted lattice hydrogen concentration ``C_L`` (mol/m^3).

    ``C_L(P,T,sigma_h) = C_s(P,T) * exp(V_H sigma_h / (R T))``.

    Hydrostatic tension (``sigma_h > 0``) enriches the lattice; the surface
    concentration ``C_s`` is the stress-free Dirichlet boundary value applied
    on the notch surface by the V2 solver.
    """
    T = np.asarray(T, dtype=np.float64)
    sigma_h = np.asarray(sigma_h, dtype=np.float64)
    c_s = surface_concentration(T, P, T_c, P_c, omega)
    return c_s * np.exp(V_H * sigma_h / (C.R * T))


# ---------------------------------------------------------------------------
# 5. Oriani classical trap equilibrium theta_T(P,T,sigma_h), C_T.
# ---------------------------------------------------------------------------
def lattice_occupancy(c_lattice: ArrayLike, N_L: float = C.N_L) -> ArrayLike:
    """Lattice occupancy ``theta_L = C_L / N_L`` (dimensionless)."""
    return np.asarray(c_lattice, dtype=np.float64) / N_L


def trap_equilibrium_constant(T: ArrayLike, Delta_G_b: float = C.DELTA_G_B) -> ArrayLike:
    """Oriani trap equilibrium constant ``K_T = exp(-Delta_G_b / (R T))``."""
    T = np.asarray(T, dtype=np.float64)
    return np.exp(-Delta_G_b / (C.R * T))


def trap_occupancy(c_lattice: ArrayLike, T: ArrayLike, N_L: float = C.N_L,
                   Delta_G_b: float = C.DELTA_G_B) -> ArrayLike:
    """Oriani trap occupancy ``theta_T`` (dimensionless, in ``[0, 1)``).

    ``theta_T = K_T theta_L / [1 - theta_L + K_T theta_L]``.
    """
    theta_l = lattice_occupancy(c_lattice, N_L)
    k_t = trap_equilibrium_constant(T, Delta_G_b)
    return k_t * theta_l / (1.0 - theta_l + k_t * theta_l)


def trapped_concentration(c_lattice: ArrayLike, T: ArrayLike, N_L: float = C.N_L,
                          N_T: float = C.N_T, Delta_G_b: float = C.DELTA_G_B) -> ArrayLike:
    """Volume-trapped concentration ``C_T = N_T * theta_T`` (mol/m^3)."""
    return N_T * trap_occupancy(c_lattice, T, N_L, Delta_G_b)


# ---------------------------------------------------------------------------
# 6. Hydrogen free-energy density Psi_H(P,T,sigma_h).
# ---------------------------------------------------------------------------
def lattice_free_energy_density(c_lattice: ArrayLike, T: ArrayLike,
                                N_L: float = C.N_L) -> ArrayLike:
    """Lattice (configurational) free-energy density ``Psi_L`` (J/m^3).

    ``Psi_L = R T N_L [ theta_L ln(theta_L) + (1 - theta_L) ln(1 - theta_L) ]``.
    """
    T = np.asarray(T, dtype=np.float64)
    theta = np.asarray(c_lattice, dtype=np.float64) / N_L
    theta = np.clip(theta, 0.0, 1.0 - 1.0e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(theta > 0.0, theta * np.log(theta), 0.0) + \
               np.where(theta < 1.0, (1.0 - theta) * np.log(1.0 - theta), 0.0)
    return C.R * T * N_L * term


def trap_free_energy_density(c_lattice: ArrayLike, T: ArrayLike, N_L: float = C.N_L,
                             N_T: float = C.N_T, Delta_G_b: float = C.DELTA_G_B) -> ArrayLike:
    """Trap free-energy density ``Psi_T`` (J/m^3).

    ``Psi_T = R T N_T [ theta_T ln(theta_T) + (1 - theta_T) ln(1 - theta_T) ]
              - N_T theta_T Delta_G_b``.
    """
    T = np.asarray(T, dtype=np.float64)
    theta_t = trap_occupancy(c_lattice, T, N_L, Delta_G_b)
    theta_t = np.clip(theta_t, 0.0, 1.0 - 1.0e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(theta_t > 0.0, theta_t * np.log(theta_t), 0.0) + \
               np.where(theta_t < 1.0, (1.0 - theta_t) * np.log(1.0 - theta_t), 0.0)
    return C.R * T * N_T * term - N_T * theta_t * Delta_G_b


def hydrogen_free_energy_density(c_lattice: ArrayLike, T: ArrayLike, N_L: float = C.N_L,
                                 N_T: float = C.N_T,
                                 Delta_G_b: float = C.DELTA_G_B) -> ArrayLike:
    """Total hydrogen free-energy density ``Psi_H = Psi_L + Psi_T`` (J/m^3).

    ``Psi_H = Psi_H(P, T, sigma_h)`` once ``c_lattice = C_L(P, T, sigma_h)`` is
    supplied from :func:`lattice_concentration`.
    """
    return (lattice_free_energy_density(c_lattice, T, N_L)
            + trap_free_energy_density(c_lattice, T, N_L, N_T, Delta_G_b))


__all__: Tuple[str, ...] = (
    "kappa_pr", "alpha", "a_c", "b_pr", "a_pr",
    "A_dim", "B_dim", "compressibility",
    "fugacity_coefficient", "fugacity",
    "sieverts_coefficient", "surface_concentration",
    "lattice_concentration",
    "lattice_occupancy", "trap_equilibrium_constant",
    "trap_occupancy", "trapped_concentration",
    "lattice_free_energy_density", "trap_free_energy_density",
    "hydrogen_free_energy_density",
)
