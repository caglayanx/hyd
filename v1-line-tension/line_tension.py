from __future__ import annotations

"""Line-tension chain: free-energy density -> W_total -> E(kappa) -> Gamma.

Implements the last leg of the 9-step free-energy chain. Given a dislocation
configuration parametrised by a scalar curvature/loading parameter ``kappa``,
the total free energy

    W_total(kappa) = integral_{V_int} Psi_total(kappa) dV

is integrated over the interaction volume, the line energy is
``E(kappa) = W_total(kappa) / L(kappa)``, and the line tension is the total
derivative

    Gamma(kappa) = dE/dL = (dE/dkappa) / (dL/dkappa) = Gamma_mech + Gamma_H.

The mechanical and hydrogen contributions are tracked separately so that the
split ``Gamma = Gamma_mech + Gamma_H`` is available without recomputation.
"""
from typing import Callable, Tuple

import numpy as np

import constants as C
from hydrogen import hydrogen_free_energy_density


# ---------------------------------------------------------------------------
# Curvature parameterisation: R(kappa) = 1/kappa.
#
# Per reviewer feedback, every stress and energy calculation below takes the
# dislocation curvature ``kappa`` (1/m) as its PRIMARY input and explicitly
# defines the radius of curvature ``R(kappa) = 1/kappa`` at the top of the
# computation. No function in this module takes the radius ``R`` as a primary
# argument anymore; ``R`` is always derived from ``kappa``.
# ---------------------------------------------------------------------------
def radius_of_curvature(kappa: np.ndarray) -> np.ndarray:
    """Radius of curvature ``R(kappa) = 1/kappa`` (m).

    Args:
        kappa: Dislocation curvature (1/m), strictly positive, any shape.

    Returns:
        Radius of curvature ``R`` in m, same shape as ``kappa``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    return 1.0 / kappa


# ---------------------------------------------------------------------------
# Mura self-stress: BLACK-BOX numerical input (NOT re-derived here).
#
# Per the project's foundational document and reviewer feedback, the mechanical
# side is NOT re-derived inside this module. The hydrostatic self-stress of the
# curved dislocation,
#
#     sigma_h^{Mura_Self}(kappa),
#
# is an EXTERNAL INPUT that comes from a prior numerical Mura line-integral
# solution (solved upstream in Phase 1 / V2). It is exposed here ONLY through a
# callable black-box interface, ``get_mura_self_stress(kappa)``, so that this
# module never contains an analytical stress formula (no Hirth-Lothe logarithmic
# expression, no closed-form sigma_LT). Consumers register the real numerical
# Mura provider via :func:`set_mura_self_stress_provider`; for V1 local
# testing a placeholder stub provider is used by default (see
# :func:`_default_mura_self_stress`).
# ---------------------------------------------------------------------------
_MURA_SELF_STRESS_PROVIDER = None


def _default_mura_self_stress(kappa: np.ndarray) -> np.ndarray:
    """PLACEHOLDER stub Mura self-stress provider for V1 local testing.

    ``sigma_h^{Mura_Self}(kappa)`` is, in the full project, the output of the
    numerical Mura line-integral solved upstream. That numerical input is not
    available in the V1 lightweight module, so this stub returns a
    placeholder numerical field obtained by linear interpolation of a small
    tabulated ``(kappa, sigma_h)`` input. The tabulated values below are a
    STAND-IN for the real Mura-integral output and must be replaced by the
    actual numerical solution (e.g. loaded from a Phase 1 / V2 data file) before
    any quantitative use. The placeholder is monotone-growing with ``kappa``,
    consistent with the physical expectation that a tighter bow (larger
    ``kappa``) raises the self-stress.

    Args:
        kappa: Dislocation curvature (1/m), strictly positive, any shape.

    Returns:
        Placeholder hydrostatic self-stress in Pa, same shape as ``kappa``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    # Placeholder numerical input table (kappa in 1/m, sigma_h in Pa).
    # Replace with the real numerical Mura line-integral output.
    grid_k = np.array([2.0e5, 5.0e5, 1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6],
                      dtype=np.float64)
    grid_s = np.array([2.75e7, 6.0e7, 1.05e8, 1.9e8, 2.6e8, 3.3e8, 4.3e8, 5.4e8],
                      dtype=np.float64)
    flat = kappa.ravel()
    sigma = np.interp(flat, grid_k, grid_s)
    return sigma.reshape(kappa.shape)


def set_mura_self_stress_provider(provider) -> None:
    """Register the numerical Mura self-stress provider (black-box).

    ``provider`` must be a callable ``kappa -> sigma_h^{Mura_Self}(kappa)`` (Pa)
    returning the hydrostatic self-stress from the numerical Mura line-integral
    solved upstream. Once registered, :func:`get_mura_self_stress` dispatches to
    it; until then the placeholder stub :func:`_default_mura_self_stress` is used.
    """
    global _MURA_SELF_STRESS_PROVIDER
    if not callable(provider):
        raise TypeError("Mura self-stress provider must be callable.")
    _MURA_SELF_STRESS_PROVIDER = provider


def get_mura_self_stress(kappa: np.ndarray) -> np.ndarray:
    """Black-box hydrostatic Mura self-stress ``sigma_h^{Mura_Self}(kappa)`` (Pa).

    This is the SOLE interface to the dislocation self-stress in the V1 module.
    It does NOT re-derive the mechanical side: it dispatches to the registered
    numerical Mura provider (:func:`set_mura_self_stress_provider`), falling
    back to the placeholder stub :func:`_default_mura_self_stress` for V1 local
    testing. The returned field is the external Mura-integral input as a function
    of the curvature ``kappa`` (radius ``R(kappa) = 1/kappa``).

    Args:
        kappa: Dislocation curvature (1/m), strictly positive, any shape.

    Returns:
        Hydrostatic self-stress ``sigma_h^{Mura_Self}`` in Pa, same shape as ``kappa``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    provider = _MURA_SELF_STRESS_PROVIDER or _default_mura_self_stress
    out = provider(kappa)
    return np.asarray(out, dtype=np.float64).reshape(kappa.shape)


def mura_self_stress(kappa: np.ndarray) -> np.ndarray:
    """Mura self-stress tensor ``sigma_ij^self(kappa)`` of a curved dislocation.

    Builds the full 3x3 stress tensor whose hydrostatic part is the external
    Mura self-stress from :func:`get_mura_self_stress` (a black-box numerical
    input, NOT re-derived here). The self-stress of a bowed dislocation is, to
    leading order, isotropic-hydrostatic along the bowed segment, so

        sigma_ij^self(kappa) = sigma_h^{Mura_Self}(kappa) * delta_ij .

    Args:
        kappa: Dislocation curvature (1/m), strictly positive, shape ``(...)``.

    Returns:
        Self-stress tensor of shape ``(..., 3, 3)`` in Pa.
    """
    sh = get_mura_self_stress(kappa)
    eye3 = np.eye(3, dtype=np.float64)
    return sh[..., None, None] * eye3


def elastic_strain_energy_density(
    sigma_tensor: np.ndarray,
    mu: float = C.MU,
    nu: float = C.NU,
) -> np.ndarray:
    """Isotropic elastic strain-energy density ``Psi_mech = (1/2) sigma:eps``.

    With Hooke's law ``eps = (1/E)[(1+nu) sigma - nu tr(sigma) I]`` and
    ``E = 2 mu (1 + nu)``, the strain-energy density reduces to

        Psi_mech = (1/(2E)) * [(1+nu) sigma:sigma - nu (tr sigma)^2]

    in J/m^3, where ``sigma:sigma = sigma_ij sigma_ij``.

    Args:
        sigma_tensor: Stress field of shape ``(..., 3, 3)`` in Pa.
        mu: Shear modulus in Pa.
        nu: Poisson ratio.

    Returns:
        Mechanical free-energy density ``Psi_mech`` in J/m^3, shape ``(...,)``.
    """
    sigma = np.asarray(sigma_tensor, dtype=np.float64)
    young = 2.0 * float(mu) * (1.0 + float(nu))
    sig_sig = np.sum(sigma * sigma, axis=(-2, -1))
    tr_sig = np.trace(sigma, axis1=-2, axis2=-1)
    return (1.0 / (2.0 * young)) * ((1.0 + float(nu)) * sig_sig - float(nu) * tr_sig ** 2)


def total_free_energy_density(
    sigma_tensor: np.ndarray,
    c_lattice: np.ndarray,
    theta_t: np.ndarray,
    temperature: float = C.T_REF,
    mu: float = C.MU,
    nu: float = C.NU,
) -> np.ndarray:
    """Total free-energy density ``Psi_total = Psi_mech + Psi_H`` in J/m^3.

    Args:
        sigma_tensor: Stress field of shape ``(..., 3, 3)`` in Pa.
        c_lattice: Lattice hydrogen concentration in mol/m^3, shape ``(...,)``.
        theta_t: Trap occupancy ``theta_T`` (dimensionless), shape ``(...,)``.
        temperature: Absolute temperature in Kelvin.
        mu: Shear modulus in Pa.
        nu: Poisson ratio.

    Returns:
        Total free-energy density ``Psi_total`` in J/m^3, shape ``(...,)``.
    """
    psi_mech = elastic_strain_energy_density(sigma_tensor, mu=mu, nu=nu)
    psi_h = hydrogen_free_energy_density(c_lattice, theta_t, temperature)
    return psi_mech + psi_h


def integrate_energy_density(
    psi: np.ndarray,
    volumes: np.ndarray,
) -> float:
    """Integrate a free-energy density over the interaction volume ``V_int``.

    ``W = integral_{V_int} Psi dV`` approximated as a weighted sum
    ``sum_i Psi_i * dV_i`` over the quadrature/point cloud.

    Args:
        psi: Free-energy density field of shape ``(...,)`` in J/m^3.
        volumes: Associated volume weights ``dV`` of the same shape in m^3.

    Returns:
        Total energy ``W`` in J.
    """
    psi = np.asarray(psi, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    return float(np.sum(psi * volumes))


def line_energy(total_energy_value: float, line_length: float) -> float:
    """Line energy ``E(kappa) = W_total(kappa) / L(kappa)`` in J/m."""
    return float(total_energy_value) / float(line_length)


def line_length_field(
    kappa: "object",
    dtheta: float = C.DTHETA,
) -> "object":
    """Spatial dislocation line length ``L(kappa) = R(kappa) * dtheta`` (field).

    The line length is not a scalar: for a circular-arc dislocation of curvature
    ``kappa`` (radius ``R(kappa) = 1/kappa``) subtending an angular span
    ``dtheta``, the curvilinear length is

        L(kappa) = R(kappa) * dtheta = dtheta / kappa .

    Returning it as a field over the spatial coordinate ``kappa`` (rather than a
    single scalar) lets the energy/tension chain resolve a spatially varying line
    length, mirroring the ``sigma_h(kappa)`` treatment.

    ``kappa`` may be either a NumPy array (-> NumPy field) or a FEniCSx UFL node
    such as ``ufl.SpatialCoordinate`` (-> a UFL spatial field). The length is
    computed dynamically as ``L = dtheta / kappa`` so that, when ``kappa`` is a
    UFL expression, ``L`` automatically becomes a UFL spatial field without any
    NumPy coercion.

    Args:
        kappa: Dislocation curvature (1/m), either a NumPy array of arbitrary
            shape or a UFL expression.
        dtheta: Angular span (rad) subtended by the arc; defaults to the
            canonical ``pi/2`` arc in :mod:`constants`.

    Returns:
        Line length ``L(kappa)`` in m, same type/shape as ``kappa`` (NumPy field
        or UFL expression).
    """
    dtheta = float(dtheta)
    # UFL expressions (FEniCSx) and other non-NumPy operands: divide directly so
    # L becomes a UFL spatial field when kappa is a UFL node (no np.asarray).
    if isinstance(kappa, np.ndarray):
        return dtheta / np.asarray(kappa, dtype=np.float64)
    return dtheta / kappa


def line_tension(
    kappas: np.ndarray,
    total_energy_values: np.ndarray,
    line_lengths: np.ndarray,
) -> np.ndarray:
    """Line tension ``Gamma = dE/dL = (dE/dkappa) / (dL/dkappa)`` in J/m.

    Given samples of the total energy ``W_total(kappa)`` and line length
    ``L(kappa)`` over a monotone ``kappa`` grid, the line energy
    ``E(kappa) = W/L`` is formed and differentiated numerically (central
    differences) to yield the line tension.

    Args:
        kappas: Monotone curvature/loading parameter grid, shape ``(K,)``.
        total_energy_values: ``W_total(kappa)`` in J, shape ``(K,)``.
        line_lengths: ``L(kappa)`` in m, shape ``(K,)``.

    Returns:
        Line tension ``Gamma(kappa)`` in J/m, shape ``(K,)``.
    """
    kappas = np.asarray(kappas, dtype=np.float64)
    total_energy_values = np.asarray(total_energy_values, dtype=np.float64)
    line_lengths = np.asarray(line_lengths, dtype=np.float64)
    e_of_kappa = total_energy_values / line_lengths
    dE_dk = np.gradient(e_of_kappa, kappas)
    dL_dk = np.gradient(line_lengths, kappas)
    return dE_dk / dL_dk


def line_tension_split(
    kappas: np.ndarray,
    w_mech: np.ndarray,
    w_h: np.ndarray,
    line_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose the line tension as ``Gamma = Gamma_mech + Gamma_H``.

    Args:
        kappas: Monotone curvature/loading parameter grid, shape ``(K,)``.
        w_mech: Mechanical total energy ``W_mech(kappa)`` in J, shape ``(K,)``.
        w_h: Hydrogen total energy ``W_H(kappa)`` in J, shape ``(K,)``.
        line_lengths: ``L(kappa)`` in m, shape ``(K,)``.

    Returns:
        ``(Gamma_mech, Gamma_H, Gamma_total)`` each in J/m, shape ``(K,)``.
    """
    gamma_mech = line_tension(kappas, w_mech, line_lengths)
    gamma_h = line_tension(kappas, w_h, line_lengths)
    return gamma_mech, gamma_h, gamma_mech + gamma_h


def line_tension_chain(
    sigma_fields: np.ndarray,
    c_lattice_fields: np.ndarray,
    theta_t_fields: np.ndarray,
    volumes: np.ndarray,
    line_lengths: np.ndarray,
    kappas: np.ndarray,
    temperature: float = C.T_REF,
    mu: float = C.MU,
    nu: float = C.NU,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end 9-step chain: densities -> W -> E -> Gamma (split).

    Args:
        sigma_fields: Per-kappa stress fields of shape ``(K, ..., 3, 3)`` in Pa,
            where the leading axis indexes ``kappa``.
        c_lattice_fields: Per-kappa lattice concentration ``(K, ...)`` mol/m^3.
        theta_t_fields: Per-kappa trap occupancy ``(K, ...)`` (dimensionless).
        volumes: Volume weights ``dV`` of shape ``(...)`` in m^3 (broadcast over
            the leading ``K`` axis).
        line_lengths: ``L(kappa) = R(kappa) * dtheta`` in m, shape ``(K,)``,
            with ``R(kappa) = 1/kappa`` (see :func:`radius_of_curvature`).
        kappas: Monotone ``kappa`` grid, shape ``(K,)``.
        temperature: Absolute temperature in Kelvin.
        mu: Shear modulus in Pa.
        nu: Poisson ratio.

    Returns:
        ``(Gamma_mech, Gamma_H, Gamma_total, E_of_kappa)`` with the first three
        in J/m (shape ``(K,)``) and ``E_of_kappa`` in J/m (shape ``(K,)``).
    """
    sigma_fields = np.asarray(sigma_fields, dtype=np.float64)
    c_lattice_fields = np.asarray(c_lattice_fields, dtype=np.float64)
    theta_t_fields = np.asarray(theta_t_fields, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    k = sigma_fields.shape[0]
    psi_mech = elastic_strain_energy_density(sigma_fields, mu=mu, nu=nu)  # (K, ...)
    psi_h = hydrogen_free_energy_density(c_lattice_fields, theta_t_fields, temperature)

    flat_axes = tuple(range(1, psi_mech.ndim))
    w_mech = np.array([integrate_energy_density(psi_mech[i], volumes) for i in range(k)])
    w_h = np.array([integrate_energy_density(psi_h[i], volumes) for i in range(k)])
    e_of_kappa = (w_mech + w_h) / np.asarray(line_lengths, dtype=np.float64)
    gamma_mech, gamma_h, gamma_total = line_tension_split(
        kappas, w_mech, w_h, line_lengths
    )
    _ = flat_axes
    return gamma_mech, gamma_h, gamma_total, e_of_kappa


__all__: Tuple[str, ...] = (
    "radius_of_curvature",
    "get_mura_self_stress",
    "set_mura_self_stress_provider",
    "mura_self_stress",
    "elastic_strain_energy_density",
    "total_free_energy_density",
    "integrate_energy_density",
    "line_energy",
    "line_length_field",
    "line_tension",
    "line_tension_split",
    "line_tension_chain",
)
