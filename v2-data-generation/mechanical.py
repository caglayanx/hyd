"""Mechanical energy block for the V2 coupled mech-hydrogen simulator.

Computes the mechanical energy density from the pre-calculated Mura-based
self-stress tensor ``sigma_ij^self(kappa)`` of a single curved dislocation:

    Psi_self = (1/2) sigma_ij^self epsilon_ij^self
    Psi_ext  = - sigma_ij^ext epsilon_ij^self
    Psi_mech = Psi_self + Psi_ext
    E_mech   = integral_V Psi_mech dV

where ``epsilon_ij^self`` follows from Hooke's law with the 30CrMo elastic
constants (isotropic ``mu``, ``nu`` or full ``C_ijkl`` / ``S_ijkl``). The
stress field itself is taken as an INPUT from the prior Mura-based solution; it
is NOT re-derived here.

This module is solver-agnostic (numpy only). The FEniCSx integration lives in
``solver.py``.
"""
from __future__ import annotations

import numpy as np

import constants as C


def elastic_strain_energy_density(sigma_tensor: np.ndarray, mu: float = C.MU,
                                   nu: float = C.NU) -> np.ndarray:
    """Self elastic energy density ``Psi_self = (1/2) sigma:epsilon`` (J/m^3).

    With Hooke's law ``epsilon = (1/E)[(1+nu) sigma - nu tr(sigma) I]`` and
    ``E = 2 mu (1 + nu)``, this reduces to

        Psi_self = (1/(2E)) [(1+nu) sigma:sigma - nu (tr sigma)^2].
    """
    sigma = np.asarray(sigma_tensor, dtype=np.float64)
    young = 2.0 * mu * (1.0 + nu)
    sig_sig = np.sum(sigma * sigma, axis=(-2, -1))
    tr_sig = np.trace(sigma, axis1=-2, axis2=-1)
    return (1.0 / (2.0 * young)) * ((1.0 + nu) * sig_sig - nu * tr_sig ** 2)


def external_stress_energy_density(sigma_self: np.ndarray, sigma_ext: np.ndarray,
                                    mu: float = C.MU, nu: float = C.NU) -> np.ndarray:
    """External-stress interaction energy ``Psi_ext = -sigma_ext:epsilon_self``.

    ``Psi_ext = -(1/E) [(1+nu) sigma_ext:sigma_self - nu tr(sigma_ext) tr(sigma_self)]``.
    """
    young = 2.0 * mu * (1.0 + nu)
    s = np.asarray(sigma_self, dtype=np.float64)
    e = np.asarray(sigma_ext, dtype=np.float64)
    cross = np.sum(e * s, axis=(-2, -1))
    return -(1.0 / young) * ((1.0 + nu) * cross - nu * np.trace(e, axis1=-2, axis2=-1)
                              * np.trace(s, axis1=-2, axis2=-1))


def mechanical_energy_density(sigma_self: np.ndarray, sigma_ext: np.ndarray,
                               mu: float = C.MU, nu: float = C.NU) -> np.ndarray:
    """Total mechanical energy density ``Psi_mech = Psi_self + Psi_ext``."""
    return (elastic_strain_energy_density(sigma_self, mu, nu)
            + external_stress_energy_density(sigma_self, sigma_ext, mu, nu))


def integrate_energy_density(psi: np.ndarray, volumes: np.ndarray) -> float:
    """``E = integral_V Psi dV`` approximated as ``sum_i Psi_i dV_i`` (J)."""
    return float(np.sum(np.asarray(psi, dtype=np.float64) * np.asarray(volumes, dtype=np.float64)))


__all__ = (
    "elastic_strain_energy_density",
    "external_stress_energy_density",
    "mechanical_energy_density",
    "integrate_energy_density",
)
