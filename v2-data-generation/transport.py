"""Hydrogen transport & trapping for the V2 coupled mech-hydrogen simulator.

Implements the stress-assisted diffusion PDE for the lattice hydrogen
concentration ``C_L``:

    J      = -D grad(C_L) + (D C_L V_H / (R T)) grad(sigma_h)
    dC_L/dt = -div(J)

with the Oriani equilibrium (see :mod:`hydrogen_thermo`) coupling the trapped
population ``C_T`` to ``C_L`` instantaneously. The surface Dirichlet boundary
condition is ``C_L = C_s(P, T)`` on the notch surface.

This module provides the flux, the time derivative, and the Oriani coupling in
pure NumPy (for unit testing and reference). The FEniCSx/UFL weak form and the
lumped-mass theta-scheme time integrator live in ``solver.py``.
"""
from __future__ import annotations

import numpy as np

import constants as C
import hydrogen_thermo as ht


def diffusion_coefficient(T: float, D_0: float = C.D_0, E_A: float = C.E_A) -> float:
    """Temperature-dependent lattice diffusivity ``D(T)`` (m^2/s), Arrhenius.

    ``D(T) = D_0 exp(-E_A / (R T))``. Calibrated so that
    ``D_0 exp(-E_A/(R*298.15)) == D_L`` (4.0e-11 m^2/s at 298 K).
    """
    return D_0 * float(np.exp(-E_A / (C.R * float(T))))


def hydrogen_flux(grad_cL: np.ndarray, cL: np.ndarray, grad_sigma_h: np.ndarray,
                  T: float, D: float = C.D_L, V_H: float = C.V_H) -> np.ndarray:
    """Stress-assisted hydrogen flux ``J`` (mol / (m^2 s)).

    ``J = -D grad(C_L) + (D C_L V_H / (R T)) grad(sigma_h)``.
    """
    return (-D * np.asarray(grad_cL, dtype=np.float64)
            + (D * np.asarray(cL, dtype=np.float64) * V_H / (C.R * float(T)))
            * np.asarray(grad_sigma_h, dtype=np.float64))


def transport_time_derivative(div_J: np.ndarray) -> np.ndarray:
    """``dC_L/dt = -div(J)`` (mol / (m^3 s))."""
    return -np.asarray(div_J, dtype=np.float64)


def oriari_update(cL: np.ndarray, T: float) -> np.ndarray:
    """Instantaneous Oriani trap concentration ``C_T = N_T theta_T`` (mol/m^3)."""
    return ht.trapped_concentration(cL, T)


__all__ = ("diffusion_coefficient", "hydrogen_flux", "transport_time_derivative", "oriari_update")
