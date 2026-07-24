"""
Material Parameter Registry for 30CrMo (30CrMo4 / AISI 4130) Storage Steel.

Application: High-Pressure Hydrogen Storage Vessels & Pressure Equipment.

Sources: ISO 11114-4, San Marchi et al. (Sandia National Labs), Hirth & Lothe.

All quantities are expressed in strict SI base units unless noted otherwise.
This module is intentionally free of runtime side-effects: importing it only
registers the parameter values below as module-level constants so that every
other subpackage (core, sim, data, models) can reference a single source of
truth for material, physical, and numerical parameters.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Universal Constants
# ---------------------------------------------------------------------------
R: float = 8.3145          # Universal Gas Constant (J / (mol * K))
T_REF: float = 298.15      # Reference Working Temperature (K) -> 25 °C

# ---------------------------------------------------------------------------
# Elastic & Dislocation Constants (30CrMo BCC Fe steel)
# ---------------------------------------------------------------------------
MU: float = 79.5e9         # Shear Modulus (Pa) - Source: Sandia / ISO 11114
NU: float = 0.29           # Poisson Ratio - Source: ISO 11114
E: float = 205.0e9         # Young Modulus (Pa) - Source: ISO 11114
BURGERS_B: float = 2.48e-10  # Burgers vector magnitude b (m) for BCC Fe
B: float = BURGERS_B       # Backward-compatible alias for the Burgers vector
CORE_RADIUS: float = BURGERS_B  # Dislocation Core Radius r_c = b (m)

# ---------------------------------------------------------------------------
# Lattice Hydrogen Transport Parameters (Tempered Cr-Mo Matrix)
# ---------------------------------------------------------------------------
DIFFUSION_D: float = 4.0e-11  # Lattice diffusion coefficient at 298 K (m^2 / s)
D_L: float = DIFFUSION_D      # Alias used by the FEniCSx solver
# Arrhenius pair kept for documentation / temperature extrapolation only.
D_0: float = 1.40e-7         # Pre-exponential Diffusion Coefficient (m^2 / s)
E_A: float = 16000.0         # Activation Energy (J / mol) -> 16.0 kJ/mol
E_D: float = float(-R * T_REF * np.log(DIFFUSION_D / D_0))  # Arrhenius activation energy (J/mol), self-consistent: D_0*exp(-E_D/(R*T_REF)) == D_L
OMEGA: float = 2.0e-6         # Partial Molar Volume of Hydrogen (m^3 / mol)
V_BAR_H: float = OMEGA        # Backward-compatible alias (symbol V_bar_H)
V_H: float = V_BAR_H          # Alias used by the FEniCSx weak form
R_GAS: float = R              # Alias for the gas constant used in the weak form

# ---------------------------------------------------------------------------
# Operating Conditions (High-Pressure H2 Storage Vessel) - FIXED V1 STATE
# ---------------------------------------------------------------------------
# Version 1 hardcodes the thermodynamic state to a single operating point:
#   T = 298.15 K (25 °C), P_ext = 35 MPa (35e6 Pa).
# The H2 fugacity f(P,T) is NOT hardcoded here; it is derived from the Abel-Noble
# real-gas equation of state in :mod:`hydrogen` (see ``abel_noble_*``).
# This replaces the earlier ad-hoc ``SQRT_F`` constant with a physically
# consistent fugacity coefficient phi = exp(b*P/(R*T)) > 1.
# ---------------------------------------------------------------------------
T_OP: float = 298.15           # Operating temperature (K) = 25 °C  (FIXED V1)
P_OP: float = 35.0e6            # Operating pressure (Pa) = 35 MPa    (FIXED V1)
# Abel-Noble co-volume of H2 (m^3/mol). San Marchi et al. (Sandia) / ISO 11114-4
# use b_AN = 15.8 cm^3/mol for the Abel-Noble EOS of high-pressure hydrogen.
B_AN: float = 15.8e-6           # Abel-Noble co-volume b (m^3/mol) for H2
# Legacy fugacity placeholders kept for backward compatibility only; the V1
# chain recomputes f from Abel-Noble via :func:`hydrogen.abel_noble_fugacity`.
SQRT_F: float = 18.7e3         # DEPRECATED: legacy sqrt(fugacity) (Pa^0.5)
FUGACITY: float = float(SQRT_F ** 2)  # DEPRECATED: legacy H2 fugacity f (Pa)
FUGACITY_COEFF: float = FUGACITY / P_OP  # DEPRECATED: legacy phi = f / P

# ---------------------------------------------------------------------------
# Sieverts Solubility (Hirth; San Marchi et al., ferritic Cr-Mo steel)
#   C_s = K_s(T) * sqrt(f)   (Sieverts surface concentration, mol / m^3)
# ---------------------------------------------------------------------------
# K_S_298 is the SI (Pa^0.5-based) form of 0.071 mol/(m^3 * MPa^0.5):
#   0.071 / sqrt(1e6) = 0.071 / 1000 = 7.1e-5 mol/(m^3 * Pa^0.5).
K_S_298: float = 7.1e-5        # Sieverts coefficient at 298.15 K (mol / (m^3 * Pa^0.5))
K_S_REF: float = K_S_298       # Alias: K_s at the reference temperature
H_S: float = 28600.0          # Heat of solution of H in alpha-Fe (J/mol)
# Pre-exponential calibrated so K_S_0*exp(-H_S/(R*T_REF)) == K_S_298.
K_S_0: float = K_S_298 / float(np.exp(-H_S / (R * T_REF)))

# ---------------------------------------------------------------------------
# Trapping Kinetics & Binding Energies
# ---------------------------------------------------------------------------
DELTA_G_B: float = 30000.0    # Trap binding free energy (J/mol) -> 30 kJ/mol
E_B_DISLOC: float = DELTA_G_B  # Backward-compatible alias
ETA_D: float = 1.0 / BURGERS_B  # Trap sites per unit line length (sites / m) = 1/b
# Interstitial lattice site density (mol / m^3) - the free-energy chain works in
# molar units so that R*T*concentration has units J/m^3.
N_L: float = 8.46e5           # Interstitial site density (mol / m^3)
N_L_MOLAR: float = N_L        # Molar lattice site density (mol / m^3) == N_L
N_AVOG: float = 6.02214076e23  # Avogadro number (1/mol)
RHO_0: float = 1.0e13         # Baseline Dislocation Density for Quenched/Tempered 30CrMo (m / m^3)

# Dislocation effective capture cross-section A_eff (m^2). Following the
# dislocation-core-radius approximation r_c = b, the effective cross-section
# over which the line-trapped population is spread to obtain the volume
# concentration c_T^vol = C_T^line / A_eff is A_eff = pi * r_c^2 ~ 1.93e-19 m^2.
R_C: float = BURGERS_B            # Dislocation core radius r_c = b (m) = 2.48e-10
A_EFF: float = float(np.pi * R_C ** 2)  # Effective cross-section A_eff (m^2) ~ 1.93e-19

# Angular span (rad) of the canonical circular-arc dislocation at the notch tip
# (theta_max - theta_min = pi/4 - (-pi/4) = pi/2). Used as the default ``dtheta``
# in the spatial line-length field L(R) = R * dtheta.
DTHETA: float = np.pi / 2

# ---------------------------------------------------------------------------
# Boundary & Numerical Constants
# ---------------------------------------------------------------------------
C_L_0: float = 1.0        # Reference Surface Hydrogen Concentration (mol / m^3)
EPS: float = 1e-30         # Numerical tolerance
# Regularisation length (m) for the symbolic sigma_h(R) stress field, used to
# remove the R -> 0 singularity at the notch/dislocation centre in both the
# UFL variational form and the NumPy free-energy chain.
SIGMA_H_EPS: float = 1.0e-12


__all__ = [
    "R",
    "T_REF",
    "MU",
    "NU",
    "E",
    "BURGERS_B",
    "B",
    "CORE_RADIUS",
    "DIFFUSION_D",
    "D_0",
    "E_D",
    "R_GAS",
    "V_H",
    "D_L",
    "E_A",
    "OMEGA",
    "V_BAR_H",
    "T_OP",
    "P_OP",
    "B_AN",
    "SQRT_F",
    "FUGACITY",
    "FUGACITY_COEFF",
    "K_S_298",
    "K_S_REF",
    "H_S",
    "K_S_0",
    "DELTA_G_B",
    "E_B_DISLOC",
    "ETA_D",
    "N_L",
    "N_L_MOLAR",
    "N_AVOG",
    "RHO_0",
    "R_C",
    "A_EFF",
    "DTHETA",
    "C_L_0",
    "EPS",
    "SIGMA_H_EPS",
]
