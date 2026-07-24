"""Material and physical constants for the V2 coupled mech-hydrogen simulator.

Two constant families are registered:

  * Hydrogen real-gas critical properties (for the Peng-Robinson EOS), and
    Sieverts / Oriani / transport parameters.
  * 30CrMo (30CrMo4 / AISI 4130) elastic and dislocation constants.

All quantities are in strict SI base units unless noted otherwise. This module
is free of runtime side-effects: importing it only registers the values below.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Universal constants
# ---------------------------------------------------------------------------
R: float = 8.3145                # Universal gas constant (J / (mol K))
N_AVOG: float = 6.02214076e23    # Avogadro number (1 / mol)

# ---------------------------------------------------------------------------
# Hydrogen real-gas critical properties (Peng-Robinson EOS inputs).
# Source: NIST / literature for molecular hydrogen (H2).
# ---------------------------------------------------------------------------
T_C_H2: float = 33.145           # Critical temperature of H2 (K)
P_C_H2: float = 1.2964e6         # Critical pressure of H2 (Pa)
OMEGA_H2: float = -0.216          # Acentric factor of H2 (dimensionless)

# ---------------------------------------------------------------------------
# Hydrogen solubility & transport (Sieverts, stress-assisted diffusion).
# ---------------------------------------------------------------------------
K_0: float = 7.1e-5 / float(np.exp(-28600.0 / (8.3145 * 298.15)))  # Sieverts pre-exponential (mol/(m^3 Pa^0.5)); calibrated so K_0*exp(-H_s/(R*T_REF)) == K_S_298
H_S: float = 28600.0             # Heat of solution of H in alpha-Fe (J/mol)
K_S_298: float = 7.1e-5           # Sieverts coefficient at 298.15 K (mol/(m^3 Pa^0.5))
D_L: float = 4.0e-11             # Lattice diffusion coefficient at 298 K (m^2/s)
V_H: float = 2.0e-6              # Partial molar volume of hydrogen (m^3/mol)

# ---------------------------------------------------------------------------
# Hydrogen trapping (Oriani equilibrium).
# ---------------------------------------------------------------------------
DELTA_G_B: float = 30000.0       # Trap binding free energy (J/mol) -> 30 kJ/mol
N_L: float = 8.46e5               # Interstitial lattice site density (mol/m^3)
N_T: float = 1.0e3                # Effective trap site density (mol/m^3)

# ---------------------------------------------------------------------------
# 30CrMo elastic & dislocation constants (BCC Fe).
# ---------------------------------------------------------------------------
MU: float = 79.5e9               # Shear modulus (Pa)
NU: float = 0.29                  # Poisson ratio
E_YOUNG: float = 205.0e9          # Young modulus (Pa)
BURGERS_B: float = 2.48e-10       # Burgers vector magnitude b (m)
CORE_RADIUS: float = BURGERS_B    # Dislocation core radius r_c = b (m)
RHO_0: float = 1.0e13             # Baseline dislocation density (1/m^2)

# ---------------------------------------------------------------------------
# Reference operating point (kept for convenience; V2 is parameterised over
# P, T but defaults to the storage-vessel state).
# ---------------------------------------------------------------------------
T_REF: float = 298.15            # Reference temperature (K) = 25 °C
P_REF: float = 35.0e6            # Reference pressure (Pa) = 35 MPa

__all__ = [
    "R", "N_AVOG",
    "T_C_H2", "P_C_H2", "OMEGA_H2",
    "K_0", "H_S", "K_S_298", "D_L", "V_H",
    "DELTA_G_B", "N_L", "N_T",
    "MU", "NU", "E_YOUNG", "BURGERS_B", "CORE_RADIUS", "RHO_0",
    "T_REF", "P_REF",
]
