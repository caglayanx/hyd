# hyd — Hydrogen-Effected Line Tension and Transport Model for 30CrMo Steel

Monorepo with three independent, versioned stages of the coupled
mechanical–hydrogen dislocation line-tension model for
**30CrMo (30CrMo4 / AISI 4130)** high-pressure hydrogen storage steel.

```
hyd/
  v1-line-tension/      # Theoretical baseline (lightweight, numpy/scipy/matplotlib)
  v2-data-generation/   # High-fidelity coupled mech-hydrogen simulator (FEniCSx)
  v3-PINN/              # Physics-informed neural-network surrogate (PyTorch)
```

Each stage is self-contained and can be developed/run independently.

## Stage map

| Stage | Directory | Stack | Role |
|-------|-----------|-------|------|
| V1 | `v1-line-tension/` | numpy, scipy, matplotlib | Fixed-state (35 MPa, 298.15 K) Abel-Noble line-tension proof |
| V2 | `v2-data-generation/` | FEniCSx/DOLFINx, Zarr v3 | Coupled mech-hydrogen transient solver, Zarr dataset export |
| V3 | `v3-PINN/` | PyTorch | Fourier-feature PINN surrogate trained on the V2 dataset |

## Final thermodynamic chain (all stages)

```
P, T -> EOS -> Z -> phi -> f -> C_s -> C_L(P,T,sigma_h) -> theta_T -> C_T
     -> Psi_H -> Psi_tot = Psi_mech + Psi_H -> E_tot -> Gamma = E_tot / L
```

V1 uses the Abel-Noble EOS; V2/V3 use the Peng-Robinson EOS.
