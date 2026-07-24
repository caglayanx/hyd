# V2 — Data Generation (Coupled Mech-Hydrogen Simulator)

High-fidelity coupled mechanical–hydrogen transient solver for a single
curved dislocation, using **FEniCSx/DOLFINx** and exporting to a chunked
**Zarr v3** dataset.

## Thermodynamic chain (Peng-Robinson EOS)

```
P, T -> PR-EOS -> Z -> phi -> f -> C_s -> C_L(P,T,sigma_h) -> theta_T -> C_T
     -> Psi_H ;  Psi_mech = Psi_self + Psi_ext ;  Psi_tot = Psi_mech + Psi_H
     -> E_tot = integral_V Psi_tot dV -> Gamma = E_tot / L
```

## Files

- `constants.py` — H2 critical properties (PR-EOS) + 30CrMo elastic/trap constants
- `hydrogen_thermo.py` — **Peng-Robinson EOS + Oriani trapping** (the thermodynamic core)
- `mechanical.py` — self/external mechanical energy density from `sigma_ij^self(kappa)`
- `transport.py` — stress-assisted flux `J` and `dC_L/dt = -div(J)`
- `solver.py` — DOLFINx reference solver (lumped-mass theta-scheme) [scaffold]
- `zarr_writer.py` — chunked Zarr v3 exporter [scaffold]

## Run (requires FEniCSx stack)

```bash
python v2-data-generation/solver.py --out data/dataset.zarr
```

`hydrogen_thermo.py` and `mechanical.py` are numpy-only and can be unit-tested
without FEniCSx.
