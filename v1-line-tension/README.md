# V1 — Line Tension Validation (Theoretical Baseline)

Lightweight, standalone proof of the hydrogen-induced dislocation
line-tension degradation at a **fixed state**:

- `T = 298.15 K` (25 °C)
- `P_ext = 35 MPa`

Dependencies: **numpy, scipy, matplotlib** only. No FEniCSx, no PyTorch.

## Run

```bash
python v1-line-tension/analyze_line_tension.py
```

Prints the fixed-state thermodynamic quantities and writes
`figures/line_tension_vs_curvature.png`.

## Files

- `analyze_line_tension.py` — the proof script (Abel-Noble → Sieverts → Oriani → Γ(κ))
- `constants.py` — 30CrMo material registry + Abel-Noble co-volume
- `hydrogen.py` — Abel-Noble fugacity, Sieverts, Oriani, free-energy densities
- `line_tension.py` — line-tension chain helpers
- `scr.md` — script console output
- `line-tension.md` — stress values from the model
