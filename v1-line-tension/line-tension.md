# line-tension.md — Stress values in the V1 line-tension model

Fixed state: `T = 298.15 K` (25 °C), `P_ext = 35 MPa`.
Material: 30CrMo (30CrMo4 / AISI 4130) — `mu = 79.5 GPa`, `nu = 0.29`, `b = 2.48e-10 m`, core radius `r_c = b`.

## Static edge dislocation field

`sigma_h(r) = A / (r + r_c)`, with `A = mu*b / (2*pi*(1 - nu)) = 4.42 Pa*m`.

| r | sigma_h |
|---|---|
| core (r = r_c = b) | 8.91 GPa |
| 1 nm | 3.54 GPa |
| 10 nm | 0.43 GPa |
| 100 nm | 0.04 GPa |

## Curvature-induced line-tension stress

`sigma_LT(k) = mu*b*k / (2*(1 - nu))`.

| kappa (1/m) | R (nm) | sigma_LT |
|---|---|---|
| 2.0e5 | 5000 | 2.78 MPa (0.003 GPa) |
| 7.6e5 | 1320 | 10.5 MPa (0.011 GPa) |
| 2.7e6 | 371 | 37.4 MPa (0.037 GPa) |
| 2.8e7 | 35.5 | 391 MPa (0.391 GPa) |

## Total driving stress

`sigma_total(r, k) = sigma_h(r) + sigma_LT(k)` — at the core this is ~9 GPa plus `sigma_LT(k)`.
