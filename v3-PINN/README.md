# V3 — PINN (Physics-Informed Neural Network)

PyTorch surrogate that learns the lattice hydrogen concentration
`c_L` from the V2 Zarr dataset, constrained by the V2 transport PDE.

## Architecture

Fourier-feature MLP mapping

```
(x, y, t, P, T, sigma_h) -> c_L
```

## Loss

```
L = L_data + lambda_bc * L_bc + lambda_pde * L_pde
L_data : MSE against V2 c_L samples
L_bc   : MSE enforcing C_s(P, T) on the notch-surface mask
L_pde  : autograd residual of dC_L/dt = -div(J) (V2 transport equation)
```

## Files

- `pinn.py` — Fourier-feature MLP surrogate
- `dataset.py` — Zarr v3 PyTorch dataset loader (streams V2 samples)
- `train.py` — Adam training loop with data / BC / PDE residual losses

## Run (requires PyTorch + V2 dataset)

```bash
python v3-PINN/train.py --dataset ../v2-data-generation/data/dataset.zarr --epochs 5000
```

`pinn.py` imports PyTorch lazily, so the module imports cleanly without it.
