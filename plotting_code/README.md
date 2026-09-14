# Paper figure plotting

This folder contains the notebooks and processed data used to reproduce the figures presented in the PolySurrogate paper.

| File | Figure |
| --- | --- |
| `training_curves.ipynb` | PNN and FNN training and validation curves |
| `testing_hist.ipynb` | Test-set mean absolute error histograms |
| `jacobian.ipynb` | Surrogate and exact PyTheus Jacobian sensitivity plots |
| `initial_to_final_fid.ipynb` | Initial and final inverse-design fidelity distributions |
| `runtime.ipynb` | PyTheus and surrogate runtime comparison |
| `inverse_design_evolution.py` | Loss trajectory and graph evolution during inverse design |

All processed plotting inputs are stored under `Data/`. The inverse-design evolution trajectory is stored under `Data/inverse_design_evolution/`. Generated figures are written to `Results/`.

The notebooks and `inverse_design_evolution.py` require NumPy and Matplotlib. The evolution script does not require PyTheus or JAX.

Run the notebooks from this directory so their relative paths resolve correctly.

`training_curves.ipynb` uses the measured `n=8` FNN history by default. Set `USE_SYNTHETIC_N8_FNN = True` in its setup cell to plot the separately identified synthetic extrapolation.
