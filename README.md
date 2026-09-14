# PolySurrogate

PolySurrogate uses JAX and Flax to learn the relationship between weighted photonic graphs and their quantum states. The trained surrogate then supports fast, gradient-based inverse design.

```text
graph weights → PNN/FNN surrogate → quantum amplitudes → inverse design
```

## Repository layout

```text
PolySurrogate/
├── data_generate.py              Generate graph and amplitude datasets
├── ml_model.py                   Train PNN or FNN surrogate models
├── inverse_design.py             Search for graphs matching a target state
├── utils.py                      Shared model and PyTheus helpers
├── sample_workflow.ipynb         Small end-to-end example
├── plotting_code/                Paper plotting notebooks, data, and reference figures
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.11 or newer is required. Development and verification used Python 3.12.

```bash
git clone https://github.com/AI-Quantum-Lab/PolySurrogate.git
cd PolySurrogate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The PyPI distribution is named `pytheusQ`, while the Python import is `pytheus`. The correct distribution is already listed in `requirements.txt`.

The default JAX installation uses the CPU. Install the appropriate JAX CUDA package separately when GPU acceleration is required.

Verify the installation:

```bash
python -c "import jax, flax, optax, pytheus; print(jax.default_backend(), jax.devices())"
```

## Quick start

Launch the example from the repository root:

```bash
jupyter notebook sample_workflow.ipynb
```

Run all cells from top to bottom. The notebook:

1. generates 5,000 graph and amplitude pairs;
2. trains a small four-node PNN for 20 epochs;
3. performs inverse design for one generated target state.

Its settings are intentionally small. The notebook demonstrates the complete software workflow, requires no external paper data, and does not reproduce the full paper-scale training runs.

## Running the three stages directly

The scripts have small, mutually compatible defaults and can be run in order:

```bash
python data_generate.py
python ml_model.py
python inverse_design.py
```

| Script | Purpose | Output |
|---|---|---|
| `data_generate.py` | Sample graph weights and compute exact amplitudes with PyTheus | Dataset shards and a merged archive in `results/data_generation/` |
| `ml_model.py` | Train a polynomial (`PNN`) or feedforward (`FNN`) surrogate | Training history, checkpoints, and metrics in `results/model_training/` |
| `inverse_design.py` | Optimise graph weights for a target state and verify the result with PyTheus | Optimisation results in `results/inverse_design/` |

Inverse design includes GHZ, W, and linear-cluster targets. A custom amplitude vector can be passed as `target_state` through the configuration dictionary.

By default, starting graphs are generated in memory. Set `generate_data=False` and provide `conditioned_data_path` to use a downloaded starting pool.

## Paper figures

`plotting_code/` contains the processed inputs, notebooks, combined inverse-design evolution script, and reference figures used for the paper.

| File | Output |
|---|---|
| `training_curves.ipynb` | PNN and FNN training curves |
| `testing_hist.ipynb` | Test-set MAE histograms |
| `jacobian.ipynb` | Surrogate and PyTheus Jacobian heatmaps |
| `initial_to_final_fid.ipynb` | Initial and final fidelity distributions |
| `runtime.ipynb` | Runtime comparison |
| `inverse_design_evolution.py` | Loss trajectory and graph evolution |

Run the notebooks with:

```bash
cd plotting_code
jupyter notebook
```

Run the evolution figure from the repository root:

```bash
python plotting_code/inverse_design_evolution.py
```

Processed inputs are stored in `plotting_code/Data/`. Reference and regenerated figures are stored in `plotting_code/Results/`. The evolution script needs only NumPy and Matplotlib.

## Paper data and pretrained models

The full paper data and pretrained models will be published on Zenodo. The DOI and download instructions will be added here when the archive is ready.

## Citation

A paper citation will be added when it is available. Until then, cite this repository:

```text
AI-Quantum-Lab. PolySurrogate. https://github.com/AI-Quantum-Lab/PolySurrogate
```

## License

PolySurrogate is released under the MIT License. See `LICENSE`.
