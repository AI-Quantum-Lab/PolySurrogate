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

All generated datasets, models, and optimisation runs are written under `results/`. This directory is ignored by Git and can be deleted at any time.

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

Its settings are intentionally small. The notebook demonstrates the complete software workflow and does not reproduce the full paper-scale training runs.

## Running the three stages directly

The scripts have small, mutually compatible defaults and can be run in order:

```bash
python data_generate.py
python ml_model.py
python inverse_design.py
```

The first command creates `results/data_generation/n4/n4_0/dataset_merged.npz`. The second trains a model from that dataset and writes `results/model_training/PNN/n4/n4_0/best_params.msgpack`. The third uses that model for inverse design.

Each script keeps its main settings near the top of the file. For larger experiments, update the dataset size, network architecture, epoch count, optimisation steps, and paths there.

### Stage 1: data generation

`data_generate.py` samples graph-edge weights and computes the corresponding exact amplitudes with PyTheus. It writes dataset shards and a merged NumPy archive under `results/data_generation/`.

### Stage 2: model training

`ml_model.py` trains either:

- a polynomial neural network (`PNN`), or
- a feedforward neural network (`FNN`).

It writes training history, validation checkpoints, test metrics, and `best_params.msgpack` under `results/model_training/`.

### Stage 3: inverse design

`inverse_design.py` optimises graph weights through the trained surrogate, verifies the resulting state with PyTheus, and prunes small graph weights. Named GHZ, W, and linear-cluster targets are included. A custom amplitude vector can be passed as `target_state` through the configuration dictionary.

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

The small `sample_workflow.ipynb` does not depend on the Zenodo archive.

## Citation

A paper citation and Zenodo DOI will be added when the associated records are published. Until then, cite this repository:

```text
AI-Quantum-Lab. PolySurrogate. https://github.com/AI-Quantum-Lab/PolySurrogate
```

## License

PolySurrogate is released under the MIT License. See `LICENSE`.
