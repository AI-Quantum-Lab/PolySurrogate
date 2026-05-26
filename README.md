# Surrogate Model for Photonic Quantum State Design

This repository contains research code for generating datasets, training surrogate neural-network models, and running inverse-optimisation workflows for photonic quantum-state design.

The main goal is to learn a surrogate model that maps graph/edge-weight representations of photonic setups to output quantum-state amplitude vectors. The trained surrogate can then be used for fast evaluation and inverse design of photonic quantum states.

---

## Repository Structure

```text
surrogate_model_clean/
├── README.MD
├── training_config.py
├── data_config.py
├── optimiser_config.py
├── data_generation.py
├── data_generation_utils.py
├── model_training.py
├── model_training_utils.py
├── optimiser.py
├── optimisation_utils.py
├── models.py
├── target_states.py
└── notebooks/
    └── optimiser_quick_test.ipynb
```

Generated data, trained models, checkpoints, logs, and optimisation results are intentionally excluded from GitHub.

---

## Main Files

| File | Purpose |
|---|---|
| `training_config.py` | Configuration for training surrogate models |
| `data_config.py` | Configuration for dataset generation |
| `optimiser_config.py` | Configuration for inverse optimisation |
| `data_generation.py` | Main script for generating datasets |
| `data_generation_utils.py` | Helper functions for dataset generation |
| `model_training.py` | Main script for training FNN/HNN surrogate models |
| `model_training_utils.py` | Training, validation, testing, plotting, and checkpoint utilities |
| `optimiser.py` | Main inverse-optimisation script |
| `optimisation_utils.py` | Helper functions for optimisation, PyTheus verification, pruning, and plotting |
| `models.py` | Defines the surrogate model architectures |
| `target_states.py` | Defines target quantum states such as GHZ, W, and linear cluster states |
| `notebooks/optimiser_quick_test.ipynb` | Notebook for quick optimiser testing and debugging |

---

## Installation

Create or activate a Python environment.

Example:

```bash
conda activate env_jax
```

Install required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, the main dependencies are:

```bash
pip install jax jaxlib flax optax numpy matplotlib scipy tqdm ipykernel jupyter
```

For GPU usage, install the JAX version compatible with the CUDA version on your machine or cluster.

---

## Data Format

Datasets are expected as `.npz` files.

The data loader accepts either:

```text
weights
amps
```

or:

```text
X
Y
```

where:

- `weights` / `X` are graph or edge-weight input features
- `amps` / `Y` are target quantum-state amplitude vectors

For an `n`-node system, the default dimensions are:

```python
INPUT_DIMS = 2 * NODES * (NODES - 1)
OUTPUT_DIMS = 2 ** NODES
```

Large datasets should be stored locally and should not be committed to GitHub.

---

## 1. Generate Data

Edit the dataset-generation settings in:

```text
data_config.py
```

Then run:

```bash
python data_generation.py
```

Typical settings include:

```python
NODES = 10
DIMENSIONS = 2
N_SAMPLES = 5_000_000
BATCH_SIZE = 5000
NORMED_DATA = True
```

The generated dataset is usually saved under a local `data/` folder.

Example:

```text
data/node10_test/dataset_merged.npz
```

The `data/` folder is ignored by Git.

---

## 2. Train a Surrogate Model

Edit the training settings in:

```text
training_config.py
```

Example configuration:

```python
TRAINING_CONFIG = {
    "NODES": 10,
    "DIMENSIONS": 2,
    "MODEL_NAME": "HNN",
    "HIDDEN_DIM": 20_000,

    "DATA_PATH": "data/node10_test/dataset_merged.npz",
    "DATA_SIZE": 5_000_000,

    "NORMALIZE_MODEL_OUTPUT": True,

    "TRAIN_SPLIT": 0.8,
    "VAL_SPLIT": 0.1,
    "TEST_SPLIT": 0.1,

    "LEARNING_RATE": 1e-3,
    "LR_AFTER_DECAY": 1e-5,
    "LR_DECAY_UNTIL_EPOCH": 5000,

    "BATCH_SIZE": 5000,
    "NUM_EPOCHS": 20000,
    "PATIENCE": 2000,
    "TOLERANCE": 1e-7,
    "INIT_KEY": 159,

    "ROOT_FOLDER": "./Models_26_05_26",
    "RUN_NAME": "HNN_10_normMAE_5M",
    "LOSS_NAME": "normalized_mae",
}
```

Then run:

```bash
python model_training.py
```

The script will:

1. load the dataset
2. split it into train/validation/test sets
3. build the selected model
4. train the model
5. save checkpoints
6. save the best model parameters
7. evaluate on the test set
8. save plots, logs, and metrics

Outputs are saved under:

```text
Models_<DATE>/run_<RUN_NAME>/
```

Example:

```text
Models_26_05_26/run_HNN_10_normMAE_5M/
```

---

## Normalised vs Unnormalised Training

The training code supports both raw-amplitude training and normalised-state training.

### Raw-amplitude training

Use:

```python
"NORMALIZE_MODEL_OUTPUT": False
```

This trains directly on the raw amplitude vectors:

```python
loss = mean(abs(y_pred - y_target))
```

The model must learn both:

1. the quantum-state direction
2. the raw amplitude scale

This can be difficult for larger systems because the amplitude scale may vary strongly.

---

### Normalised-state training

Use:

```python
"NORMALIZE_MODEL_OUTPUT": True
```

This normalises both prediction and target before computing loss, validation metrics, and test metrics:

```python
loss = mean(abs(normalize(y_pred) - normalize(y_target)))
```

This focuses training on the quantum-state direction rather than raw amplitude scale.

For larger systems, such as `n = 10` or above, this is often a useful first training mode.

---

## 3. Evaluate Training Results

After training, check the run folder:

```text
Models_<DATE>/run_<RUN_NAME>/
```

Important files:

| File | Purpose |
|---|---|
| `run_log.txt` | Training log |
| `model_info.json` | Training configuration and recorded metrics |
| `params.msgpack` | Final saved model parameters |
| `test_metrics.npz` | Test loss, fidelity, MSE, and MAE |
| `plots/training_curves.png` | Training and validation loss curves |
| `plots/test_fidelity_curve.png` | Per-sample test fidelity curve |

For quantum-state prediction, fidelity is usually the most important metric.

For output dimension:

```python
OUTPUT_DIMS = 2 ** NODES
```

a random-state fidelity baseline is approximately:

```python
1 / OUTPUT_DIMS
```

For `NODES = 10`:

```text
1 / 1024 ≈ 0.001
```

A useful surrogate should achieve fidelity clearly above this baseline.

---

## 4. Run Inverse Optimisation

Edit the optimisation settings in:

```text
optimiser_config.py
```

Example configuration:

```python
OPTIMISER_CONFIG = {
    "n": 8,
    "dimensions": 2,
    "target_name": "W",
    "model_type": "HNN",

    "generate_data": False,
    "conditioned_data_path": "path/to/conditioned_dataset.npz",
    "max_initial_samples": None,

    "architecture": 15000,
    "model_path": "path/to/params.msgpack",
    "normalize_model_output": True,

    "input_dim": 2 * 8 * (8 - 1),
    "out_dim": 2 ** 8,

    "lambda_l1": 1e-3,

    "seed": 46,
    "num_steps": 10000,
    "early_stop_nn_fid": 0.9999,

    "learning_rate": 1e-2,
    "min_learning_rate": 1e-6,
    "lr_decay_steps": 10000,

    "clip_min": -1.0,
    "clip_max": 1.0,

    "print_every": 1,
    "verify_every": 1,

    "store_step_vectors": True,

    "prune_fid_tolerance": 1e-4,
    "prune_thresholds": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],

    "results_root": "optimisation_results",
    "folder_name": "W_100_sample_clean",
}
```

Then run:

```bash
python optimiser.py
```

The optimiser will:

1. load or generate initial graph samples
2. load the trained surrogate model
3. optimise graph weights toward the target state
4. compute neural-network fidelity
5. verify selected states with PyTheus
6. prune weak graph weights
7. save per-sample optimisation results
8. save summary plots and JSON files

---

## Optimisation Objective

The inverse optimiser uses the objective:

```python
loss = 1 - fidelity + lambda_l1 * sum(abs(x))
```

where:

- `fidelity` measures overlap with the target state
- `x` is the graph-weight vector
- `lambda_l1` controls sparsity pressure

Increasing `lambda_l1` encourages sparser graph solutions, but may reduce fidelity.

---

## Optimisation Outputs

The optimiser saves results under:

```text
<results_root>/<folder_name>/
```

Important files:

| File | Purpose |
|---|---|
| `cfg.json` | Saved optimisation configuration |
| `log.txt` | Full optimisation log |
| `optimisation_summary.json` | Summary of all samples |
| `best_graph_solution.json` | Best sparse graph found |
| `zero_state_samples.json` | Records of zero-vector failures, if any |
| `edge_vs_fidelity.png` | Sparsity vs fidelity plot |
| `time_per_sample.png` | Runtime per sample |
| `cumulative_time.png` | Cumulative runtime |
| `sample_<id>/sample_info_file.json` | Detailed per-sample information |
| `sample_<id>/loss_fid_curve.png` | Loss and fidelity curve |
| `sample_<id>/gradient_norm.png` | Gradient norm curve |

---

## Important Runtime Notes

### Step-vector storage

In `optimiser_config.py`:

```python
"store_step_vectors": True
```

stores gradients and update vectors at every optimisation step.

This is useful for debugging, but can create very large JSON files.

For large runs, use:

```python
"store_step_vectors": False
```

---

### Verification frequency

```python
"verify_every": 1
"print_every": 1
```

prints and verifies every step.

For long runs, increase these values:

```python
"verify_every": 100
"print_every": 100
```

This reduces logging overhead.

---

## Target States

Target states are defined in:

```text
target_states.py
```

Typical supported targets include:

```text
GHZ
W
LINEAR_CLUSTER
SINGLE
```

Example:

```python
TARGET_NAME = "W"
```

The target state should match the number of nodes/photons:

```python
NPHOTONS = 8
```

---

## Models

Model definitions are in:

```text
models.py
```

The repository supports:

| Model | Description |
|---|---|
| `FNN` | Standard feed-forward neural network |
| `HNN` | Polynomial / high-order neural network motivated by perfect-matching structure |

The HNN is motivated by the fact that photonic amplitudes can be expressed as sums over perfect matchings, where each amplitude contains products of edge weights.

---

## Recommended Workflow

### Step 1 — Generate data

```bash
python data_generation.py
```

### Step 2 — Train surrogate model

```bash
python model_training.py
```

### Step 3 — Inspect training results

Check:

```text
Models_<DATE>/run_<RUN_NAME>/
```

Look at:

```text
run_log.txt
model_info.json
plots/training_curves.png
plots/test_fidelity_curve.png
```

### Step 4 — Run inverse optimisation

```bash
python optimiser.py
```

### Step 5 — Inspect optimisation results

Check:

```text
<results_root>/<folder_name>/
```

Look at:

```text
optimisation_summary.json
best_graph_solution.json
edge_vs_fidelity.png
time_per_sample.png
```

---

## GitHub / Version Control Notes

The repository is designed to track source code and configuration files only.

Do not commit:

```text
data/
Models_*/
*.npz
*.npy
*.msgpack
*.log
optimiser_notebook_results/
results/
outputs/
__pycache__/
```

These files are ignored through `.gitignore`.

Before committing, check:

```bash
git status
```

and verify that no large data/model files are staged.

---

## Example Git Commands

Check status:

```bash
git status
```

Add changed code files:

```bash
git add README.MD
git add training_config.py data_config.py optimiser_config.py
git add data_generation.py data_generation_utils.py
git add model_training.py model_training_utils.py
git add optimiser.py optimisation_utils.py
git add models.py target_states.py
```

Commit:

```bash
git commit -m "Update surrogate model code"
```

Push:

```bash
git push
```

---

## Notes

This repository is research code for surrogate modelling and inverse design of photonic quantum-state experiments.

The project intentionally uses a flat file structure so imports remain simple, for example:

```python
from models import create_model
from model_training_utils import create_train_step
from optimisation_utils import build_optimizer
from target_states import get_target_state
```

A more complex package structure can be introduced later if the project grows substantially.