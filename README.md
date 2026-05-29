# Surrogate Model Clean

This repository contains a clean workflow for surrogate-model-based photonic quantum-state design.

The code supports three main stages:

1. **Data generation**: generate random dense graph weights and their corresponding quantum-state amplitude vectors.
2. **Model training**: train a surrogate neural network to learn the graph-to-state map.
3. **Inverse optimisation**: use a trained surrogate model to optimise graph weights toward a target quantum state, followed by PyTheus-style verification and pruning.

The main intended use is:

```text
graph / edge weights  ->  surrogate model  ->  predicted amplitude vector
```

The trained surrogate can then be used inside an inverse-design loop to search for graph configurations that generate target states such as GHZ, W, or linear-cluster states.

---

## Repository layout

```text
surrogate_model_clean/
├── configs/
│   ├── data_config.py
│   ├── training_config.py
│   └── optimiser_config.py
├── notebooks/
│   └── workflow_data_training_optimisation_notebook.ipynb
├── src/
│   ├── data_generation.py
│   ├── data_generation_utils.py
│   ├── model_training.py
│   ├── model_training_utils.py
│   ├── models.py
│   ├── optimisation_utils.py
│   ├── optimiser.py
│   └── target_states.py
├── .gitignore
└── README.md
```

Generated data, trained models, checkpoints, logs, and optimisation outputs should normally stay local and should not be committed to GitHub.

---

## Main files

| File | Purpose |
|---|---|
| `configs/data_config.py` | Settings for data generation. |
| `configs/training_config.py` | Settings for surrogate model training. |
| `configs/optimiser_config.py` | Settings for inverse optimisation. |
| `src/data_generation.py` | Main data-generation script. |
| `src/data_generation_utils.py` | Helper functions for PyTheus catalogue construction, amplitude computation, shard merging, and graph generation. |
| `src/model_training.py` | Main training script. |
| `src/model_training_utils.py` | Training utilities: dataset loading, training steps, validation, testing, checkpointing, plotting, and logging. |
| `src/models.py` | Defines the model architectures: `FNN` and `HNN`. |
| `src/optimiser.py` | Main inverse-optimisation script. |
| `src/optimisation_utils.py` | Optimisation helpers, target-state construction, PyTheus verification, pruning, plotting, and saving. |
| `src/target_states.py` | Target-state definitions such as GHZ, W, and linear-cluster states. |
| `notebooks/workflow_data_training_optimisation_notebook.ipynb` | Jupyter notebook for running the full workflow interactively. |

---

## Installation

Create or activate a Python environment with JAX, Flax, Optax, NumPy, and plotting tools.

Example:

```bash
conda activate env_jax
```

Install the required packages. If the repository does not yet include a `requirements.txt`, install the main dependencies manually:

```bash
pip install numpy scipy matplotlib tqdm jupyter ipykernel flax optax
```

Install JAX according to your machine or cluster setup. For GPU usage, use the JAX version matching the CUDA version available on the cluster.

For example, check the official JAX installation command for your CUDA version before installing GPU-enabled JAX.

---

## Important import setup

The repository separates source files and config files:

```text
src/
configs/
```

The scripts import config modules directly, for example:

```python
from training_config import TRAINING_CONFIG
```

Therefore, when running from the terminal, add both folders to `PYTHONPATH`:

```bash
cd /path/to/surrogate_model_clean
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
```

In Jupyter notebooks, add the paths manually:

```python
from pathlib import Path
import sys

REPO_DIR = Path("/path/to/surrogate_model_clean")
SRC_DIR = REPO_DIR / "src"
CONFIG_DIR = REPO_DIR / "configs"

for p in [SRC_DIR, CONFIG_DIR]:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
```

---

# 1. Data generation

Data generation creates random dense graph-weight vectors and computes the corresponding amplitude vectors using the perfect-matching catalogue.

The generated files are saved as `.npz` shards.

Each shard contains:

```text
weights  # input graph/edge-weight vectors
amps     # output amplitude vectors
```

For an `n`-node system with local dimension 2, the default dimensions are:

```python
input_dim = 2 * n * (n - 1)
out_dim = 2 ** n
```

Example:

```python
n = 4
input_dim = 2 * 4 * (4 - 1)  # 24
out_dim = 2 ** 4              # 16
```

---

## Data-generation config

Edit:

```text
configs/data_config.py
```

Important settings:

```python
VERTICES = 4
DIMENSIONS = 2

N_SAMPLES = 1000
BATCH_SIZE = 100
SEED = 0

SAVE_TO_FILE = True
OUT_DIR = "data/smoke_test"
SHARD_SIZE = 1000

NORMED_DATA = True
```

Meaning:

| Setting | Meaning |
|---|---|
| `VERTICES` | Number of graph vertices / photons / nodes. |
| `DIMENSIONS` | Local dimension. Usually `2`. |
| `N_SAMPLES` | Number of random graph samples to generate. |
| `BATCH_SIZE` | Number of samples processed per JAX batch. |
| `SEED` | Random seed. |
| `SAVE_TO_FILE` | If `True`, save `.npz` shards to disk. |
| `OUT_DIR` | Output folder for generated shards. |
| `SHARD_SIZE` | Number of samples per saved shard. |
| `NORMED_DATA` | If `True`, save normalised amplitudes. If `False`, save raw unnormalised amplitudes. |

---

## Run data generation in Jupyter

```python
from pathlib import Path
from data_generation import generate_dataset

out_dir = generate_dataset(
    vertices=4,
    dimensions=2,
    n_samples=1000,
    batch_size=100,
    seed=59,
    save_to_file=True,
    out_dir_path="data/node4_test",
    shard_size=500,
    normed_data=False,
)

out_dir = Path(out_dir)
print(out_dir)
```

This produces files such as:

```text
data/node4_test/
├── data_00000.npz
├── data_00001.npz
├── metadata.json
└── generation.log
```

---

## Merge data shards

Training usually expects one merged dataset file, for example:

```text
dataset_merged.npz
```

Merge shards in Jupyter:

```python
from pathlib import Path
from data_generation_utils import merge_shards_to_npz

merged_path = merge_shards_to_npz(
    out_dir=Path("data/node4_test"),
    output_name="dataset_merged.npz",
)

print(merged_path)
```

The merged file contains:

```text
weights
amps
```

Check the dataset:

```python
import numpy as np

with np.load(merged_path, mmap_mode="r") as data:
    print(data.files)
    print("weights:", data["weights"].shape)
    print("amps:", data["amps"].shape)
```

---

## Run data generation from terminal

From the repository root:

```bash
cd /path/to/surrogate_model_clean
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
python src/data_generation.py
```

Then merge shards:

```bash
python - <<'PY'
from pathlib import Path
from data_generation_utils import merge_shards_to_npz

out = merge_shards_to_npz(
    out_dir=Path("data/smoke_test"),
    output_name="dataset_merged.npz",
)
print(out)
PY
```

Change `data/smoke_test` to the `OUT_DIR` used in `configs/data_config.py`.

---

# 2. Model training

Training learns the map:

```text
weights -> amps
```

The training script supports two model types:

| Model | Description | `HIDDEN_DIM` format |
|---|---|---|
| `HNN` | Polynomial/HNN-style model used for graph-to-state mapping. | Integer, e.g. `15000`. |
| `FNN` | Fully connected feed-forward neural network. | Tuple/list, e.g. `(2000, 2000, 2000)`. |

---

## Training config

Edit:

```text
configs/training_config.py
```

The main dictionary is:

```python
TRAINING_CONFIG = {
    "NODES": 10,
    "DIMENSIONS": 2,
    "DATE": "26_05_26",

    "MODEL_NAME": "HNN",
    "HIDDEN_DIM": 45000,

    "DATA_PATH": "data/node10_test/dataset_merged.npz",
    "DATA_SIZE": 5_000_000,

    "NORMALIZE_MODEL_OUTPUT": False,

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

    "RESUME_FULL_STATE": False,
    "CKPT_DIR_RESTORE": "",

    "ROOT_FOLDER": "./Models_26_05_26",
    "RUN_NAME": "HNN_10_normMAE_5M_8",
    "LOSS_NAME": "mae",
}
```

Important settings:

| Setting | Meaning |
|---|---|
| `NODES` | Number of nodes/photons used in the dataset. Must match the dataset. |
| `MODEL_NAME` | `"HNN"` or `"FNN"`. |
| `HIDDEN_DIM` | Integer for HNN; tuple/list for FNN. |
| `DATA_PATH` | Path to `dataset_merged.npz`. |
| `DATA_SIZE` | Number of samples to use. Set `None` to use the full dataset. |
| `NORMALIZE_MODEL_OUTPUT` | If `True`, normalise predictions/targets before computing some losses/metrics. Keep this consistent with the dataset. |
| `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT` | Dataset split fractions. |
| `LEARNING_RATE` | Initial learning rate. |
| `LR_AFTER_DECAY` | Final learning rate after decay. |
| `LR_DECAY_UNTIL_EPOCH` | Epoch until which learning rate is decayed. |
| `BATCH_SIZE` | Training batch size. |
| `NUM_EPOCHS` | Maximum number of epochs. |
| `PATIENCE` | Early-stopping patience. |
| `ROOT_FOLDER` | Folder where trained model outputs are saved. |
| `RUN_NAME` | Name of the training run. |
| `LOSS_NAME` | Training objective, for example `"mae"`. |

---

## Run training in Jupyter

Important: `src/model_training.py` reads `TRAINING_CONFIG` when the module is imported. Therefore, if you change `TRAINING_CONFIG` inside a notebook, reload the module before training.

```python
import importlib
import training_config

training_config.TRAINING_CONFIG.update({
    "NODES": 4,
    "DIMENSIONS": 2,

    "MODEL_NAME": "HNN",
    "HIDDEN_DIM": 400,

    "DATA_PATH": "data/node4_test/dataset_merged.npz",
    "DATA_SIZE": None,

    "NORMALIZE_MODEL_OUTPUT": False,

    "TRAIN_SPLIT": 0.8,
    "VAL_SPLIT": 0.1,
    "TEST_SPLIT": 0.1,

    "LEARNING_RATE": 1e-3,
    "LR_AFTER_DECAY": 1e-5,
    "LR_DECAY_UNTIL_EPOCH": 100,

    "BATCH_SIZE": 100,
    "NUM_EPOCHS": 20,
    "PATIENCE": 10,
    "TOLERANCE": 1e-7,
    "INIT_KEY": 159,

    "RESUME_FULL_STATE": False,
    "CKPT_DIR_RESTORE": "",

    "ROOT_FOLDER": "./Models_notebook_test",
    "RUN_NAME": "HNN_4_quick_test",
    "LOSS_NAME": "mae",
})

import model_training
importlib.reload(model_training)

run_dir = model_training.train_surrogate_model()
print(run_dir)
```

The trained parameter file is normally saved as:

```text
<ROOT_FOLDER>/<RUN_NAME>/params.msgpack
```

For example:

```python
from pathlib import Path
params_path = Path(run_dir) / "params.msgpack"
print(params_path, params_path.exists())
```

---

## Run training from terminal

1. Edit `configs/training_config.py`.
2. Make sure `DATA_PATH` points to the merged dataset.
3. Run:

```bash
cd /path/to/surrogate_model_clean
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
python src/model_training.py
```

Training outputs include files such as:

```text
Models_*/<RUN_NAME>/
├── params.msgpack
├── model_info.json
├── test_metrics.npz
├── log.txt
├── plots/
│   ├── training_curves.png
│   └── test_fidelity_curve.png
└── checkpoints/
```

The exact folder names depend on `ROOT_FOLDER` and `RUN_NAME`.

---

# 3. Inverse optimisation

The optimiser uses a trained surrogate model to optimise input graph weights toward a target state.

Conceptually:

```text
start graph weights
      ↓
surrogate prediction
      ↓
fidelity with target state
      ↓
gradient-based update
      ↓
pruning + PyTheus verification
```

---

## Optimiser config

Edit:

```text
configs/optimiser_config.py
```

Important settings:

| Setting | Meaning |
|---|---|
| `n` | Number of nodes/photons. Must match the trained model. |
| `dimensions` | Local dimension, usually `2`. |
| `target_name` | Target state, e.g. `"GHZ"`, `"W"`, or `"LINEAR_CLUSTER"`. |
| `model_type` | `"HNN"` or `"FNN"`. Must match trained model. |
| `generate_data` | If `True`, generate fresh starting samples. If `False`, load starting samples from `conditioned_data_path`. |
| `conditioned_data_path` | Path to initial samples when `generate_data=False`. |
| `data_samples` | Number of starting samples when generating fresh data. |
| `low_fidelity_threshold` | Initial-sample filtering threshold. |
| `architecture` | HNN integer hidden dimension or FNN tuple/list. Must match trained model. |
| `model_path` | Path to trained `params.msgpack`. |
| `normalize_model_output` | Must match the model/data training choice. |
| `input_dim` | Usually `2 * n * (n - 1)`. |
| `out_dim` | Usually `2 ** n`. |
| `lambda_l1` | L1 sparsity penalty. Objective is approximately `1 - fidelity + lambda_l1 * L1`. |
| `num_steps` | Maximum optimiser steps. |
| `learning_rate` | Initial optimiser learning rate. |
| `min_learning_rate` | Minimum learning rate. |
| `clip_min`, `clip_max` | Bounds for graph weights. Usually `[-1, 1]`. |
| `verify_every` | How often to run verification. |
| `store_step_vectors` | If `True`, stores detailed step vectors. Can create very large JSON files. Use `False` for large runs. |
| `prune_thresholds` | Edge-weight thresholds tested during pruning. |
| `results_root` | Output root folder for optimisation results. |
| `folder_name` | Name of this optimisation run. |

---

## Run optimisation in Jupyter

Unlike training, the optimiser is already easy to call from a notebook because `run_optimisation()` accepts a config dictionary.

```python
from optimiser import run_optimisation

CFG_OPT = {
    "description": "notebook inverse optimisation test",

    "n": 4,
    "dimensions": 2,
    "target_name": "GHZ",
    "model_type": "HNN",

    "generate_data": True,
    "conditioned_data_path": "",
    "max_initial_samples": None,

    "data_samples": 5,
    "data_batch_size": 5,
    "data_seed": 4,
    "low_fidelity_threshold": 0.999,
    "save_generated_data": False,
    "data_out_dir": "data/optimiser_generated",
    "normed_data": True,
    "generation_gpu_batch_size": 5,
    "data_shard_size": 5,

    "architecture": 400,
    "model_path": "Models_notebook_test/HNN_4_quick_test/params.msgpack",
    "normalize_model_output": False,

    "input_dim": 2 * 4 * (4 - 1),
    "out_dim": 2 ** 4,

    "lambda_l1": 1e-3,

    "seed": 46,
    "num_steps": 20,
    "early_stop_nn_fid": 0.9999,

    "learning_rate": 1e-2,
    "min_learning_rate": 1e-6,
    "lr_decay_steps": 20,
    "lr_exponent": 1.0,

    "clip_min": -1.0,
    "clip_max": 1.0,

    "print_every": 1,
    "verify_every": 1,

    "store_step_vectors": False,

    "prune_fid_tolerance": 1e-4,
    "prune_thresholds": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],

    "results_root": "optimiser_notebook_results",
    "folder_name": "GHZ_4_quick_test",

    "date": "notebook",
    "k_value": 0,
    "noise": 0,
    "CPU_GPU": "GPU",
}

result_dir = run_optimisation(CFG_OPT)
print(result_dir)
```

---

## Run optimisation from terminal

1. Edit `configs/optimiser_config.py`.
2. Make sure the following match the trained model:

```python
"n"
"model_type"
"architecture"
"model_path"
"normalize_model_output"
"input_dim"
"out_dim"
```

3. Run:

```bash
cd /path/to/surrogate_model_clean
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
python src/optimiser.py
```

Optimisation outputs usually include:

```text
<results_root>/<folder_name>/
├── optimisation_summary.json
├── best_graph_solution.json
├── zero_state_samples.json
├── optimisation.log
├── edge_fidelity_plot.png
├── time_plots.png
└── sample_*/
    ├── sample_info_file.json
    ├── loss_fid_plot.png
    └── gradient_plot.png
```

Exact filenames can vary depending on the utilities used inside `optimisation_utils.py`.

---

# 4. Full workflow in Jupyter

The recommended notebook is:

```text
notebooks/workflow_data_training_optimisation_notebook.ipynb
```

The notebook workflow is:

```python
# 1. Add paths
REPO_DIR = Path("/path/to/surrogate_model_clean")
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "configs"))

# 2. Generate data
from data_generation import generate_dataset
out_dir = generate_dataset(...)

# 3. Merge shards
from data_generation_utils import merge_shards_to_npz
merged_path = merge_shards_to_npz(...)

# 4. Train model
import training_config
training_config.TRAINING_CONFIG.update({...})

import model_training
import importlib
importlib.reload(model_training)
run_dir = model_training.train_surrogate_model()

# 5. Run optimisation
from optimiser import run_optimisation
result_dir = run_optimisation(CFG_OPT)
```

This notebook is useful for:

- testing the full workflow with small sample sizes,
- debugging configs,
- verifying data shapes,
- checking that training produces `params.msgpack`,
- checking that optimisation can load the trained model.

For large production runs, use terminal or Slurm jobs instead of Jupyter.

---

# 5. Full workflow from terminal

Use this after editing the config files.

```bash
cd /path/to/surrogate_model_clean
conda activate env_jax
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
```

## Step 1: edit data config

Edit:

```text
configs/data_config.py
```

Then run:

```bash
python src/data_generation.py
```

Merge shards:

```bash
python - <<'PY'
from pathlib import Path
from data_generation_utils import merge_shards_to_npz

out = merge_shards_to_npz(
    out_dir=Path("data/smoke_test"),
    output_name="dataset_merged.npz",
)
print("Merged dataset:", out)
PY
```

## Step 2: edit training config

Edit:

```text
configs/training_config.py
```

Set:

```python
"DATA_PATH": "data/smoke_test/dataset_merged.npz"
```

Then run:

```bash
python src/model_training.py
```

After training, locate:

```text
<ROOT_FOLDER>/<RUN_NAME>/params.msgpack
```

## Step 3: edit optimiser config

Edit:

```text
configs/optimiser_config.py
```

Set:

```python
"model_path": "<ROOT_FOLDER>/<RUN_NAME>/params.msgpack"
```

Then run:

```bash
python src/optimiser.py
```

---

# 6. Quick sanity checks

## Check dataset shape

```python
import numpy as np

DATA_PATH = "data/node4_test/dataset_merged.npz"

with np.load(DATA_PATH, mmap_mode="r") as data:
    print(data.files)
    print("weights:", data["weights"].shape)
    print("amps:", data["amps"].shape)
```

## Check zero-output MAE baseline

This checks the MAE obtained if the model predicted an all-zero output vector:

```python
import numpy as np

DATA_PATH = "data/node10_test/dataset_merged.npz"

with np.load(DATA_PATH, mmap_mode="r") as data:
    Y = data["amps"]
    mae_zero = np.mean(np.abs(Y))

print("Zero-output MAE baseline:", mae_zero)
```

If the trained model loss is close to this number, the model may be predicting values close to zero.

## Check train/validation/test zero-output MAE

```python
import numpy as np

DATA_PATH = "data/node10_test/dataset_merged.npz"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

with np.load(DATA_PATH, mmap_mode="r") as data:
    Y = np.asarray(data["amps"], dtype=np.float32)

n = Y.shape[0]
n_train = int(n * TRAIN_SPLIT)
n_val = int(n * VAL_SPLIT)

Y_train = Y[:n_train]
Y_val = Y[n_train:n_train + n_val]
Y_test = Y[n_train + n_val:]

print("Train zero MAE:", np.mean(np.abs(Y_train)))
print("Val zero MAE:", np.mean(np.abs(Y_val)))
print("Test zero MAE:", np.mean(np.abs(Y_test)))
```

---

# 7. Notes for large cluster runs

For large datasets and large models:

- Use the cluster/GPU, not a local notebook.
- Keep `data/`, `Models_*`, `optimiser_*`, logs, and checkpoints out of Git.
- Use small smoke tests first, for example `N_SAMPLES=1000`, `NUM_EPOCHS=20`, `num_steps=20`.
- Increase to full-scale values only after the full pipeline runs correctly.
- Set `store_step_vectors=False` in the optimiser unless you explicitly need all gradient/update vectors.
- Use absolute paths in config files when running through Slurm.
- Keep `NORMALIZE_MODEL_OUTPUT`, `NORMED_DATA`, and `normalize_model_output` consistent between data generation, training, and optimisation.

---

# 8. Common mistakes

## Import error: `ModuleNotFoundError: No module named 'training_config'`

Add `src/` and `configs/` to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
```

## Training cannot find the dataset

Check `DATA_PATH` in:

```text
configs/training_config.py
```

Make sure the merged file exists:

```bash
ls -lh data/smoke_test/dataset_merged.npz
```

## Optimiser cannot load the model

Check `model_path` in:

```text
configs/optimiser_config.py
```

It should point to:

```text
<training output folder>/params.msgpack
```

## HNN/FNN architecture mismatch

For HNN:

```python
"MODEL_NAME": "HNN"
"HIDDEN_DIM": 15000
```

For FNN:

```python
"MODEL_NAME": "FNN"
"HIDDEN_DIM": (2000, 2000, 2000)
```

The optimiser must use the same architecture format:

```python
"model_type": "HNN"
"architecture": 15000
```

or:

```python
"model_type": "FNN"
"architecture": (2000, 2000, 2000)
```

## Normalisation mismatch

Keep these settings consistent:

```python
# data_config.py
NORMED_DATA

# training_config.py
NORMALIZE_MODEL_OUTPUT

# optimiser_config.py
normalize_model_output
```

Do not mix normalised and unnormalised assumptions without checking the loss and fidelity definitions.

---

# 9. Minimal command summary

```bash
cd /path/to/surrogate_model_clean
conda activate env_jax
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"

# Generate data after editing configs/data_config.py
python src/data_generation.py

# Merge generated shards
python - <<'PY'
from pathlib import Path
from data_generation_utils import merge_shards_to_npz
print(merge_shards_to_npz(Path("data/smoke_test"), "dataset_merged.npz"))
PY

# Train model after editing configs/training_config.py
python src/model_training.py

# Run inverse optimisation after editing configs/optimiser_config.py
python src/optimiser.py
```
