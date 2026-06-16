# Surrogate Model Clean

A JAX/Flax pipeline for photonic quantum-state design via a learned surrogate model.

```text
graph / edge weights  ->  surrogate model  ->  predicted amplitude vector
```

The pipeline has three stages:

1. **Data generation** — generate random dense graph weights and the corresponding quantum-state amplitude vectors using a PyTheus perfect-matching catalogue.
2. **Model training** — train a surrogate neural network (`FNN` or `HNN`) to learn the graph-weights -> amplitudes map.
3. **Inverse design optimisation** — use the trained surrogate to optimise graph weights toward a target state (GHZ, W, linear-cluster, ...), verified against PyTheus and pruned to a sparse graph.

---

## Installation on Windows / Linux

The required Python packages, based on what the code actually imports, are:

```text
jax
flax
optax
numpy
matplotlib
pytheusQ   (PyPI distribution name; imports as "pytheus")
```

`jupyter` and `ipykernel` are only needed if you plan to use the notebook.

**Install `pytheusQ`, not `pytheus`.** PyPI's `pytheus` package is an unrelated Prometheus metrics client. The quantum-optics inverse-design library this repository imports as `from pytheus import theseus as th` is published under the distribution name `pytheusQ`, which installs the `pytheus` module (including `pytheus/theseus.py`).

### Linux / macOS

```bash
cd /path/to/surrogate_model_clean
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib jax flax optax pytheusQ
pip install jupyter ipykernel   # only if you will use the notebook
```

Add `src/` and `configs/` to `PYTHONPATH` for every terminal session you run scripts from:

```bash
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
```

### Windows (PowerShell)

```powershell
cd C:\path\to\surrogate_model_clean
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy matplotlib jax flax optax pytheusQ
pip install jupyter ipykernel   # only if you will use the notebook
```

Add `src\` and `configs\` to `PYTHONPATH` for the current PowerShell session:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\configs;$env:PYTHONPATH"
```

For GPU use, install the JAX build that matches your CUDA version by following JAX's own installation instructions for your platform. The code runs on CPU as well; JAX will fall back to `CpuDevice` if no usable GPU is found.

---

## Table of contents

- [Repository structure](#repository-structure)
- [Minimal smoke test](#minimal-smoke-test)
- [Data generation](#data-generation)
- [Model training](#model-training)
- [Inverse design optimisation](#inverse-design-optimisation)
- [Notebook workflow](#notebook-workflow)
- [Expected outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [Citation / License](#citation--license)

---

## Repository structure

```text
surrogate_model_clean/
├── configs/
│   ├── __init__.py
│   ├── data_config.py
│   ├── training_config.py
│   └── optimiser_config.py
├── notebooks/
│   └── workflow_data_training_optimisation_notebook.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── data_generation_utils.py
│   ├── model_training.py
│   ├── model_training_utils.py
│   ├── models.py
│   ├── optimiser.py
│   ├── optimisation_utils.py
│   └── target_states.py
├── .gitignore
└── README.md
```

| File | Purpose |
|---|---|
| `configs/data_config.py` | Settings for data generation. |
| `configs/training_config.py` | Settings for surrogate model training. |
| `configs/optimiser_config.py` | Settings for inverse-design optimisation. |
| `src/data_generation.py` | Main data-generation script (`generate_dataset()`). |
| `src/data_generation_utils.py` | PyTheus catalogue construction, amplitude computation, shard merging. |
| `src/model_training.py` | Main training script (`train_surrogate_model()`). |
| `src/model_training_utils.py` | Dataset loading/splitting, training/eval/test steps, checkpointing, plotting. |
| `src/models.py` | Model architectures: `FNN`, `HNN`, and `create_model()`. |
| `src/optimiser.py` | Main inverse-design script (`run_optimisation()`). |
| `src/optimisation_utils.py` | Model loading, optimisation step, PyTheus verification, pruning, plotting. |
| `src/target_states.py` | Target-state definitions: GHZ, W, linear-cluster, single, zero. |
| `notebooks/workflow_data_training_optimisation_notebook.ipynb` | Interactive walkthrough of the full workflow. |

Generated data, trained models, checkpoints, logs, and optimisation outputs are gitignored and should stay local.

None of the scripts accept command-line arguments. Every script reads its settings from the matching file in `configs/` at import time. To change a setting, edit the config file (or, in a notebook, edit the imported config dict before reloading the module).

---

## Minimal smoke test

`configs/data_config.py` already ships with small smoke-test values (`VERTICES=4`, `N_SAMPLES=1000`, `OUT_DIR='data/smoke_test'`), so data generation works as committed.

`configs/training_config.py` and `configs/optimiser_config.py` ship with production-scale values and absolute paths from the original author's machine (e.g. `NODES=10`, `HIDDEN_DIM=45000`, and `DATA_PATH`/`model_path`/`conditioned_data_path` under `/home/...`). **These two config files must be edited manually before a smoke test will run.**

1. Run data generation with the existing defaults:

```bash
python src/data_generation.py
```

2. Merge the generated shards into one file:

```bash
python -c "from pathlib import Path; from data_generation_utils import merge_shards_to_npz; print(merge_shards_to_npz(Path('data/smoke_test'), 'dataset_merged.npz'))"
```

3. Edit `configs/training_config.py` and set `TRAINING_CONFIG` to small test values, for example:

```python
"NODES": 4,
"MODEL_NAME": "HNN",
"HIDDEN_DIM": 400,
"DATA_PATH": "data/smoke_test/dataset_merged.npz",
"DATA_SIZE": None,
"NUM_EPOCHS": 20,
"PATIENCE": 10,
"BATCH_SIZE": 100,
"LR_DECAY_UNTIL_EPOCH": 20,
"RESUME_FULL_STATE": False,
"ROOT_FOLDER": "./Models_smoke_test",
"RUN_NAME": "HNN_4_smoke_test",
```

Then run:

```bash
python src/model_training.py
```

4. Edit `configs/optimiser_config.py` and set `NPHOTONS = 4`, `MODEL_TYPE = "HNN"`, and inside `OPTIMISER_CONFIG` set `"architecture": 400`, `"generate_data": True` (so no `conditioned_data_path` is needed), and `"model_path"` to the `params.msgpack` produced in step 3 (`./Models_smoke_test/run_HNN_4_smoke_test/params.msgpack`). Then run:

```bash
python src/optimiser.py
```

This mirrors the small-scale run already exercised in the notebook (see [Notebook workflow](#notebook-workflow)) and is the fastest way to confirm the environment is set up correctly.

---

## Data generation

Edit `configs/data_config.py`:

| Setting | Meaning |
|---|---|
| `VERTICES` | Number of graph vertices / photons / nodes. |
| `DIMENSIONS` | Local dimension. Usually `2`. |
| `N_SAMPLES` | Number of random graph samples to generate. |
| `BATCH_SIZE` | Samples processed per JAX batch. |
| `SEED` | Random seed. |
| `SAVE_TO_FILE` | If `True`, save `.npz` shards to disk. |
| `OUT_DIR` | Output folder for generated shards. |
| `SHARD_SIZE` | Samples per saved shard. |
| `NORMED_DATA` | If `True`, save normalised amplitudes; if `False`, raw amplitudes. |

Run from the repository root, with `PYTHONPATH` set as above:

```bash
python src/data_generation.py
```

This calls `generate_dataset()` with the values from `data_config.py` and writes shards named `data_00000.npz`, `data_00001.npz`, ... plus `metadata.json` and a `logs/` folder inside `OUT_DIR`.

To call `generate_dataset()` directly (e.g. from a notebook or REPL) with different arguments instead of editing the config file:

```python
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
```

Training expects one merged file. Merge the shards with:

```bash
python -c "from pathlib import Path; from data_generation_utils import merge_shards_to_npz; print(merge_shards_to_npz(Path('data/node4_test'), 'dataset_merged.npz'))"
```

This produces `dataset_merged.npz` containing the keys `weights` (input graph/edge weights) and `amps` (output amplitude vectors).

For an `n`-node, local-dimension-2 system:

```python
input_dim = 2 * n * (n - 1)
out_dim = 2 ** n
```

---

## Model training

Edit `configs/training_config.py` (`TRAINING_CONFIG` dict):

| Setting | Meaning |
|---|---|
| `NODES` | Number of nodes/photons in the dataset. Must match the dataset. |
| `MODEL_NAME` | `"HNN"` or `"FNN"`. |
| `HIDDEN_DIM` | Integer for `HNN`; tuple/list for `FNN`, e.g. `(2000, 2000, 2000)`. |
| `DATA_PATH` | Path to the merged `.npz` dataset. **Must be edited from the shipped absolute path.** |
| `DATA_SIZE` | Number of samples to use; `None` uses the full dataset. |
| `NORMALIZE_MODEL_OUTPUT` | Must match `NORMED_DATA` used during data generation. |
| `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT` | Dataset split fractions (must sum to 1.0). |
| `LEARNING_RATE`, `LR_AFTER_DECAY`, `LR_DECAY_UNTIL_EPOCH` | Cosine learning-rate schedule. |
| `BATCH_SIZE`, `NUM_EPOCHS`, `PATIENCE`, `TOLERANCE`, `INIT_KEY` | Training loop settings. |
| `RESUME_FULL_STATE`, `CKPT_DIR_RESTORE` | Set `RESUME_FULL_STATE=True` and point `CKPT_DIR_RESTORE` at a checkpoint directory to resume training. |
| `ROOT_FOLDER`, `RUN_NAME` | Where outputs are written: `<ROOT_FOLDER>/run_<RUN_NAME>/`. |
| `LOSS_NAME` | Label used in plots/logs (training itself always uses MAE). |

Run:

```bash
python src/model_training.py
```

Because `model_training.py` reads `TRAINING_CONFIG` at import time, changing it inside a running Python/notebook session requires reloading the module:

```python
import importlib
import training_config

training_config.TRAINING_CONFIG.update({
    "NODES": 4,
    "MODEL_NAME": "HNN",
    "HIDDEN_DIM": 400,
    "DATA_PATH": "data/node4_test/dataset_merged.npz",
    "DATA_SIZE": None,
    "NUM_EPOCHS": 20,
    "ROOT_FOLDER": "./Models_notebook_test",
    "RUN_NAME": "HNN_4_quick_test",
})

import model_training
importlib.reload(model_training)

run_dir = model_training.train_surrogate_model()
```

The trained parameters are saved as `<run_dir>/params.msgpack`, used as `model_path` in the optimiser config.

---

## Inverse design optimisation

Edit `configs/optimiser_config.py` (`NPHOTONS`, `TARGET_NAME`, `MODEL_TYPE`, and the `OPTIMISER_CONFIG` dict):

| Setting | Meaning |
|---|---|
| `n` | Number of nodes/photons. Must match the trained model. |
| `target_name` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"` (also `"CLUSTER"`/`"LINEAR"`), `"SINGLE"`, or `"ZERO"`. `"SINGLE"` requires `2**n > 10` (i.e. `n >= 4`). |
| `model_type` | `"HNN"` or `"FNN"`. Must match the trained model. |
| `generate_data` | `True` generates fresh starting samples via `data_generation.generate_dataset()`; `False` loads samples from `conditioned_data_path`. |
| `conditioned_data_path` | Path to an existing `.npz` with `weights`/`amps` keys, used when `generate_data=False`. **Must be edited from the shipped absolute path if you use this mode.** |
| `architecture` | `HNN` integer hidden dim, or `FNN` tuple/list. Must match the trained model. |
| `model_path` | Path to the trained `params.msgpack`. **Must be edited from the shipped absolute path.** |
| `normalize_model_output` | Must match the training/data normalisation choice. |
| `input_dim`, `out_dim` | `2 * n * (n - 1)` and `2 ** n`. |
| `lambda_l1` | L1 sparsity weight. Objective is `loss = 1 - fidelity + lambda_l1 * sum(abs(x))`. |
| `num_steps`, `early_stop_nn_fid` | Optimisation step budget and early-stop fidelity. |
| `learning_rate`, `min_learning_rate`, `lr_decay_steps`, `lr_exponent` | Cosine-decay Adam schedule for the graph weights. |
| `clip_min`, `clip_max` | Bounds applied to graph weights after each update. |
| `print_every`, `verify_every` | Logging/PyTheus-verification frequency. |
| `store_step_vectors` | `True` stores every gradient/update vector — large JSON files for long runs; use `False` for production runs. |
| `prune_fid_tolerance`, `prune_thresholds` | Progressive-threshold pruning settings. |
| `results_root`, `folder_name` | Output location: `<results_root>/<folder_name>/`. |

Run:

```bash
python src/optimiser.py
```

`run_optimisation()` also accepts a config dict directly, which is the pattern used in the notebook:

```python
from optimiser import run_optimisation

CFG_OPT = {
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
    "model_path": "Models_notebook_test/run_HNN_4_quick_test/params.msgpack",
    "normalize_model_output": False,
    "input_dim": 24,
    "out_dim": 16,
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
}

result_dir = run_optimisation(CFG_OPT)
```

---

## Notebook workflow

`notebooks/workflow_data_training_optimisation_notebook.ipynb` runs all three stages interactively and is the version of this workflow that has actually been executed end-to-end (its saved cell outputs show a full `NODES=4` run completing successfully on CPU).

The notebook's first code cell sets:

```python
REPO_DIR = Path("/home/bo48god/projects/deep_dreaming/surrogate_model_clean_all_codes_with_config/surrogate_model_clean")
```

**This path must be edited to your own repository location before running the notebook.** The notebook then inserts `REPO_DIR/src` and `REPO_DIR/configs` into `sys.path`, generates data, merges shards, trains a model, and runs inverse optimisation, all with small `NODES=4` test values. After confirming the workflow runs, increase `NODES`, sample counts, epochs, and optimisation steps for a real run, and move final settings back into the `configs/*.py` files for terminal/cluster use.

---

## Expected outputs

**Data generation** (`OUT_DIR`):

```text
data/smoke_test/
├── data_00000.npz
├── data_00001.npz
├── metadata.json
├── dataset_merged.npz      # after merging
└── logs/
    └── data_generation_<timestamp>.log
```

**Model training** (`<ROOT_FOLDER>/run_<RUN_NAME>/`):

```text
Models_smoke_test/run_HNN_4_smoke_test/
├── checkpoints/
├── params_snapshots/
├── plots/
│   ├── training_curves.png
│   └── test_fidelity_curve.png
├── model_info.json
├── params.msgpack
├── run_log.txt
└── test_metrics.npz
```

**Inverse optimisation** (`<results_root>/<folder_name>/`):

```text
optimiser_notebook_results/GHZ_4_quick_test/
├── best_graph_solution.json
├── cfg.json
├── optimisation_summary.json
├── zero_state_samples.json
├── log.txt
├── edge_vs_fidelity.png
├── time_per_sample.png
├── cumulative_time.png
└── sample_0/, sample_1/, ...
    ├── sample_info_file.json
    ├── loss_fid_curve.png
    └── gradient_norm.png
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pytheus'`**
Run `pip install pytheusQ` (not `pip install pytheus` — that installs an unrelated Prometheus metrics package on PyPI). `pytheusQ` is the distribution name; it installs the `pytheus` module that `data_generation_utils.py` and `optimisation_utils.py` import.

**`ModuleNotFoundError: No module named 'training_config'` (or `data_config'` / `optimiser_config'`)**
`src/` and `configs/` are not on `PYTHONPATH`. Set it for the current session:

```bash
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"     # Linux/macOS
```
```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\configs;$env:PYTHONPATH"  # Windows PowerShell
```

**Training or optimisation fails with a `FileNotFoundError` pointing at `/home/...`**
The committed `configs/training_config.py` and `configs/optimiser_config.py` ship with absolute paths from the original author's machine (`DATA_PATH`, `CKPT_DIR_RESTORE`, `model_path`, `conditioned_data_path`, `results_root`). Edit these to local paths before running.

**`Jax plugin configuration error` / `cuInit failed` on startup**
This means JAX tried to use a CUDA GPU plugin that doesn't match the available CUDA libraries. It is non-fatal — JAX falls back to `CpuDevice` and the scripts continue to run on CPU. To use a GPU, install the JAX build matching your machine's CUDA version.

**`HNN`/`FNN` architecture mismatch between training and optimisation**
`HIDDEN_DIM` in `configs/training_config.py` and `architecture` in `configs/optimiser_config.py` must use the same format and value: an integer for `HNN`, a tuple/list for `FNN`.

**Normalisation mismatch**
Keep `NORMED_DATA` (`data_config.py`), `NORMALIZE_MODEL_OUTPUT` (`training_config.py`), and `normalize_model_output` (`optimiser_config.py`) consistent with each other, or loss/fidelity values will not be comparable across stages.

---

## Citation / License

This repository does not currently include a `LICENSE` or `CITATION` file. Add one before distributing or relying on this code outside of personal/internal use.
