# Surrogate Model Clean

A JAX/Flax pipeline for photonic quantum-state design via a learned surrogate model.

```text
graph / edge weights  ->  surrogate model  ->  predicted amplitude vector
```

The pipeline has three stages:

1. **Data generation** — generate random dense graph weights and the corresponding quantum-state amplitude vectors using a PyTheus perfect-matching catalogue.
2. **Model training** — train a surrogate neural network (FNN or HNN) to learn the graph-weights → amplitudes map.
3. **Inverse design optimisation** — use the trained surrogate to optimise graph weights toward a target state (GHZ, W, linear-cluster, ...), verified against PyTheus and pruned to a sparse graph.

---

## Installation on Windows / Linux

Required packages are listed in `requirements.txt` (`jax`, `flax`, `optax`, `numpy`, `matplotlib`, `pytheusQ`). `jupyter`/`ipykernel` are only needed for the notebook and are not in `requirements.txt`.

> **Install `pytheusQ`, not `pytheus`.** PyPI's `pytheus` is an unrelated Prometheus metrics client. `pytheusQ` is the distribution that installs the actual `pytheus` module this code imports (`pytheus.theseus`).

> **Windows note:** installing `flax` pulls in `orbax-checkpoint`, which can fail on native Windows with `WinError 206: The filename or extension is too long`, even in a fresh virtual environment with a short path. This is a Windows path-length limitation, not a bug in this repository — see [Troubleshooting](#troubleshooting) before installing. **Native Windows support for the full training workflow is unverified and partial until this is resolved.** WSL2, native Linux, or an HPC/cluster environment is recommended for running the full pipeline (data generation, training, and optimisation).

### Linux / macOS

```bash
cd /path/to/surrogate_model_clean
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel   # only if you will use the notebook
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
```

### Windows (PowerShell)

```powershell
cd C:\path\to\surrogate_model_clean
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyter ipykernel   # only if you will use the notebook
$env:PYTHONPATH = "$PWD\src;$PWD\configs;$env:PYTHONPATH"
```

`PYTHONPATH` must be set in every new terminal session before running scripts. For GPU use, install the JAX build matching your CUDA version (see JAX's own install docs) — the code also runs on CPU, falling back to `CpuDevice` automatically if no GPU is found.

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

Scripts take no command-line arguments — each reads its settings from the matching file in `configs/` at import time. Generated data, trained models, checkpoints, logs, and optimisation outputs are gitignored and stay local.

<details>
<summary>File reference</summary>

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

</details>

---

## Minimal smoke test

Confirms the environment works end to end on a tiny 4-node example, matching the run already captured in the [notebook](#notebook-workflow).

`configs/data_config.py` ships ready to go. `configs/training_config.py` and `configs/optimiser_config.py` ship with production values and another machine's absolute paths — edit both before running.

**1. Generate data**

```bash
python src/data_generation.py
```
Output: shards in `data/smoke_test/`.

**2. Merge shards**

```bash
python -c "from pathlib import Path; from data_generation_utils import merge_shards_to_npz; print(merge_shards_to_npz(Path('data/smoke_test'), 'dataset_merged.npz'))"
```
Output: `data/smoke_test/dataset_merged.npz`.

**3. Train**

Edit `configs/training_config.py` with small test values (see dropdown), then:

```bash
python src/model_training.py
```
Output: `Models_smoke_test/run_HNN_4_smoke_test/params.msgpack`.

<details>
<summary>Smoke-test values for TRAINING_CONFIG</summary>

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

</details>

**4. Optimise**

Edit `configs/optimiser_config.py`: `NPHOTONS = 4`, `MODEL_TYPE = "HNN"`, and in `OPTIMISER_CONFIG` set `"architecture": 400`, `"generate_data": True`, and `"model_path"` to the `params.msgpack` from step 3. Then:

```bash
python src/optimiser.py
```
Output: results under `optimiser_notebook_results/` — see [Expected outputs](#expected-outputs).

---

## Data generation

Generates random dense graph weights and the corresponding amplitude vectors, saved as `.npz` shards.

**File to edit:** `configs/data_config.py`

**Command:**
```bash
python src/data_generation.py
```

**Expected output:** shard files (`data_00000.npz`, ...), `metadata.json`, and a `logs/` folder inside `OUT_DIR`.

<details>
<summary>Config reference (data_config.py)</summary>

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

</details>

<details>
<summary>Optional: call generate_dataset() directly with custom arguments</summary>

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

</details>

Training expects one merged file:

```bash
python -c "from pathlib import Path; from data_generation_utils import merge_shards_to_npz; print(merge_shards_to_npz(Path('data/node4_test'), 'dataset_merged.npz'))"
```

This produces `dataset_merged.npz` with keys `weights` and `amps`. For an `n`-node, dimension-2 system: `input_dim = 2 * n * (n - 1)`, `out_dim = 2 ** n`.

---

## Model training

Trains a surrogate (`FNN` or `HNN`) to learn the weights → amplitudes map.

**File to edit:** `configs/training_config.py` (`TRAINING_CONFIG`)

**Command:**
```bash
python src/model_training.py
```

**Expected output:** `<ROOT_FOLDER>/run_<RUN_NAME>/params.msgpack`, plus `model_info.json`, plots, checkpoints, and `test_metrics.npz`.

<details>
<summary>Config reference (TRAINING_CONFIG)</summary>

| Setting | Meaning |
|---|---|
| `NODES` | Number of nodes/photons in the dataset. Must match the dataset. |
| `MODEL_NAME` | `"HNN"` or `"FNN"`. |
| `HIDDEN_DIM` | Integer for `HNN`; tuple/list for `FNN`, e.g. `(2000, 2000, 2000)`. |
| `DATA_PATH` | Path to the merged `.npz` dataset. Must be edited from the shipped absolute path. |
| `DATA_SIZE` | Number of samples to use; `None` uses the full dataset. |
| `NORMALIZE_MODEL_OUTPUT` | Must match `NORMED_DATA` used during data generation. |
| `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT` | Dataset split fractions (must sum to 1.0). |
| `LEARNING_RATE`, `LR_AFTER_DECAY`, `LR_DECAY_UNTIL_EPOCH` | Cosine learning-rate schedule. |
| `BATCH_SIZE`, `NUM_EPOCHS`, `PATIENCE`, `TOLERANCE`, `INIT_KEY` | Training loop settings. |
| `RESUME_FULL_STATE`, `CKPT_DIR_RESTORE` | Set `RESUME_FULL_STATE=True` and point `CKPT_DIR_RESTORE` at a checkpoint directory to resume training. |
| `ROOT_FOLDER`, `RUN_NAME` | Where outputs are written: `<ROOT_FOLDER>/run_<RUN_NAME>/`. |
| `LOSS_NAME` | Label used in plots/logs (training itself always uses MAE). |

</details>

<details>
<summary>Optional: edit config and retrain from a notebook/REPL</summary>

`model_training.py` reads `TRAINING_CONFIG` at import time, so changing it in a running session requires reloading the module:

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

</details>

The resulting `params.msgpack` becomes `model_path` in the optimiser config.

---

## Inverse design optimisation

Uses the trained surrogate to optimise graph weights toward a target state, verified against PyTheus and pruned to a sparse graph.

**File to edit:** `configs/optimiser_config.py` (`NPHOTONS`, `TARGET_NAME`, `MODEL_TYPE`, `OPTIMISER_CONFIG`)

**Command:**
```bash
python src/optimiser.py
```

**Expected output:** results under `<results_root>/<folder_name>/` — see [Expected outputs](#expected-outputs).

<details>
<summary>Config reference (OPTIMISER_CONFIG)</summary>

| Setting | Meaning |
|---|---|
| `n` | Number of nodes/photons. Must match the trained model. |
| `target_name` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"` (also `"CLUSTER"`/`"LINEAR"`), `"SINGLE"`, or `"ZERO"`. `"SINGLE"` requires `2**n > 10` (i.e. `n >= 4`). |
| `model_type` | `"HNN"` or `"FNN"`. Must match the trained model. |
| `generate_data` | `True` generates fresh starting samples via `data_generation.generate_dataset()`; `False` loads samples from `conditioned_data_path`. |
| `conditioned_data_path` | Path to an existing `.npz` with `weights`/`amps` keys, used when `generate_data=False`. Must be edited from the shipped absolute path if used. |
| `architecture` | `HNN` integer hidden dim, or `FNN` tuple/list. Must match the trained model. |
| `model_path` | Path to the trained `params.msgpack`. Must be edited from the shipped absolute path. |
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

</details>

<details>
<summary>Optional: run with a custom config dict (as used in the notebook)</summary>

`run_optimisation()` accepts a config dict directly:

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

</details>

---

## Notebook workflow

`notebooks/workflow_data_training_optimisation_notebook.ipynb` runs all three stages interactively. Its saved output shows a full `NODES=4` run completing successfully on CPU.

Edit `REPO_DIR` in the first code cell to your own repository path before running:

```python
REPO_DIR = Path("/path/to/surrogate_model_clean")
```

After confirming the workflow runs, increase `NODES`, sample counts, epochs, and optimisation steps, then move final settings back into `configs/*.py` for terminal/cluster use.

---

## Expected outputs

<details>
<summary>Data generation — <code>OUT_DIR</code></summary>

```text
data/smoke_test/
├── data_00000.npz
├── data_00001.npz
├── metadata.json
├── dataset_merged.npz      # after merging
└── logs/
    └── data_generation_<timestamp>.log
```

</details>

<details>
<summary>Model training — <code>&lt;ROOT_FOLDER&gt;/run_&lt;RUN_NAME&gt;/</code></summary>

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

</details>

<details>
<summary>Inverse optimisation — <code>&lt;results_root&gt;/&lt;folder_name&gt;/</code></summary>

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

</details>

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pytheus'`**
Run `pip install pytheusQ`, not `pip install pytheus` (the latter is an unrelated Prometheus metrics package).

**`[WinError 206] The filename or extension is too long`** (native Windows only)
- **Cause:** Installing `flax` pulls in `orbax-checkpoint`, whose package contents include deeply nested file paths that exceed Windows' default 260-character path-length limit. This is a Windows/packaging limitation, not a bug in this repository's code.
- **Recommended solution:** Use WSL2, native Linux, or an HPC/cluster environment for the full training workflow. This is currently the most reliable path and the one this project's workflow has actually been verified on.
- **Alternative:** If you have administrator rights, enable Windows Long Path support (`gpedit.msc` → Computer Configuration → Administrative Templates → System → Filesystem → "Enable Win32 long paths", or set the registry value `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`), then retry the install in a new terminal.
- **Note:** Using a shorter project or virtual-environment path by itself may not be sufficient to avoid this error.
- **Status:** Native Windows is **partially supported** — installation of `flax`/`optax` and the full training/optimisation workflow are not confirmed to work on native Windows unless Long Path support is enabled and the workflow has been independently verified end to end.

**`ModuleNotFoundError: No module named 'training_config'`** (or `data_config`/`optimiser_config`)
`src/` and `configs/` are not on `PYTHONPATH` for this session — see [Installation](#installation-on-windows--linux).

**`FileNotFoundError` pointing at `/home/...`**
The committed configs ship with absolute paths from the original author's machine (`DATA_PATH`, `CKPT_DIR_RESTORE`, `model_path`, `conditioned_data_path`, `results_root`). Edit these to local paths.

**`Jax plugin configuration error` / `cuInit failed` on startup**
JAX tried to use a CUDA plugin that doesn't match the available CUDA libraries. Non-fatal — it falls back to `CpuDevice` and continues. Install a matching JAX/CUDA build to use a GPU.

**`HNN`/`FNN` architecture mismatch**
`HIDDEN_DIM` (training) and `architecture` (optimiser) must use the same format and value: an integer for `HNN`, a tuple/list for `FNN`.

**Normalisation mismatch**
Keep `NORMED_DATA`, `NORMALIZE_MODEL_OUTPUT`, and `normalize_model_output` consistent across data generation, training, and optimisation, or loss/fidelity values won't be comparable.

---

## Citation / License

This repository does not currently include a `LICENSE` or `CITATION` file. Add one before distributing or relying on this code outside of personal/internal use.
