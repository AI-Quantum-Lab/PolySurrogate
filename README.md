# Surrogate Model Clean

A JAX/Flax pipeline for photonic quantum-state inverse design via a learned surrogate model.

```text
graph weights  →  surrogate PNN  →  predicted (unnormalised) amplitude vector
```

The pipeline has three stages:

1. **Data generation** — generate random dense graph weights and the corresponding unnormalised quantum-state amplitude vectors using a PyTheus perfect-matching catalogue.
2. **Model training** — train a surrogate PNN (Polynomial Neural Network) to learn the graph-weights → unnormalised amplitude map.
3. **Inverse design optimisation** — use the trained surrogate to optimise graph weights toward a target state (GHZ, W, linear-cluster, …), verified against PyTheus and pruned to a sparse graph.

<details>
<summary>Glossary — PNN, fidelity, GHZ, PyTheus, …</summary>

| Term | Meaning |
|------|---------|
| **PNN** | Polynomial Neural Network. One hidden layer with a monomial activation x^(n/2). The default surrogate model. |
| **FNN** | Feedforward Neural Network. Multi-layer network with GELU activations. |
| **Unnormalised amplitude vector** | The raw output of the quantum simulator (PyTheus). Not L2-normalised. The PNN is trained to predict this directly. |
| **Fidelity** | Squared overlap between the generated state and the target state: fidelity = \|⟨ψ\|target⟩\|². A value close to 1 means the state closely matches the target. |
| **GHZ state** | Greenberger-Horne-Zeilinger state — a maximally entangled N-photon state. |
| **W state** | A different maximally entangled N-photon state with a distinct entanglement structure from GHZ. |
| **Linear-cluster state** | A graph state arranged in a 1D chain. Used in measurement-based quantum computing. |
| **PyTheus** | A graph-state source enumeration library that computes the exact quantum amplitude vector for a given photonic graph via perfect-matching catalogues. Installed from PyPI as `pytheusQ`. |
| **Inverse design** | Searching (optimising) over graph weights to find a configuration whose quantum state matches a desired target state. |

</details>

---

## Repository structure

```text
surrogate_model_clean/
├── configs/            ← all user-facing settings (edit these)
│   ├── data_config.py
│   ├── training_config.py
│   └── optimiser_config.py
├── notebooks/
│   └── sample_workflow.ipynb   ← start here
├── scripts/
│   └── run_quick_test.py       ← one-command full-pipeline test
├── src/                ← core library (no edits needed for standard runs)
├── LICENSE
├── WORKFLOW.md         ← detailed walkthrough for new users
└── requirements.txt
```

<details>
<summary>Full file reference</summary>

| File | Purpose |
|------|---------|
| `configs/data_config.py` | Settings for data generation. |
| `configs/training_config.py` | Settings for surrogate model training. |
| `configs/optimiser_config.py` | Settings for inverse-design optimisation. |
| `src/data_generation.py` | Main data-generation script (`generate_dataset()`). |
| `src/data_generation_utils.py` | PyTheus catalogue construction, amplitude computation, shard merging. |
| `src/model_training.py` | Main training script (`train_surrogate_model()`). |
| `src/model_training_utils.py` | Dataset loading/splitting, training/eval/test steps, checkpointing, plotting. |
| `src/models.py` | Model architectures: `FNN`, `PNN`, and `create_model()`. |
| `src/optimiser.py` | Main inverse-design script (`run_optimisation()`). |
| `src/optimisation_utils.py` | Model loading, optimisation step, PyTheus verification, pruning, plotting. |
| `src/target_states.py` | Target-state definitions: GHZ, W, linear-cluster, single, zero. |
| `scripts/run_quick_test.py` | One-command end-to-end smoke test (~10 s on CPU). |
| `WORKFLOW.md` | Detailed walkthrough for new users. |
| `notebooks/sample_workflow.ipynb` | **Start here.** Beginner-friendly notebook; all stages; no config edits needed. |

</details>

---

## Installation

**Python ≥ 3.9 recommended.** JAX and Flax compatibility can vary by platform and CUDA version — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) if you need GPU support.

Required packages: `jax`, `flax`, `optax`, `numpy`, `matplotlib`, `pytheusQ`.  
`jupyter`/`ipykernel` are only needed for the notebook.

> **Install `pytheusQ`, not `pytheus`.** PyPI's `pytheus` is an unrelated Prometheus metrics client. `pytheusQ` is the distribution that installs the `pytheus` module (`pytheus.theseus`) that this code imports.

> **Windows note:** installing `flax` pulls in `orbax-checkpoint`, which can fail on native Windows with `WinError 206: The filename or extension is too long`. WSL2, native Linux, or an HPC/cluster environment is recommended — see [Troubleshooting](#troubleshooting).

### Linux / macOS

```bash
cd /path/to/surrogate_model_clean
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: only needed for the notebook
pip install jupyter ipykernel
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
```

### Windows (PowerShell)

```powershell
cd C:\path\to\surrogate_model_clean
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: only needed for the notebook
pip install jupyter ipykernel
$env:PYTHONPATH = "$PWD\src;$PWD\configs;$env:PYTHONPATH"
```

`PYTHONPATH` must be set in every new terminal session before running scripts.  For GPU use, install the JAX build matching your CUDA version — the code falls back to CPU automatically if no GPU is found.

---

## Quick start

### Option A — Sample notebook (easiest, interactive)

```bash
cd surrogate_model_clean
source .venv/bin/activate
pip install jupyter ipykernel   # first time only
jupyter notebook notebooks/sample_workflow.ipynb
```

Open and run all cells top to bottom — no config edits needed.  Outputs go to `results/sample_workflow/`.  Runs in under two minutes on a laptop CPU.

### Option B — Quick-test script (fastest, terminal)

```bash
cd surrogate_model_clean
source .venv/bin/activate
PYTHONPATH="$PWD/src:$PWD/configs" python scripts/run_quick_test.py
```

<details>
<summary>Expected terminal output</summary>

```text
surrogate_model_clean — quick end-to-end smoke test

============================================================
  Stage 1 / 4 — Data generation
============================================================
  Generated 1 shard(s) in 0.2s -> data/quick_test

============================================================
  Stage 2 / 4 — Shard merging
============================================================
  Merged in 0.0s -> data/quick_test/dataset_merged.npz

============================================================
  Stage 3 / 4 — Model training
============================================================
  Training complete in 3.4s -> Models_quick_test/run_PNN_4_quick_test
  [OK] params.msgpack
  [OK] model_info.json
  [OK] training_curves.png

============================================================
  Stage 4 / 4 — Inverse-design optimisation
============================================================
  Optimisation complete in 5.1s -> optimiser_quick_test/GHZ_4_quick_test
  [OK] best_graph_solution.json
  [OK] optimisation_summary.json
  [OK] log.txt

============================================================
  ALL STAGES PASSED  (11.0s total)
============================================================
```

</details>

> **Low fidelity in the demo is expected.** The quick test uses 200 samples, 10 epochs, and 10 optimisation steps — far too few for meaningful results. The goal is to verify the pipeline runs end-to-end. For real results use the notebook (500 samples, 15 epochs) or the full production pipeline.

---

## Configuration

All user-facing settings are in `configs/`.  The configs ship with **smoke-test defaults** (4 nodes, small sample counts, relative paths) that work out of the box.  Each file also contains a commented-out production block for large HPC runs.

<details>
<summary>data_config.py — data generation parameters</summary>

| Setting | Default | Meaning |
|---------|---------|---------|
| `VERTICES` | `4` | Graph nodes / photons. Must match training and optimisation. |
| `DIMENSIONS` | `2` | Local dimension. Use `2` for qubit-like systems. |
| `N_SAMPLES` | `1000` | Number of random graph samples to generate. |
| `BATCH_SIZE` | `100` | Samples per JAX batch. |
| `SEED` | `0` | Random seed. |
| `SAVE_TO_FILE` | `True` | Write `.npz` shards to disk. |
| `OUT_DIR` | `'data/smoke_test'` | Output folder (relative to project root). |
| `SHARD_SIZE` | `1000` | Samples per shard file. |
| `NORMED_DATA` | `False` | `False` → store raw unnormalised amplitude vectors. `True` → L2-normalise before saving. Must match `normed_data` in optimiser config. |

</details>

<details>
<summary>training_config.py — model training parameters</summary>

| Setting | Default | Meaning |
|---------|---------|---------|
| `NODES` | `4` | Node count. Must match the dataset. |
| `MODEL_NAME` | `"PNN"` | `"PNN"` (Polynomial Neural Network) or `"FNN"` (Feedforward NN). |
| `HIDDEN_DIM` | `400` | Integer for PNN; tuple for FNN, e.g. `(2000, 2000, 2000)`. |
| `DATA_PATH` | `"data/smoke_test/dataset_merged.npz"` | Path to merged dataset. Edit for your own data. |
| `DATA_SIZE` | `None` | Cap samples; `None` uses all. |
| `NORMALIZE_MODEL_OUTPUT` | `False` | Whether to L2-normalise model output inside the loss function. Independent of `NORMED_DATA` — see normalisation note. |
| `NUM_EPOCHS` | `20` | Training epochs (smoke test). Use `20000` for production. |
| `PATIENCE` | `10` | Early-stop patience in epochs. |
| `BATCH_SIZE` | `100` | Mini-batch size. |
| `ROOT_FOLDER` | `"Models_smoke_test"` | Output root. |
| `RUN_NAME` | `"PNN_4_smoke_test"` | Run sub-folder name. |
| `RESUME_FULL_STATE` | `False` | Set `True` to resume from `CKPT_DIR_RESTORE`. |

</details>

<details>
<summary>optimiser_config.py — inverse design parameters</summary>

| Setting | Default | Meaning |
|---------|---------|---------|
| `NPHOTONS` | `4` | Node count. Must match the trained model. |
| `TARGET_NAME` | `"GHZ"` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"`, `"SINGLE"`, or `"ZERO"`. |
| `MODEL_TYPE` | `"PNN"` | Must match the trained model (`"PNN"` or `"FNN"`). |
| `generate_data` | `True` | Generate fresh starts on-the-fly (no file needed). |
| `conditioned_data_path` | — | Used when `generate_data=False`. Edit to your `.npz` path. |
| `architecture` | `400` | Must match `HIDDEN_DIM` from training (int for PNN, tuple for FNN). |
| `model_path` | `"Models_smoke_test/run_PNN_4_smoke_test/params.msgpack"` | **Edit this** after training. |
| `normed_data` | `False` | Must match `NORMED_DATA` from data generation. |
| `normalize_model_output` | `False` | Must match `NORMALIZE_MODEL_OUTPUT` from training. |
| `num_steps` | `20` | Optimisation steps per sample (smoke test). Use `10000` for production. |
| `results_root` | `"optimiser_results"` | Output root folder. |
| `store_step_vectors` | `False` | `True` stores gradients per step (large files; use for debugging only). |

</details>

<details>
<summary>Normalisation note — two independent flag pairs</summary>

| Flag | Controls |
|------|---------|
| `NORMED_DATA` / `normed_data` | Whether amplitude vectors are L2-normalised **in the dataset** before saving / in generated starting samples. |
| `NORMALIZE_MODEL_OUTPUT` / `normalize_model_output` | Whether the model output is L2-normalised **inside the loss / fidelity evaluation** at training / optimisation time. |

**Required consistency:**
- `NORMED_DATA` (data config) **must** equal `normed_data` (optimiser config).
- `NORMALIZE_MODEL_OUTPUT` (training config) **must** equal `normalize_model_output` (optimiser config).
- The two pairs are **independent** of each other.

Default: both pairs are `False`. The model is trained to predict raw unnormalised amplitude vectors.

</details>

---

## Data generation

```bash
# Edit configs/data_config.py first (or use defaults for smoke test)
python src/data_generation.py

# Then merge shards into one file for training
python -c "
from pathlib import Path
from data_generation_utils import merge_shards_to_npz
merge_shards_to_npz(Path('data/smoke_test'), 'dataset_merged.npz')
"
```

<details>
<summary>Call generate_dataset() directly with custom arguments</summary>

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
    normed_data=False,  # False = raw unnormalised (project default)
)
```

</details>

---

## Model training

```bash
# Edit configs/training_config.py: set DATA_PATH to your merged dataset
python src/model_training.py
```

The resulting `params.msgpack` is the `model_path` for the optimiser.

> **Re-run note:** if training was already run for this `RUN_NAME`, delete the run folder before re-running (e.g. `rm -rf Models_smoke_test/run_PNN_4_smoke_test/`). orbax-checkpoint raises `ValueError: Destination already exists` if checkpoints are present.

<details>
<summary>Override config at runtime (notebook / REPL)</summary>

```python
import importlib
import training_config

training_config.TRAINING_CONFIG.update({
    "NODES": 4,
    "MODEL_NAME": "PNN",
    "HIDDEN_DIM": 400,
    "DATA_PATH": "data/node4_test/dataset_merged.npz",
    "NUM_EPOCHS": 20,
    "ROOT_FOLDER": "./Models_test",
    "RUN_NAME": "PNN_4_quick",
})

import model_training
importlib.reload(model_training)

run_dir = model_training.train_surrogate_model()
```

</details>

---

## Inverse design optimisation

```bash
# Edit configs/optimiser_config.py:
#   - set model_path to the params.msgpack from training
#   - set NPHOTONS, TARGET_NAME, architecture to match your model
python src/optimiser.py
```

<details>
<summary>Run with a custom config dict (as used in the notebook)</summary>

```python
from optimiser import run_optimisation

CFG_OPT = {
    "n": 4,
    "dimensions": 2,
    "target_name": "GHZ",
    "model_type": "PNN",
    "generate_data": True,
    "conditioned_data_path": "",
    "max_initial_samples": None,
    "data_samples": 5,
    "data_batch_size": 5,
    "data_seed": 4,
    "low_fidelity_threshold": 0.999,
    "save_generated_data": False,
    "data_out_dir": "data/optimiser_generated",
    "normed_data": False,
    "generation_gpu_batch_size": 5,
    "data_shard_size": 5,
    "architecture": 400,
    "model_path": "Models_smoke_test/run_PNN_4_smoke_test/params.msgpack",
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
    "results_root": "optimiser_results",
    "folder_name": "GHZ_4_test",
}

result_dir = run_optimisation(CFG_OPT)
```

</details>

---

## Expected outputs

<details>
<summary>Data generation — <code>OUT_DIR/</code></summary>

```text
data/smoke_test/
├── data_00000.npz
├── metadata.json
├── dataset_merged.npz      ← after merging
└── logs/
    └── data_generation_<timestamp>.log
```

</details>

<details>
<summary>Model training — <code>&lt;ROOT_FOLDER&gt;/run_&lt;RUN_NAME&gt;/</code></summary>

```text
Models_smoke_test/run_PNN_4_smoke_test/
├── checkpoints/
├── params_snapshots/
├── plots/
│   ├── training_curves.png
│   └── test_fidelity_curve.png
├── model_info.json
├── params.msgpack          ← use as model_path in optimiser config
├── run_log.txt
└── test_metrics.npz
```

</details>

<details>
<summary>Inverse optimisation — <code>&lt;results_root&gt;/&lt;folder_name&gt;/</code></summary>

```text
optimiser_results/GHZ_4_smoke_test/
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

## Files not tracked by Git

<details>
<summary>Gitignored patterns — generated outputs (do not commit)</summary>

| Pattern | What it covers |
|---------|---------------|
| `data/` | All generated datasets |
| `*.npz` | Shard and merged data files |
| `Models*/` | Trained model directories |
| `*.msgpack` | Serialised model parameters |
| `checkpoints/` | Orbax checkpoint directories |
| `optimiser_*/`, `results/` | Optimisation output directories |
| `*.log`, `*.out`, `*.err` | Log files |
| `.venv/`, `__pycache__/` | Environment and bytecode |

</details>

---

## Troubleshooting

<details>
<summary>Common issues and fixes</summary>

**`ModuleNotFoundError: No module named 'pytheus'`**  
Run `pip install pytheusQ`, not `pip install pytheus`.

**`ModuleNotFoundError: No module named 'training_config'`** (or `data_config` / `optimiser_config`)  
`PYTHONPATH` does not include `src/` and `configs/` — see [Installation](#installation).

**`FileNotFoundError` on `DATA_PATH` or `model_path`**  
The config ships with relative paths that assume you run from the project root.  Check that your working directory is `surrogate_model_clean/` and that the file was actually generated.

**`ValueError: Destination .../checkpoints/checkpoint_N already exists`**  
Training was run before for this `RUN_NAME` and checkpoints already exist in the output folder.  orbax-checkpoint refuses to overwrite.  Fix: delete the run folder and retry.

```bash
rm -rf Models_smoke_test/run_PNN_4_smoke_test/
# or whichever run folder is specified in training_config.py ROOT_FOLDER/RUN_NAME
python src/model_training.py
```

**`WARNING:absl:Tensorflow library not found`**  
Harmless.  This warning comes from the Flax/orbax checkpoint backend and only means that TensorFlow-specific checkpoint conversion is unavailable — which is not used in this workflow.  Safe to ignore.

**`WARNING:absl:The 'aggregate' option is deprecated`**  
Also harmless.  A Flax checkpoint API deprecation warning.  Does not affect correctness.

**`[WinError 206] The filename or extension is too long`** (Windows only)  
Installing `flax` pulls in `orbax-checkpoint`, which exceeds Windows' 260-character path limit.  Enable Windows Long Path support (`gpedit.msc` → System → Filesystem → "Enable Win32 long paths"), or use WSL2 / native Linux.

**`PNN`/`FNN` architecture mismatch**  
`HIDDEN_DIM` (training) and `architecture` (optimiser) must be the same value and type: an integer for PNN, a tuple/list for FNN.

**Normalisation mismatch**  
Two independent pairs must each match internally:  
- `NORMED_DATA` (data) ↔ `normed_data` (optimiser)  
- `NORMALIZE_MODEL_OUTPUT` (training) ↔ `normalize_model_output` (optimiser)  
They do not need to equal each other.

**`Jax plugin configuration error` / `cuInit failed` on startup**  
JAX tried to use a CUDA plugin that doesn't match the available CUDA libraries.  Non-fatal — it falls back to CPU automatically.  Install a matching JAX/CUDA build to use a GPU.

</details>

---

## Plotting code and data

<details>
<summary>Reproduce manuscript figures from plotting_code/</summary>

The `plotting_code/` folder contains the notebooks and processed data used to
reproduce the figures shown in the manuscript/project.

| Notebook | What it produces |
|----------|-----------------|
| `training_curves.ipynb` | PNN vs FNN training loss dynamics for 4-, 6-, and 8-node systems (log-scale; colour and greyscale variants) |
| `testing_hist.ipynb` | Per-sample MAE distribution on the held-out test set for each node count |
| `jacobian.ipynb` | Jacobian sensitivity heatmaps comparing PNN and PyTheus input→output sensitivities, plus their absolute difference |
| `runtime.ipynb` | Runtime comparison between direct PyTheus search and surrogate-guided optimisation across GHZ, W, and LC targets |
| `initial_to_final_fid.ipynb` | 3×3 grid of initial vs final fidelity histograms for all target-state and node-count combinations |

`plotting_code/Data/` contains pre-computed results (training logs, test MAE arrays, Jacobian matrices, runtime measurements, optimisation summaries). `plotting_code/Results/` contains the final generated figures.

This folder is **separate from the main demo workflow**. The quick-test script and sample notebook show a small end-to-end run; `plotting_code/` reproduces analysis figures from full-scale production runs.

To regenerate the plots:

```bash
cd plotting_code
jupyter notebook
```

Open each notebook and run all cells top to bottom. All paths are relative to
`plotting_code/` — no `PYTHONPATH` changes needed. Only `numpy` and `matplotlib`
are required.

</details>

---

## Citation / License

This code is released under the MIT License — see [LICENSE](LICENSE).

If you use this work in research, please cite the associated paper (to be added).
