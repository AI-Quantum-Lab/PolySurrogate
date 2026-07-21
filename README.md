# PolySurrogate

A JAX/Flax pipeline for photonic quantum-state inverse design via a learned surrogate model.

```text
graph weights  →  surrogate PNN  →  predicted (unnormalised) amplitude vector
```

The pipeline is three flat, numbered scripts — no config files, no `src/`/`configs/` split. Every tunable parameter lives at the top of the script that uses it.

1. **`01_data_generate.py`** — generate random dense graph weights and the corresponding unnormalised quantum-state amplitude vectors, using a PyTheus perfect-matching catalogue.
2. **`02_ml_model.py`** — train a surrogate PNN (Polynomial Neural Network) or FNN to learn the graph-weights → unnormalised amplitude map.
3. **`03_inverse_design.py`** — use the trained surrogate to optimise graph weights toward a target state (GHZ, W, linear-cluster, …), verified against PyTheus and pruned to a sparse graph.

<details>
<summary>Glossary — PNN, fidelity, GHZ, PyTheus, …</summary>

| Term | Meaning |
|------|---------|
| **PNN** | Polynomial Neural Network. One hidden layer with a monomial activation x^(n/2). The default surrogate model. |
| **FNN** | Feedforward Neural Network. Multi-layer network with GELU activations. |
| **Unnormalised amplitude vector** | The raw output of the quantum simulator (PyTheus). Not L2-normalised. The surrogate is trained to predict this directly. |
| **Fidelity** | Squared overlap between the generated state and the target state: fidelity = \|⟨ψ\|target⟩\|². A value close to 1 means the state closely matches the target. |
| **GHZ state** | Greenberger-Horne-Zeilinger state — a maximally entangled N-photon state. |
| **W state** | A different maximally entangled N-photon state with a distinct entanglement structure from GHZ. |
| **Linear-cluster state** | A graph state arranged in a 1D chain. Used in measurement-based quantum computing. |
| **PyTheus** | A graph-state source enumeration library that computes the exact quantum amplitude vector for a given photonic graph via perfect-matching catalogues. Installed from PyPI as `pytheusQ`. |
| **Inverse design** | Searching (optimising) over graph weights to find a configuration whose quantum state matches a desired target state. |

</details>

---

## Table of contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Data generation](#data-generation)
- [Model training](#model-training)
- [Inverse design optimisation](#inverse-design-optimisation)
- [Expected outputs](#expected-outputs)
- [Files not tracked by Git](#files-not-tracked-by-git)
- [Troubleshooting](#troubleshooting)
- [Plotting code and data](#plotting-code-and-data)
- [Citation / License](#citation--license)

---

## Repository structure

```text
PolySurrogate/
├── 01_data_generate.py   ← Stage 1: generate_dataset()
├── 02_ml_model.py         ← Stage 2: train_surrogate_model()
├── 03_inverse_design.py   ← Stage 3: run_optimisation()
├── utils.py                ← shared code used by all three stages (see below)
├── notebooks/
│   ├── sample_workflow.ipynb   ← start here
│   └── run_quick_test.py       ← one-command full-pipeline smoke test
├── plotting_code/           ← standalone notebooks reproducing manuscript figures
├── LICENSE
├── README.md
└── requirements.txt
```

Each numbered script is self-contained: parameters are top-of-file constants you edit directly, and each stage's `main()` function (`generate_dataset()`, `train_surrogate_model()`, `run_optimisation()`) also accepts an optional `cfg` dict to override those constants at runtime (used by the notebook and `run_quick_test.py`). Stages are connected only by the files they read/write on disk — Stage 2 never imports Stage 1's code, it just reads the `.npz` path Stage 1 produced (with one exception: Stage 3 does call Stage 1's `generate_dataset()` directly, to generate its own starting samples, so there is exactly one implementation of the data-generation logic in the repo).

<details>
<summary>What's in <code>utils.py</code></summary>

| Category | Functions |
|----------|-----------|
| PyTheus catalogue / amplitudes | `build_pytheus_catalog`, `compute_amplitudes`, `compute_amplitudes_no_norms`, `pytheus_state_from_x`, `generate_random_dense_weights`, `fidelity_np` |
| Model definitions | `FNN`, `PNN`, `create_model`, `load_trained_model`, `save_params_msgpack` |
| Reproducible PRNG key plan | `derive_root_keys`, `epoch_key_for`, `optimiser_jitter_root_key`, `optimiser_sample_key_for`, `optimiser_jitter_event_key_for` |
| Reproducibility manifest | `get_git_info`, `get_environment_info`, `hash_file`, `hash_array`, `hash_pytree`, `build_manifest`, `write_manifest`, `append_trace_line` |
| Precision / JSON I/O | `configure_precision`, `dtype_for_precision`, `deterministic_xla_env_is_set`, `to_jsonable`, `save_json`, `write_json` |

Only genuinely-shared code lives here. One deliberate exception: `02_ml_model.py`'s XLA-determinism bootstrap stays inline at the very top of that file (before `import jax`), since importing `utils.py` would itself trigger `import jax` and defeat the point.

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
cd /path/to/PolySurrogate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel   # optional, only for the notebook
```

### Windows (PowerShell)

```powershell
cd C:\path\to\PolySurrogate
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyter ipykernel   # optional, only for the notebook
```

No `PYTHONPATH` setup needed — there's no `src/`/`configs/` split to point at. Just run the numbered scripts from the project root. For GPU use, install the JAX build matching your CUDA version — the code falls back to CPU automatically if no GPU is found.

---

## Quick start

### Option A — Sample notebook (easiest, interactive)

```bash
cd PolySurrogate
source .venv/bin/activate
jupyter notebook notebooks/sample_workflow.ipynb
```

Open and run all cells top to bottom — every tunable parameter across all three stages is listed in one Parameters cell, no other edits needed. Outputs go to `results/sample_workflow/`. Runs in under two minutes on a laptop CPU.

### Option B — Quick-test script (fastest, terminal)

```bash
cd PolySurrogate
source .venv/bin/activate
python notebooks/run_quick_test.py
```

<details>
<summary>Expected terminal output</summary>

```text
PolySurrogate -- quick end-to-end smoke test

============================================================
  Stage 1 / 3 -- Data generation
============================================================
  Generated + merged in 0.6s -> results/quick_test/data_generation/n4/n4_0/dataset_merged.npz
  weights=(200, 24)  amps=(200, 16)
  [OK] merged dataset: results/quick_test/data_generation/n4/n4_0/dataset_merged.npz

============================================================
  Stage 2 / 3 -- Model training
============================================================
  Training complete in 4.4s -> results/quick_test/model_training/PNN/n4/n4_0
  [OK] params.msgpack
  [OK] model_info.json
  [OK] reproducibility_manifest.json
  [OK] training_curves.png

============================================================
  Stage 3 / 3 -- Inverse-design optimisation
============================================================
  Optimisation complete in 6.0s -> results/quick_test/inverse_design/GHZ_n4/GHZ_n4_0
  [OK] best_graph_solution.json
  [OK] optimisation_summary.json
  [OK] reproducibility_manifest.json
  [OK] log.txt

============================================================
  ALL STAGES PASSED  (13.3s total)
============================================================
```

</details>

> **Low fidelity in the demo is expected.** The quick test uses 200 samples, 10 training epochs, and 10 optimisation steps — far too few for meaningful results. The goal is to verify the pipeline runs end-to-end. For real results, increase the sample counts / epochs / steps in the script constants (or the notebook's Parameters cell).

### Option C — Run the scripts directly

```bash
cd PolySurrogate
python 01_data_generate.py    # writes results/data_generation/n4/n4_0/
python 02_ml_model.py          # edit DATA_PATH first if not n4_0; writes results/model_training/PNN/n4/n4_0/
python 03_inverse_design.py    # edit MODEL_PATH first if not n4_0; writes results/inverse_design/GHZ_n4/GHZ_n4_0/
```

Each script's parameters are top-of-file constants — open the file and edit them directly.

---

## Configuration

All parameters are top-of-file constants in the three numbered scripts — no external config files. Every constant can also be overridden per-call via a `cfg` dict argument to `generate_dataset()` / `train_surrogate_model()` / `run_optimisation()`.

> **For the full, comprehensive per-stage parameter reference** (every constant, its meaning, and cross-stage consistency requirements), see [`PARAMETERS.md`](PARAMETERS.md). The tables below are a condensed summary; `notebooks/sample_workflow.ipynb` exposes only a minimal subset of these.

<details>
<summary>01_data_generate.py — data generation parameters</summary>

| Setting | Default | Meaning |
|---------|---------|---------|
| `VERTICES` | `4` | Graph nodes / photons. Must match training and optimisation. |
| `DIMENSIONS` | `2` | Local dimension. Use `2` for qubit-like systems. |
| `N_SAMPLES` | `1000` | Number of random graph samples to generate. |
| `BATCH_SIZE` | `100` | Samples per JAX batch. |
| `SEED` | `0` | `np.random.default_rng` seed. |
| `SHARD_SIZE` | `1000` | Max samples per saved shard file. |
| `SAVE_DATA` | `True` | If `True`, writes shards + merged `.npz` + manifest to an auto-numbered folder. If `False`, nothing is written — only the in-memory arrays are returned. |
| `DATA_ROOT` | `"results/data_generation"` | Parent folder for output — the actual `n{VERTICES}/n{VERTICES}_{i}` subfolder is auto-numbered, never chosen by hand. |
| `NORMED_DATA` | `False` | `False` → store raw unnormalised amplitude vectors (what the surrogate is trained to predict). `True` → L2-normalise before saving. Must match `normed_data` in Stage 3. |

`generate_dataset()` always returns `(weights, amps, out_dir)` — the arrays are always populated; `out_dir` is the auto-numbered `Path` when `SAVE_DATA=True`, or `None` when `False`.

</details>

<details>
<summary>02_ml_model.py — model training parameters</summary>

| Setting | Default | Meaning |
|---------|---------|---------|
| `NODES` | `4` | Node count. Must match the dataset. |
| `MODEL_NAME` | `"PNN"` | `"PNN"` (Polynomial Neural Network) or `"FNN"` (Feedforward NN). |
| `HIDDEN_DIM` | `400` | Integer for PNN; tuple for FNN, e.g. `(400, 400, 400)`. |
| `DATA_PATH` | `"results/data_generation/n4/n4_0/dataset_merged.npz"` | Path to the merged dataset written by Stage 1. Edit to point at whichever `n{v}_{i}` run you want to train on. |
| `DATA_SIZE` | `None` | Cap samples; `None` uses all. |
| `NORMALIZE_MODEL_OUTPUT` | `False` | Whether to L2-normalise model output inside the loss function. Independent of `NORMED_DATA` — see the note below. |
| `TRAIN_SPLIT`/`VAL_SPLIT`/`TEST_SPLIT` | `0.8`/`0.1`/`0.1` | Dataset split ratios (must sum to 1.0). |
| `SPLIT_MODE` | `"contiguous"` | `"contiguous"` (no randomness) or `"shuffled"` (seed-derived permutation). |
| `NUM_EPOCHS` | `20` | Training epochs (smoke-test default). |
| `PATIENCE` | `10` | Early-stop patience in epochs. |
| `BATCH_SIZE` | `100` | Mini-batch size. |
| `SEED` | `159` | Master seed — every PRNG key (model init, data split, epoch shuffling) is derived from this via `fold_in`, never sequential `split()`, so resumed runs reproduce exactly. |
| `PRECISION` | `"float32"` | `"float32"` or `"float64"` (`jax_enable_x64`, slower/more memory). |
| `DATASET_HASH_MODE` | `"fast"` | `"fast"` (size/mtime + partial hash), `"full"` (SHA-256 of entire file), or `"none"` — used to fingerprint `DATA_PATH` in the manifest. |
| `DETERMINISTIC_XLA` | `True` | Best-effort request for GPU-deterministic reductions (see the note in the file — most reliable when set via shell `XLA_FLAGS` before launch). |
| `RESUME_FULL_STATE` / `CKPT_DIR_RESTORE` | `False` / — | Set `True` and point at an existing `checkpoints/` folder to resume a run. |
| `ROOT_FOLDER` | `"results/model_training"` | Output root — the actual `<MODEL_NAME>/n{NODES}/n{NODES}_{i}` subfolder is auto-numbered. |

`train_surrogate_model()` returns the auto-numbered run directory (a `Path`).

> **No collision handling needed** — unlike some pipelines, rerunning never reuses a folder (each run gets a fresh `n{v}_{i}`), so there's no `orbax` "destination already exists" risk from re-running with the same settings.

</details>

<details>
<summary>03_inverse_design.py — inverse design parameters</summary>

| Setting | Default | Meaning |
|---------|---------|---------|
| `NPHOTONS` | `4` | Node count. Must match the trained model. |
| `TARGET_NAME` | `"GHZ"` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"`, `"SINGLE"`, or `"ZERO"`. |
| `MODEL_TYPE` | `"PNN"` | Must match the trained model (`"PNN"` or `"FNN"`). |
| `GENERATE_DATA` | `True` | `True` → generate fresh starting graphs on the fly via Stage 1's own `generate_dataset()` (`save_data=False`, nothing written to disk). `False` → load from `CONDITIONED_DATA_PATH`. |
| `DATA_SAMPLES` | `5` | How many starting samples to optimise. |
| `DATA_SEED` | `4` | Seed for generating starting samples. |
| `LOW_FIDELITY_THRESHOLD` | `0.999` | Fresh samples already this close to the target are discarded and replaced. |
| `NORMED_DATA` | `False` | Must match `NORMED_DATA` from data generation. |
| `ARCHITECTURE` | `400` | Must match `HIDDEN_DIM` from training. |
| `MODEL_PATH` | `"results/model_training/PNN/n4/n4_0/params.msgpack"` | **Edit this** after training, to point at the run you want to use. |
| `NORMALIZE_MODEL_OUTPUT` | `False` | Must match `NORMALIZE_MODEL_OUTPUT` from training. |
| `LAMBDA_L1` | `1e-3` | L1 penalty weight — higher pushes toward sparser (fewer-edge) solutions. |
| `SEED` | `46` | Master seed — drives all reproducible jitter noise. |
| `NUM_STEPS` / `MAX_TOTAL_STEPS` | `20` | Max optimisation steps per sample (smoke-test default). |
| `EARLY_STOP_NN_FID` | `0.99999` | Stopping condition: `nn_fid >= EARLY_STOP_NN_FID OR step >= MAX_TOTAL_STEPS` (single threshold, no escalation). |
| `JITTER_ENABLED` | `True` | Master switch for stall-detection jitter (reproducible noise injected if fidelity plateaus). |
| `PRUNE_THRESHOLDS` | `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]` | Progressive weight-zeroing thresholds tried after optimisation. |
| `RESULTS_ROOT` | `"results/inverse_design"` | Output root — the actual `<TARGET>_n{NPHOTONS}/<TARGET>_n{NPHOTONS}_{i}` subfolder is auto-numbered. |

`run_optimisation()` returns the auto-numbered run directory (a `Path`).

</details>

<details>
<summary>Normalisation note — two independent flag pairs</summary>

| Flag | Controls |
|------|---------|
| `NORMED_DATA` (Stage 1 / Stage 3) | Whether amplitude vectors are L2-normalised **in the dataset** before saving / in generated starting samples. |
| `NORMALIZE_MODEL_OUTPUT` (Stage 2 / Stage 3) | Whether the model output is L2-normalised **inside the loss / fidelity evaluation** at training / optimisation time. |

**Required consistency:**
- `NORMED_DATA` (Stage 1) **must** equal `NORMED_DATA` (Stage 3).
- `NORMALIZE_MODEL_OUTPUT` (Stage 2) **must** equal `NORMALIZE_MODEL_OUTPUT` (Stage 3).
- The two pairs are **independent** of each other.

Default: both pairs are `False`. The model is trained to predict raw unnormalised amplitude vectors.

</details>

---

## Data generation

```bash
python 01_data_generate.py
```

<details>
<summary>Call generate_dataset() directly with custom arguments</summary>

```python
from importlib import import_module
stage1 = import_module("01_data_generate")   # "01_data_generate" isn't a valid bare import target

weights, amps, out_dir = stage1.generate_dataset(
    vertices=4,
    dimensions=2,
    n_samples=1000,
    batch_size=100,
    seed=59,
    shard_size=500,
    normed_data=False,   # False = raw unnormalised (project default)
    save_data=True,
    data_root="results/data_generation",
)
# out_dir is the auto-numbered Path, e.g. results/data_generation/n4/n4_1
```

</details>

---

## Model training

```bash
# Edit DATA_PATH at the top of 02_ml_model.py to point at your merged dataset
python 02_ml_model.py
```

The resulting `params.msgpack` is the `MODEL_PATH` for Stage 3.

<details>
<summary>Override config at runtime (notebook / REPL)</summary>

```python
from importlib import import_module
stage2 = import_module("02_ml_model")

cfg = dict(stage2._default_cfg())
cfg.update({
    "DATA_PATH": "results/data_generation/n4/n4_1/dataset_merged.npz",
    "NUM_EPOCHS": 20,
    "SEED": 42,
})

run_dir = stage2.train_surrogate_model(cfg)
```

</details>

---

## Inverse design optimisation

```bash
# Edit MODEL_PATH, NPHOTONS, TARGET_NAME, ARCHITECTURE at the top of
# 03_inverse_design.py to match your trained model
python 03_inverse_design.py
```

<details>
<summary>Run with a custom config dict (as used in the notebook)</summary>

```python
from importlib import import_module
stage3 = import_module("03_inverse_design")

cfg = dict(stage3._default_cfg())
cfg.update({
    "model_path": "results/model_training/PNN/n4/n4_0/params.msgpack",
    "target_name": "GHZ",
    "data_samples": 5,
    "num_steps": 200,
    "max_total_steps": 200,
})

result_dir = stage3.run_optimisation(cfg)
```

</details>

---

## Expected outputs

All three stages write under one `results/` folder, one auto-numbered subfolder per run — no folder name is ever chosen by hand.

<details>
<summary>Data generation — <code>results/data_generation/n{VERTICES}/n{VERTICES}_{i}/</code></summary>

```text
results/data_generation/n4/n4_0/
├── data_00000.npz
├── metadata.json
├── dataset_merged.npz          ← use as DATA_PATH in 02_ml_model.py
├── reproducibility_manifest.json
└── logs/
    └── data_generation_<timestamp>.log
```

</details>

<details>
<summary>Model training — <code>results/model_training/{PNN|FNN}/n{NODES}/n{NODES}_{i}/</code></summary>

```text
results/model_training/PNN/n4/n4_0/
├── checkpoints/
├── params_snapshots/
├── plots/
│   ├── training_curves.png
│   └── test_fidelity_curve.png
├── model_info.json
├── params.msgpack               ← use as MODEL_PATH in 03_inverse_design.py
├── reproducibility_manifest.json
├── run_log.txt
└── test_metrics.npz
```

</details>

<details>
<summary>Inverse optimisation — <code>results/inverse_design/{TARGET}_n{NPHOTONS}/{TARGET}_n{NPHOTONS}_{i}/</code></summary>

```text
results/inverse_design/GHZ_n4/GHZ_n4_0/
├── best_graph_solution.json
├── cfg.json
├── optimisation_summary.json
├── zero_state_samples.json
├── reproducibility_manifest.json
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

Generated outputs should never be committed — only the code, notebooks, and docs are tracked:

| Pattern | What it covers |
|---------|---------------|
| `results/` | All three stages' auto-numbered outputs (datasets, trained models, optimisation runs) |
| `*.npz`, `*.dat` | Shard and merged data files |
| `*.msgpack` | Serialised model parameters |
| `checkpoints/` | Orbax checkpoint directories |
| `*.log`, `*.out`, `*.err` | Log files |
| `.venv/`, `__pycache__/` | Environment and bytecode |
| `.ipynb_checkpoints/` | Jupyter checkpoint files |

---

## Troubleshooting

<details>
<summary>Common issues and fixes</summary>

**`ModuleNotFoundError: No module named 'pytheus'`**
Run `pip install pytheusQ`, not `pip install pytheus`.

**`FileNotFoundError` on `DATA_PATH` or `MODEL_PATH`**
These are literal paths to a specific auto-numbered run folder (e.g. `n4_0`). Check that the folder actually exists — if Stage 1/2 was rerun, the newest data may be in `n4_1`, `n4_2`, etc., and the constant needs updating to match.

**`WARNING:absl:Tensorflow library not found`**
Harmless. This warning comes from the Flax/orbax checkpoint backend and only means that TensorFlow-specific checkpoint conversion is unavailable — which is not used in this workflow.

**`WARNING:absl:The 'aggregate' option is deprecated`**
Also harmless — a Flax checkpoint API deprecation warning. Does not affect correctness.

**`[WinError 206] The filename or extension is too long`** (Windows only)
Installing `flax` pulls in `orbax-checkpoint`, which exceeds Windows' 260-character path limit. Enable Windows Long Path support (`gpedit.msc` → System → Filesystem → "Enable Win32 long paths"), or use WSL2 / native Linux.

**`PNN`/`FNN` architecture mismatch**
`HIDDEN_DIM` (Stage 2) and `ARCHITECTURE` (Stage 3) must be the same value and type: an integer for PNN, a tuple/list for FNN.

**Normalisation mismatch**
Two independent pairs must each match internally — see [Configuration](#configuration):
- `NORMED_DATA` (Stage 1) ↔ `NORMED_DATA` (Stage 3)
- `NORMALIZE_MODEL_OUTPUT` (Stage 2) ↔ `NORMALIZE_MODEL_OUTPUT` (Stage 3)

**`Jax plugin configuration error` / `cuInit failed` on startup**
JAX tried to use a CUDA plugin that doesn't match the available CUDA libraries. Non-fatal — it falls back to CPU automatically. Install a matching JAX/CUDA build to use a GPU.

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

This folder is **separate from the main demo workflow** and independent of the three numbered pipeline scripts — it reproduces analysis figures from full-scale production runs, using its own pre-computed data.

To regenerate the plots:

```bash
cd plotting_code
jupyter notebook
```

Open each notebook and run all cells top to bottom. All paths are relative to
`plotting_code/` — no path setup needed. Only `numpy` and `matplotlib`
are required.

</details>

---

## Citation / License

This code is released under the MIT License — see [LICENSE](LICENSE).

If you use this work in research, please cite the associated paper (to be added).
