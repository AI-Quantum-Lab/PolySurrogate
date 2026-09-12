# PolySurrogate

A JAX/Flax pipeline for photonic quantum-state inverse design via a learned surrogate model:

```text
graph weights  →  surrogate (PNN/FNN)  →  predicted (unnormalised) amplitude vector
```

The pipeline is three flat, independent scripts — `data_generate.py`, `ml_model.py`, `inverse_design.py` — connected only through the files they read and write on disk (a dataset `.npz`, a model `.msgpack`), plus shared code in `utils.py`. There is no `src/`/`configs/` split; every tunable parameter lives at the top of the script that uses it.

<details>
<summary>Glossary — PNN, fidelity, GHZ, PyTheus, …</summary>

| Term | Meaning |
|------|---------|
| **PNN** | Polynomial Neural Network. One hidden `Dense` layer followed by a monomial activation `x ** (nodes / 2)`, then a final `Dense` layer. The project's primary surrogate architecture. |
| **FNN** | Feedforward Neural Network. Several `Dense` layers with GELU activations, used for comparison against PNN. |
| **Unnormalised amplitude vector** | The raw output of the quantum simulator (PyTheus) — not L2-normalised. The surrogate is trained to predict this directly. |
| **Fidelity** | Squared overlap between two quantum states: `fidelity = |⟨ψ|target⟩|²`. A value close to 1 means a close match. |
| **GHZ state** | Greenberger–Horne–Zeilinger state — a maximally entangled N-photon state. |
| **W state** | A different maximally entangled N-photon state, with a distinct entanglement structure from GHZ. |
| **Linear-cluster state** | A graph state arranged in a 1D chain, used in measurement-based quantum computing. |
| **PyTheus** | A graph-state source enumeration library that computes the exact quantum amplitude vector for a photonic graph via perfect-matching catalogues. Installed from PyPI as `pytheusQ`. |
| **Inverse design** | Searching (via gradient descent) over graph weights to find a configuration whose quantum state matches a target state. |

</details>

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Main Python Scripts](#main-python-scripts)
4. [Notebooks](#notebooks)
5. [Paper Data](#paper-data)
6. [Plotting Code](#plotting-code)
7. [Troubleshooting](#troubleshooting)
8. [Citation](#citation)

---

## Repository Structure

```text
PolySurrogate/
├── data_generate.py      Stage 1 — generate_dataset(): random graphs → quantum amplitudes
├── ml_model.py            Stage 2 — train(): train a PNN/FNN surrogate model
├── inverse_design.py    Stage 3 — run_optimisation(): surrogate-guided inverse design
├── utils.py                 Shared code: PyTheus catalogue/amplitudes, PNN/FNN model
│                           definitions, reproducible PRNG key plan, reproducibility
│                           manifests, JSON I/O helpers
├── PARAMETERS.md            Extended per-parameter reference for the three scripts above
├── notebooks/
│   ├── sample_workflow.ipynb   Small, verified, end-to-end demo (see Notebooks)
│   └── run_quick_test.py       Terminal smoke-test script — all 3 stages, ~20s (see Notebooks)
├── paper_data/               Curated starting-sample data + settings/provenance used for
│                           the manuscript's production runs (see Paper Data)
├── plotting_code/            Notebooks + data that reproduce the manuscript's figures
│                           (see Plotting Code)
├── LICENSE                  MIT License
├── README.md
└── requirements.txt
```

Running any of the three main scripts creates a `results/` folder locally (`results/data_generation/`, `results/model_training/`, `results/inverse_design/`) — this folder is **not tracked by git** (see `.gitignore`), since it holds large, regenerable, machine-specific outputs (datasets, model checkpoints, optimisation logs). Re-running a script never overwrites a previous run: each stage auto-numbers its own output folder (`n{N}_0`, `n{N}_1`, ...).

`.gitignore` also excludes generic `*.npz` files everywhere in the repo **except** under `paper_data/`, which is explicitly re-included — that folder's `.npz` datasets are tracked on purpose (see [Paper Data](#paper-data)).

---

## Installation

**Python 3.10+ recommended** (developed and tested with Python 3.12.7). JAX/Flax compatibility can vary by platform and CUDA version — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for GPU-specific builds.

### 1. Clone the repository

```bash
git clone https://github.com/AI-Quantum-Lab/PolySurrogate.git
cd PolySurrogate
```

### 2. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows (PowerShell): .venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Install `pytheusQ`, not `pytheus`.** PyPI's `pytheus` is an unrelated package. `pytheusQ` is the distribution that installs the `pytheus` module (`pytheus.theseus`) this code imports — `requirements.txt` already specifies the correct name.

### 4. GPU vs. CPU

`requirements.txt` installs a **CPU-only** JAX build by default — this is enough to run the small demo in `notebooks/sample_workflow.ipynb`. For GPU acceleration (needed for real-scale runs), install a matching CUDA build instead, for example:

```bash
pip install --upgrade "jax[cuda12]"
```

Match the CUDA extra (`cuda12`, `cuda11`, ...) to your system's driver — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html). If no compatible GPU/driver is found, JAX automatically falls back to CPU (see [Troubleshooting](#troubleshooting)).

### 5. Verify the installation

```bash
python -c "import jax, flax, optax, pytheus, numpy, matplotlib; print(jax.default_backend(), jax.devices())"
```

This should print `cpu` (or `gpu`, if a compatible GPU/driver is available) with no import errors. For a fuller check that exercises the actual pipeline, run through `notebooks/sample_workflow.ipynb` (see [Notebooks](#notebooks)) — it generates data, trains a model, and runs inverse design in well under a minute.

---

## Main Python Scripts

Each script keeps its parameters as top-of-file constants — edit them directly, then run the script. `PARAMETERS.md` has the extended per-parameter reference.

### `data_generate.py`

**Purpose:** generates the training dataset for the surrogate model — pairs of *(random graph edge weights) → (exact quantum amplitude vector)*, computed via a PyTheus perfect-matching catalogue.

**Dataset produced:** for an `n`-node graph, edge weights are drawn uniformly from `[-1, 1]`; the corresponding **unnormalised** amplitude vector is computed exactly (not L2-normalised — this is what the surrogate is trained to predict directly).

<details>
<summary><strong>Key parameters</strong></summary>

| Parameter | Default | Meaning |
|---|---|---|
| `VERTICES` | `4` | Graph nodes / photons. Must match `ml_model.py`'s `NODES` and `inverse_design.py`'s `NPHOTONS`. |
| `DIMENSIONS` | `2` | Local Hilbert-space dimension (`2` = qubit-like). |
| `N_SAMPLES` | `20_000_000` | Total samples to generate. |
| `BATCH_SIZE` | `5_000` | Samples per JAX batch during generation. |
| `SHARD_SIZE` | `500_000` | Max samples per intermediate shard file before merging. |
| `SEED` | `59` | `np.random.default_rng` seed. |
| `NORMED_DATA` | `False` | `False` → raw unnormalised amplitudes (project default). `True` → L2-normalised. Must match `inverse_design.py`'s `NORMED_DATA`. |
| `SAVE_DATA` | `True` | If `False`, nothing is written to disk — only in-memory arrays are returned. |

</details>

**How to run:**

```bash
python data_generate.py
```

**Where output is saved:** `results/data_generation/n{VERTICES}/n{VERTICES}_{i}/` (auto-numbered — never overwrites a previous run), containing:

```text
n4_0/
├── data_00000.npz, ...          shard files
├── dataset_merged.npz            merged dataset — use as DATA_PATH in ml_model.py
├── metadata.json
├── reproducibility_manifest.json  git commit, environment, seeds, dataset hash
└── logs/data_generation_<timestamp>.log
```

---

### `ml_model.py`

**Purpose:** trains the surrogate neural network on the dataset from `data_generate.py`, learning the graph-weights → amplitude-vector mapping.

**Supported model types** (set via `MODEL_NAME`):

- **`"PNN"`** (Polynomial Neural Network) — one hidden `Dense` layer of size `HIDDEN_DIM` (an integer) with a monomial activation, then a final `Dense` layer. The project's primary architecture.
- **`"FNN"`** (Feedforward Neural Network) — several `Dense` layers (sizes given as a tuple, e.g. `(400, 400, 400)`) with GELU activations, for comparison against PNN.

<details>
<summary><strong>Key parameters</strong></summary>

| Parameter | Default | Meaning |
|---|---|---|
| `NODES` | `8` | Must match the dataset's `VERTICES`. |
| `MODEL_NAME` | `"PNN"` | `"PNN"` or `"FNN"`. |
| `HIDDEN_DIM` | `15000` | Integer for PNN; tuple for FNN. |
| `DATA_PATH` | *(path to a `dataset_merged.npz`)* | **Edit this** to point at the dataset you want to train on. |
| `DATA_SIZE` | `10_000_000` | Cap on samples used; `None` uses the full dataset. |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | `0.8` / `0.1` / `0.1` | Dataset split ratios. |
| `SPLIT_MODE` | `"contiguous"` | `"contiguous"` (deterministic) or `"shuffled"` (seeded permutation). |
| `LEARNING_RATE` / `FINAL_LEARNING_RATE` / `LR_DECAY_UNTIL_EPOCH` | `1e-3` / `1e-5` / `2000` | Cosine learning-rate schedule. |
| `BATCH_SIZE` | `8000` | Mini-batch size (training set size must be divisible by this). |
| `NUM_EPOCHS` | `10` | Epoch cap — early stopping (`PATIENCE`) may end training sooner. |
| `PATIENCE` | `100000` | Early-stop patience, in epochs. |
| `SEED` | `158` | Master seed — model init, data split, and per-epoch shuffling are all derived from this via `jax.random.fold_in`. |
| `PRECISION` | `"float32"` | `"float32"` or `"float64"` (`jax_enable_x64`). |
| `ROOT_FOLDER` | `"results/model_training"` | Output root. |
| `RESUME_RUN_DIR` | `None` | Set to an existing run directory to resume training from its last checkpoint. |

</details>

**How to run:**

```bash
python ml_model.py    # edit DATA_PATH, NODES, MODEL_NAME first
```

**Where output is saved:** `results/model_training/{PNN|FNN}/n{NODES}/n{NODES}_{i}/`, containing:

```text
n8_0/
├── config.json                    resolved configuration for this run
├── run.log                        per-epoch train/validation loss
├── training_history.npy           structured array of per-epoch metrics
├── checkpoints/{latest,best}/      Flax/Orbax checkpoints
├── best_params.msgpack             params at the best validation epoch — use as MODEL_PATH
├── params.msgpack                  final restored-best params
├── test_metrics.npz                held-out test-set MAE/MSE/fidelity
├── summary.json                    final loss/fidelity summary
└── reproducibility_manifest.json   git commit, environment, seeds, dataset hash
```

---

### `inverse_design.py`

**Purpose:** uses a trained surrogate model as a differentiable proxy for PyTheus to search for graph weights whose predicted quantum state matches a target state, via gradient descent. The result is verified against the exact PyTheus simulator (not just the surrogate) and pruned to a sparse graph.

**How it uses the surrogate:** `MODEL_PATH` is loaded once at startup; gradient descent on the graph weights minimises `loss = (1 - fidelity) + LAMBDA_L1 * ||x||₁` with the surrogate providing the fidelity gradient. If optimisation stalls, reproducible jitter (escalating noise, seeded from `(SEED, sample_id, event_index)`) perturbs the trajectory. The final graph is checked against exact PyTheus and small weights are progressively pruned.

<details>
<summary><strong>Key parameters</strong></summary>

| Parameter | Default | Meaning |
|---|---|---|
| `NPHOTONS` | `4` | Must match the trained model's `NODES`. |
| `TARGET_NAME` | `"GHZ"` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"`, `"SINGLE"`, or `"ZERO"`. |
| `MODEL_TYPE` | `"PNN"` | Must match the trained model (`"PNN"` or `"FNN"`). |
| `GENERATE_DATA` | `False` | `False` → load starting graphs from `CONDITIONED_DATA_PATH`. `True` → generate fresh random starting graphs on the fly. |
| `CONDITIONED_DATA_PATH` | `paper_data/inverse_init_fin_data/n4/GHZ/random_start_pool.npz` | Starting-graph dataset, used when `GENERATE_DATA=False` (see [Paper Data](#paper-data)). |
| `ARCHITECTURE` | `400` | Must match `HIDDEN_DIM`/`MODEL_NAME` from training. |
| `MODEL_PATH` | *(path to a trained `.msgpack`)* | **Edit this** to point at your trained model. |
| `LAMBDA_L1` | `1e-3` | L1 penalty weight — higher pushes toward sparser (fewer-edge) graphs. |
| `SEED` | `46` | Master seed for all jitter noise. |
| `NUM_STEPS` / `MAX_TOTAL_STEPS` | `100000` | Max optimisation steps per sample. |
| `EARLY_STOP_NN_FID` | `0.99999` | Stop once the surrogate-predicted fidelity reaches this value. |
| `JITTER_ENABLED` | `True` | Master switch for stall-detection jitter. |
| `PRUNE_THRESHOLDS` | `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]` | Progressive weight-zeroing thresholds tried after optimisation. |
| `RESULTS_ROOT` | `"results/inverse_design"` | Output root. |

</details>

**How to run:**

```bash
python inverse_design.py    # edit MODEL_PATH, NPHOTONS, TARGET_NAME, ARCHITECTURE first
```

**Where output is saved:** `results/inverse_design/{TARGET}_n{NPHOTONS}/{TARGET}_n{NPHOTONS}_{i}/`, containing:

```text
GHZ_n4_0/
├── cfg.json                       resolved configuration for this run
├── log.txt                        per-step optimisation log
├── best_graph_solution.json       sparsest solution found, with its PyTheus fidelity
├── optimisation_summary.json      per-sample fidelities, steps, timing
├── zero_state_samples.json        any samples that collapsed to a zero vector
├── reproducibility_manifest.json  git commit, environment, seeds, model hash
├── edge_vs_fidelity.png, cumulative_time.png, time_per_sample.png
└── sample_0/, sample_1/, ...
    ├── sample_info_file.json      full per-sample trajectory and pruning log
    └── loss_fid_curve.png, gradient_norm.png
```

---

## Notebooks

The `notebooks/` folder contains:

| File | Purpose |
|---|---|
| **`sample_workflow.ipynb`** | Demonstrates the **complete workflow end to end**: generates a small dataset (5,000 samples), trains a 4-photon PNN (20 epochs), then uses it for inverse design against one of the generated samples' own quantum state. Deliberately tiny so it runs in well under a minute; not intended to produce research-quality results — see the notebook's own caveats. This is the notebook to start with. |
| `run_quick_test.py` | A terminal smoke-test script that runs all three stages without opening a notebook — generates 200 samples, trains a 4-photon PNN for 10 epochs, then runs a 3-sample/10-step GHZ inverse-design pass, checking that every expected output file was written at each stage. Runs in well under a minute (~20s on CPU). Useful for quickly verifying an installation or a code change without launching Jupyter. |

No other notebooks live in this folder — `plotting_code/` has its own separate set of notebooks for reproducing manuscript figures (see [Plotting Code](#plotting-code)).

**To launch:**

```bash
source .venv/bin/activate
jupyter notebook notebooks/sample_workflow.ipynb
```

Run all cells top to bottom; every configuration value used lives in one cell near the top. Outputs are written under `results/sample_workflow/`. (`jupyter`/`ipykernel` must be installed — see [Installation](#installation) / `requirements.txt`.)

Alternatively, run the terminal smoke test (no Jupyter needed):

```bash
python notebooks/run_quick_test.py
```

<details>
<summary><strong>Expected terminal output</strong></summary>

```text
PolySurrogate -- quick end-to-end smoke test
Project root: /path/to/PolySurrogate

============================================================
  Stage 1 / 3 -- Data generation
============================================================
  Generated + merged in 2.4s -> .../results/quick_test/data_generation/n4/n4_0/dataset_merged.npz
  weights=(200, 24)  amps=(200, 16)
  [OK] merged dataset: ...

============================================================
  Stage 2 / 3 -- Model training
============================================================
  Training complete in 7.5s -> .../results/quick_test/model_training/PNN/n4/n4_0
  [OK] best_params.msgpack: ...
  [OK] config.json: ...
  [OK] summary.json: ...
  [OK] reproducibility_manifest.json: ...

============================================================
  Stage 3 / 3 -- Inverse-design optimisation
============================================================
  Optimisation complete in 10.9s -> .../results/quick_test/inverse_design/GHZ_n4/GHZ_n4_0
  [OK] best_graph_solution.json: ...
  [OK] optimisation_summary.json: ...
  [OK] reproducibility_manifest.json: ...
  [OK] log.txt: ...

============================================================
  ALL STAGES PASSED  (20.6s total)
============================================================
```

</details>

---

## Paper Data

`paper_data/` is tracked in git (it is explicitly re-included in `.gitignore`, which otherwise excludes `*.npz` and `results/` everywhere else). It contains two kinds of content:

**1. Curated starting-sample data — `inverse_init_fin_data/n{4,6,8}/{GHZ,W,LINEAR_CLUSTER}/`**

For each combination of node count (4, 6, 8) and target state (GHZ, W, linear-cluster), this holds a fixed pool of 1,000 starting graphs pre-filtered to a controlled initial-fidelity range (roughly 0.01–0.5 to the target, per `summary_all_runs.json`):

```text
n4/GHZ/
├── paper_like_initial_fidelity_dataset.npz    the 1,000 curated starting graphs + amplitudes
├── random_start_pool.npz                       the pool inverse_design.py's CONDITIONED_DATA_PATH loads by default
├── paper_like_initial_fidelity_histogram.png
├── paper_like_initial_fidelity_metadata.json
└── run_status.json
```

This **is** the data `inverse_design.py` actually uses by default (`GENERATE_DATA=False`, `CONDITIONED_DATA_PATH` points here for n=4/GHZ) — running the script out of the box uses these exact starting conditions, so the starting point of any n4/n6/n8 GHZ/W/linear-cluster inverse-design run can be reproduced exactly.

**2. Settings/provenance references — `data_generation_settings.md`, `model_training_settings.md`, `inverse_design_settings.md`**

Markdown tables recording the *exact* `data_generate.py`, `ml_model.py`, and `inverse_design.py` parameter values, git commit, environment, and file hashes used to produce the n4/n6/n8 production datasets, surrogate models, and inverse-design runs — pulled directly from each run's own `cfg.json`/`config.json`/`reproducibility_manifest.json`. `inverse_design_settings.md` also confirms that every listed inverse-design parameter is identical across all three target states (GHZ, W, LINEAR_CLUSTER) and across system sizes, aside from the node-count- and target-dependent fields. Use these files to reproduce the same settings if you re-run the pipeline yourself.

**What is *not* included:** the actual trained model checkpoints and full optimisation outputs referenced by path in these settings files (under `results/model_training/`, `results/inverse_design/`) are **not** part of this repository — `results/` is excluded by `.gitignore`. Reproducing the exact reported numbers requires re-running `data_generate.py` → `ml_model.py` → `inverse_design.py` locally with the settings documented here (using the curated starting-sample data above as `CONDITIONED_DATA_PATH`), not just cloning the repo. Note also that as of this writing, the n=8 inverse-design runs (GHZ, W, LINEAR_CLUSTER) are complete and documented in `inverse_design_settings.md`, while FNN training at n=8 in `model_training_settings.md` is still in progress (a long-running resume job) and not yet finalised.

---

## Plotting Code

`plotting_code/` contains the notebooks and pre-computed data used to reproduce the manuscript's figures. It is **independent of the three main scripts and their `results/` output** — it works entirely from its own tracked `Data/` folder.

<details>
<summary><strong>Notebook → figure → input data</strong></summary>

| Notebook | Figure(s) produced | Input data (under `plotting_code/Data/`) |
|---|---|---|
| `training_curves.ipynb` | PNN vs FNN training-loss dynamics at n=4/6/8 (`Results/training_curves/`) | `training_data/{4,6,8}_{PNN,FNN}_model.json` |
| `testing_hist.ipynb` | Held-out test-set MAE distributions (`Results/testing_mae_histograms/`) | `testing_data/mae_values_n{4,6,8}.npy`, `mae_summary_n{4,6,8}.json` |
| `jacobian.ipynb` | PNN vs PyTheus input→output sensitivity heatmaps (`Results/jacobian_sensitivity/`) | `jacobian_data/{4,6,8}_node/*.npy`, `jacobian_summary.json` |
| `runtime.ipynb` | PyTheus-search vs surrogate-optimisation runtime comparison (`Results/runtime_bar/`) | `runtime_data/pytheus_data_time/{GHZ,LC,W}/{4,6,8}n_time_info.json`, `runtime_data/surrogate_optimisation/{GHZ,LC,W}/{4,6,8}n.json` |
| `initial_to_final_fid.ipynb` | 3×3 grid of initial-vs-final fidelity histograms across targets/node counts (`Results/initial_final_fid/`) | `runtime_data/surrogate_optimisation/{GHZ,LC,W}/{4,6,8}n.json` |

</details>

**How to run:**

```bash
cd plotting_code
jupyter notebook
```

Open a notebook and run all cells top to bottom — each resolves its own paths relative to `plotting_code/` (`Path.cwd()`), so no path setup is needed as long as the notebook is launched from inside that folder. Only `numpy` and `matplotlib` are required (already in `requirements.txt`).

Generated figures are written under `plotting_code/Results/<figure-name>/`, alongside the already-committed reference copies (`.png`/`.pdf`, some also `.svg`) — re-running a notebook overwrites those files with a freshly generated version from the same input data.

---

## Troubleshooting

<details>
<summary>Common issues and fixes</summary>

**`ModuleNotFoundError: No module named 'pytheus'`**
Run `pip install pytheusQ`, not `pip install pytheus` — see [Installation](#installation).

**`Jax plugin configuration error` / `cuInit(0) failed` / `Unknown CUDA error 303`**
JAX tried to initialise a CUDA plugin that doesn't match the available driver/GPU (or no GPU is visible, e.g. on a login node). This is **non-fatal** — JAX automatically falls back to CPU, reported by `jax.devices()` returning `[CpuDevice(...)]`. Install a JAX build matching your CUDA version to use a GPU (see [Installation](#installation)).

**CPU vs GPU execution is silent**
`jax.default_backend()` and `jax.devices()` (printed by `data_generate.py`/`ml_model.py`/`inverse_design.py` at startup) tell you which backend is actually active — always check this first if a run seems unexpectedly slow.

**`FileNotFoundError` on `DATA_PATH` / `MODEL_PATH` / `CONDITIONED_DATA_PATH`**
These are literal paths to one specific auto-numbered run folder (e.g. `n4_0`). If the referenced stage was re-run since, the newest output may be in `n4_1`, `n4_2`, etc. — check `results/.../n{N}/` for what actually exists and update the constant.

**`PNN`/`FNN` architecture mismatch between training and inverse design**
`HIDDEN_DIM` (`ml_model.py`) and `ARCHITECTURE` (`inverse_design.py`) must be the same value *and* type — an integer for PNN, a tuple for FNN.

**Normalisation mismatch**
`NORMED_DATA` (`data_generate.py`) must equal `NORMED_DATA` (`inverse_design.py`); `NORMALIZE_MODEL_OUTPUT` (`ml_model.py`) must equal `NORMALIZE_MODEL_OUTPUT` (`inverse_design.py`). These are two independent pairs.

**`ValueError` about training set size not divisible by `BATCH_SIZE`**
`ml_model.py` requires the training split's sample count to divide evenly by `BATCH_SIZE`. Pick a `BATCH_SIZE` that divides `N_SAMPLES * TRAIN_SPLIT`.

**GPU out-of-memory (`RESOURCE_EXHAUSTED` / CUDA OOM) with large datasets or `HIDDEN_DIM`**
This is GPU **device memory**, a fixed hardware limit per card — increasing host RAM (e.g. a job scheduler's `--mem`) does not help. Reduce `DATA_SIZE` (`ml_model.py`) or `HIDDEN_DIM`, use `PRECISION="float32"` instead of `"float64"` (roughly half the memory), or use a GPU with more VRAM. As one concrete data point from this project: the full 20M-sample dataset at `HIDDEN_DIM=15000` and `float64` needs about 50+ GB of GPU memory — it does not fit on a 40 GB GPU, but does fit on an 80 GB one.

**Results differ slightly between runs, or between GPUs**
Both `ml_model.py` and `inverse_design.py` set `--xla_gpu_deterministic_ops=true` via `XLA_FLAGS`, and every random draw (model init, data split/shuffle, jitter noise) is derived from a fixed `SEED` through `jax.random.fold_in`. This gives exact, bit-for-bit reproducibility **on the same GPU architecture and software stack** (same CUDA/cuDNN/XLA/JAX versions). It does **not** guarantee bit-identical results across genuinely different hardware or library versions — floating-point rounding order can still differ, and small per-step differences can compound over a long optimisation.

**Notebook `ModuleNotFoundError` for `data_generate`/`ml_model`/`inverse_design`**
The project root must be on `sys.path`. `sample_workflow.ipynb` handles this automatically in its first cell (it detects the project root whether launched from the repo root or from inside `notebooks/`) — if you copy code out of the notebook elsewhere, make sure to add the project root to `sys.path` first.

**`[WinError 206] The filename or extension is too long`** (Windows only)
Installing `flax` pulls in `orbax-checkpoint`, which can exceed Windows' 260-character path limit. Enable Windows Long Path support (`gpedit.msc` → System → Filesystem → "Enable Win32 long paths"), or use WSL2 / native Linux.

**`WARNING:absl:Tensorflow library not found` / `WARNING:absl:The transformations API will eventually be replaced...`**
Both harmless — Flax/Orbax checkpoint-backend warnings unrelated to correctness in this workflow.

</details>

---

## Citation

No associated paper is published yet. Once available, cite both the paper and this repository. In the meantime, please cite the repository:

```text
Author(s). "PolySurrogate: A JAX/Flax surrogate-model pipeline for photonic
quantum-state inverse design." GitHub repository,
https://github.com/AI-Quantum-Lab/PolySurrogate, <year>.
```

```bibtex
@misc{polysurrogate,
  author       = {TBD},
  title        = {PolySurrogate: A JAX/Flax surrogate-model pipeline for photonic quantum-state inverse design},
  year         = {TBD},
  howpublished = {\url{https://github.com/AI-Quantum-Lab/PolySurrogate}},
  note         = {Associated paper: TBD}
}
```

Licensed under the MIT License — see [LICENSE](LICENSE).
