# Project Workflow

This document explains what the project does, how the code is organised, and how to run the three-stage pipeline without opening every Python file.

---

## What the project does

The project learns a **PNN (Polynomial Neural Network) surrogate model** that maps photonic graph weights to unnormalised quantum-state amplitude vectors.  Once trained, the surrogate is used inside an **inverse-design optimiser** that searches for graph configurations that produce a target quantum state (GHZ, W, linear-cluster, …).

```
random graph weights  ──►  [PyTheus perfect-matching]  ──►  unnormalised amplitude vector
                                                                   │
                                   ┌───────────────────────────────┘
                                   ▼
                         [Surrogate model (PNN / FNN)]
                           learns weights ──► unnormalised amps map
                                   │
                                   ▼
                         [Inverse optimiser]
                           gradient-descends graph weights
                           toward a target amplitude vector,
                           verified and pruned with PyTheus
```

---

## Repository layout

```
surrogate_model_clean/
├── configs/                    # All user-facing settings live here
│   ├── data_config.py          # Data-generation parameters
│   ├── training_config.py      # Model-training parameters
│   └── optimiser_config.py     # Inverse-design parameters
├── src/                        # Core library code
│   ├── data_generation.py      # Entry point: generate_dataset()
│   ├── data_generation_utils.py
│   ├── model_training.py       # Entry point: train_surrogate_model()
│   ├── model_training_utils.py
│   ├── models.py               # FNN and PNN architectures
│   ├── optimiser.py            # Entry point: run_optimisation()
│   ├── optimisation_utils.py
│   └── target_states.py        # GHZ, W, cluster, … state definitions
├── notebooks/
│   └── sample_workflow.ipynb   # Beginner-friendly interactive demo
├── scripts/
│   └── run_quick_test.py       # One-command pipeline test (~10 s on CPU)
├── requirements.txt
├── .gitignore
├── README.md
└── WORKFLOW.md                 # ← this file
```

### Files a user needs to touch

| File | When to edit |
|------|-------------|
| `configs/data_config.py` | Change node count, sample size, output folder |
| `configs/training_config.py` | Change model size, epochs, data path, output folder |
| `configs/optimiser_config.py` | Change target state, node count, model path, output folder |

### Files a user should not need to edit

`src/` files contain the implementation.  They can be read for reference but do not need to be modified for a standard run.

---

## The three pipeline stages

### Stage 1 — Data generation

**Script:** `src/data_generation.py`  
**Config:** `configs/data_config.py`  
**Entry point:** `generate_dataset()`

What it does:
1. Calls PyTheus to enumerate all perfect matchings for an `n`-node, dimension-2 graph.
2. Generates random dense graph weight vectors in `[-1, 1]`.
3. Computes the resulting **unnormalised** quantum amplitude vector for each weight sample (using JAX, GPU-accelerated when available).
4. Saves the data as `.npz` shard files.

Key parameters in `data_config.py`:

| Parameter | Meaning |
|-----------|---------|
| `VERTICES` | Node / photon count. Must match training and optimisation. |
| `N_SAMPLES` | Total samples to generate. |
| `OUT_DIR` | Where shards are written (relative to project root). |
| `NORMED_DATA` | `False` (default) → store raw unnormalised amplitude vectors. `True` → L2-normalise before saving. Must match `normed_data` in optimiser_config.py. |

After generation, merge shards into a single file:

```bash
# PYTHONPATH must already include src/ — see README Installation section
python -c "
from pathlib import Path
from data_generation_utils import merge_shards_to_npz
merge_shards_to_npz(Path('data/smoke_test'), 'dataset_merged.npz')
"
```

---

### Stage 2 — Model training

**Script:** `src/model_training.py`  
**Config:** `configs/training_config.py`  
**Entry point:** `train_surrogate_model()`

What it does:
1. Loads the merged dataset.
2. Constructs a `PNN` (Polynomial Neural Network) or `FNN` model.
3. Trains with AdamW + cosine LR decay using MAE loss against the unnormalised amplitude vectors.
4. Saves the best validation checkpoint as `params.msgpack`.

Key parameters in `training_config.py`:

| Parameter | Meaning |
|-----------|---------|
| `NODES` | Node count. Must match the dataset. |
| `MODEL_NAME` | `"PNN"` (Polynomial Neural Network, single hidden layer) or `"FNN"` (multi-layer). |
| `HIDDEN_DIM` | Integer for PNN; tuple for FNN, e.g. `(2000, 2000, 2000)`. |
| `DATA_PATH` | Path to `dataset_merged.npz`. Use a relative path from the project root. |
| `NORMALIZE_MODEL_OUTPUT` | Whether to L2-normalise model output *inside the loss function*. **Independent of `NORMED_DATA`** — see normalisation note below. |
| `NUM_EPOCHS`, `PATIENCE` | Training budget and early-stop patience. |
| `ROOT_FOLDER`, `RUN_NAME` | Outputs go in `<ROOT_FOLDER>/run_<RUN_NAME>/`. |

The `params.msgpack` produced here is the input to Stage 3.

---

### Stage 3 — Inverse-design optimisation

**Script:** `src/optimiser.py`  
**Config:** `configs/optimiser_config.py`  
**Entry point:** `run_optimisation()`

What it does:
1. Loads or generates starting graph-weight vectors.
2. Loads the trained surrogate from `params.msgpack`.
3. Runs gradient descent on each starting vector to minimise `loss = (1 − fidelity) + λ·‖x‖₁`.
4. At each step, computes the true PyTheus amplitude vector and logs fidelity to the target state.
5. After optimisation, progressively zeros small weights (pruning) and saves the sparsest graph that maintains fidelity within tolerance.

Key parameters in `optimiser_config.py`:

| Parameter | Meaning |
|-----------|---------|
| `NPHOTONS` | Node count. Must match the trained model. |
| `TARGET_NAME` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"`, `"SINGLE"`, or `"ZERO"`. |
| `generate_data` | `True` → generate fresh random starts; `False` → load from `conditioned_data_path`. |
| `architecture` | Must match `HIDDEN_DIM` from training (integer for PNN, tuple for FNN). |
| `model_path` | Path to `params.msgpack` from Stage 2. |
| `normalize_model_output` | Must match `NORMALIZE_MODEL_OUTPUT` from training. |
| `num_steps` | Gradient-descent budget per sample. |
| `results_root`, `folder_name` | Outputs go in `<results_root>/<folder_name>/`. |

---

## Normalisation note

There are **two independent normalisation flags** that users sometimes confuse:

| Flag | Location | What it controls |
|------|----------|-----------------|
| `NORMED_DATA` | `data_config.py` | Whether amplitude vectors are L2-normalised **in the dataset file** before saving. |
| `normed_data` | `optimiser_config.py` | Whether starting samples are generated with normalised amplitudes. Must match `NORMED_DATA`. |
| `NORMALIZE_MODEL_OUTPUT` | `training_config.py` | Whether model output is L2-normalised **inside the loss function** during training. |
| `normalize_model_output` | `optimiser_config.py` | Whether model output is L2-normalised inside fidelity evaluation. Must match `NORMALIZE_MODEL_OUTPUT`. |

**Consistency requirements:**
- `NORMED_DATA` (data) **must** match `normed_data` (optimiser).
- `NORMALIZE_MODEL_OUTPUT` (training) **must** match `normalize_model_output` (optimiser).
- These two pairs are **independent** — they do not need to have the same value as each other.

**Default project settings:** `NORMED_DATA=False`, `NORMALIZE_MODEL_OUTPUT=False`.  
The model is trained to predict raw unnormalised amplitude vectors.

---

## Running the full pipeline

### Minimal quick test (no edits needed)

```bash
cd surrogate_model_clean
source .venv/bin/activate
PYTHONPATH="$PWD/src:$PWD/configs" python scripts/run_quick_test.py
```

Completes in ~10 seconds on CPU.  All outputs go to `data/quick_test/`, `Models_quick_test/`, and `optimiser_quick_test/`.

### Standard smoke test using config files

```bash
# Set PYTHONPATH first (required for all commands below)
export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"

# 1. Generate data (uses data_config.py defaults: 4 nodes, 1000 samples)
python src/data_generation.py

# 2. Merge shards
python -c "from pathlib import Path; from data_generation_utils import merge_shards_to_npz; merge_shards_to_npz(Path('data/smoke_test'), 'dataset_merged.npz')"

# 3. Train (uses training_config.py defaults: 4 nodes, 20 epochs)
python src/model_training.py

# 4. Optimise (uses optimiser_config.py defaults: GHZ, 4 nodes, 20 steps)
python src/optimiser.py
```

### Production / HPC run

1. Edit `configs/data_config.py`: set `VERTICES=10`, `N_SAMPLES=5_000_000`, `OUT_DIR='data/node10'`.
2. Generate and merge data.
3. Edit `configs/training_config.py`: uncomment the production block at the bottom.
4. Run training (typically on a GPU node).
5. Edit `configs/optimiser_config.py`: update `NPHOTONS`, `MODEL_TYPE`, `architecture`, `model_path`, uncomment production block.
6. Run optimisation.

---

## Data flow between stages

```
data_config.py
    ↓
data_generation.py  →  data/<OUT_DIR>/data_*.npz
                               ↓  (merge)
                        data/<OUT_DIR>/dataset_merged.npz
                               ↓
training_config.py            DATA_PATH
    ↓
model_training.py   →  <ROOT_FOLDER>/run_<RUN_NAME>/params.msgpack
                               ↓
optimiser_config.py           model_path
    ↓
optimiser.py        →  <results_root>/<folder_name>/
```

---

## Files not tracked by Git

The `.gitignore` excludes all generated outputs.  These are created locally and must **not** be committed:

| Pattern | What it covers |
|---------|---------------|
| `data/` | All generated datasets |
| `*.npz` | Individual shard and merged files |
| `Models*/` | Trained model directories |
| `*.msgpack` | Serialised model parameters |
| `checkpoints/` | Orbax checkpoint directories |
| `optimiser_*/`, `results/` | Optimisation output directories |
| `*.log`, `*.out`, `*.err` | Log files |
| `.venv/`, `__pycache__/` | Environment and bytecode |

---

## Assumptions the code makes

- **Working directory:** all scripts must be run from the project root (`surrogate_model_clean/`), and `PYTHONPATH` must include `src/` and `configs/`.
- **Node-count consistency:** `VERTICES` (data) = `NODES` (training) = `n` / `NPHOTONS` (optimisation). Mismatches cause shape errors.
- **Normalisation consistency:** see the Normalisation note section above. Two independent pairs must match internally; they need not match each other.
- **Architecture consistency:** `HIDDEN_DIM` (training) must equal `architecture` (optimisation), same type (int for PNN, tuple for FNN).
- **JAX device:** code runs on CPU automatically if no GPU is found.  GPU use is transparent — no code changes needed.
- **PyTheus import:** install `pytheusQ` (not `pytheus`) from PyPI.
