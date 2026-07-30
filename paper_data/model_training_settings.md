# Model training settings — PNN/FNN at n4 / n6 / n8

Exact `ml_model.py` parameters for the runs where the learning rate decays
from `1e-3` to `1e-6` over 2500 epochs, pulled directly from each run's own
`config.json` / `reproducibility_manifest.json`.

| Run | Path |
|---|---|
| PNN n4 | `results/model_training/PNN/n4/n4_5/` |
| PNN n6 | `results/model_training/PNN/n6/n6_0/` |
| PNN n8 | `results/model_training/PNN/n8/n8_1/` |
| FNN n4 | `results/model_training/FNN/n4/n4_1/` |
| FNN n6 | `results/model_training/FNN/n6/n6_0/` |
| FNN n8 | `results/model_training/FNN/n8/` — same shared settings below, `HIDDEN_DIM = (8000, 8000, 8000)` |

To reproduce: edit the listed constants at the top of `ml_model.py` to the
values below for the target (model, n) pair, leave every other constant as
listed, then run `python ml_model.py`.

## Shared settings (identical across all six runs)

| Setting | Value |
|---|---|
| `LEARNING_RATE` | 1e-3 |
| `FINAL_LEARNING_RATE` | 1e-6 |
| `LR_DECAY_UNTIL_EPOCH` | 2500 |
| `NUM_EPOCHS` | 20,000 (ceiling — early stop governs actual length) |
| `PATIENCE` | 2,000 |
| `TOLERANCE` | 1e-7 |
| `WEIGHT_DECAY` | 1e-4 |
| `BATCH_SIZE` | 5,000 |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | 0.8 / 0.1 / 0.1 |
| `SPLIT_MODE` | contiguous |
| `SEED` | 159 |
| `PRECISION` | float64 |
| `NORMALIZE_MODEL_OUTPUT` | False |
| `DATA_SIZE` | `null` (full dataset, no subsampling) |
| `DATASET_HASH_MODE` | fast |
| `DETERMINISTIC_XLA` | True |
| `CHECKPOINT_EVERY` | 1 |
| `KEEP_LATEST_CHECKPOINTS` | 1 |

## Per-run settings

| Setting | PNN n4 | PNN n6 | PNN n8 | FNN n4 | FNN n6 | FNN n8 |
|---|---|---|---|---|---|---|
| `NODES` | 4 | 6 | 8 | 4 | 6 | 8 |
| `MODEL_NAME` | PNN | PNN | PNN | FNN | FNN | FNN |
| `HIDDEN_DIM` | 400 | 2000 | 15000 | (400, 400, 400) | (2000, 2000, 2000) | (8000, 8000, 8000) |
| `DATA_PATH` | `results/data_generation/n4/n4_0/dataset_merged.npz` | `results/data_generation/n6/n6_0/dataset_merged.npz` | `results/data_generation/n8/n8_0/dataset_merged.npz` | `results/data_generation/n4/n4_0/dataset_merged.npz` | `results/data_generation/n6/n6_0/dataset_merged.npz` | `results/data_generation/n8/n8_0/dataset_merged.npz` |

## Per-run provenance

| | PNN n4 | PNN n6 | PNN n8 | FNN n4 | FNN n6 | FNN n8 |
|---|---|---|---|---|---|---|
| Timestamp | 2026-07-30T00:21:13+0200 | 2026-07-29T20:32:13+0200 | 2026-07-30T14:19:33+0200 | 2026-07-30T12:59:44+0200 | 2026-07-30T11:46:30+0200 | *(fill in once run)* |
| Git commit | `571b482` (dirty) | `571b482` (dirty) | `571b482` (dirty) | `571b482` (dirty) | `571b482` (dirty) | *(fill in once run)* |
| Train / val / test size | 16.0M / 2.0M / 2.0M | 16.0M / 2.0M / 2.0M | 16.0M / 2.0M / 2.0M | 16.0M / 2.0M / 2.0M | 16.0M / 2.0M / 2.0M | 16.0M / 2.0M / 2.0M |
| Input dataset sha256 (fast) | `394a69de74d1ec8c651ac78ce99b98edd4f8a7e0d2bba06e3b12a72a4302c717` | `900f3c597312d0ee50b9c3b70a3024400f3b2364f8cdd46c77eee3b11b63451a` | `010dd487a9d76718091d128b7e4bd2ab75efc50e140af5549e179f9c0ed2a3b1` | `394a69de74d1ec8c651ac78ce99b98edd4f8a7e0d2bba06e3b12a72a4302c717` | `900f3c597312d0ee50b9c3b70a3024400f3b2364f8cdd46c77eee3b11b63451a` | `010dd487a9d76718091d128b7e4bd2ab75efc50e140af5549e179f9c0ed2a3b1` |

All six runs share the same reproducibility keys (`master_key`,
`model_initialisation_key`, `data_split_key`, `epoch_shuffle_root`,
`split_permutation_hash`) since they all use `SEED=159`.

### Note on PNN n8

This run hit its 12h SLURM time limit (`gpu-test` partition) at epoch
1007/20000 — not a crash — and was resumed from checkpoint with the exact
same config (`resume_n8_1.py`) so it continued reproducibly from the same
seeded RNG state. It requires an 80GB A100 for the full 20M-sample n8
dataset (submit with `--constraint=a100_80gb` if your cluster supports it).

## Environment

Identical across all six runs (note `jax_x64_enabled=True` here, vs
`False` for the Stage-1 data-generation runs — this pipeline trains in
float64):

| Package | Version |
|---|---|
| Python | 3.12.7 (Anaconda, GCC 11.2.0) |
| Platform | Linux-4.18.0-425.13.1.el8_7.x86_64 |
| JAX / jaxlib | 0.8.2 |
| Flax | 0.12.2 |
| Optax | 0.2.6 |
| pytheusQ | 1.2.6 |
| NumPy | 2.4.2 |
| JAX backend | GPU (`cuda:0`) |
| `jax_x64_enabled` | True |
| XLA flags | `--xla_gpu_deterministic_ops=true` |

Source: `results/model_training/{PNN,FNN}/n{4,6,8}/.../config.json` and
`reproducibility_manifest.json`.
