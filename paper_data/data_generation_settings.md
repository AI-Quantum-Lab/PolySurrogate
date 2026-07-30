# Data generation settings — n4 / n6 / n8

Exact `data_generate.py` parameters used to produce the Stage-1 datasets
under `results/data_generation/n{4,6,8}/n{4,6,8}_0/`, pulled directly from
each run's own `reproducibility_manifest.json`. All three runs share every
setting except `VERTICES`.

To reproduce: edit `VERTICES` in `data_generate.py` to the target value
below, leave every other constant as listed, then run `python data_generate.py`.

| Setting | n4 | n6 | n8 |
|---|---|---|---|
| `VERTICES` | 4 | 6 | 8 |
| `DIMENSIONS` | 2 | 2 | 2 |
| `N_SAMPLES` | 20,000,000 | 20,000,000 | 20,000,000 |
| `BATCH_SIZE` | 5,000 | 5,000 | 5,000 |
| `SHARD_SIZE` | 500,000 | 500,000 | 500,000 |
| `SEED` | 59 | 59 | 59 |
| `NORMED_DATA` | False | False | False |
| `PRECISION` | float32 | float32 | float32 |
| `SAVE_DATA` | True | True | True |

## Per-run provenance

| | n4 | n6 | n8 |
|---|---|---|---|
| Timestamp | 2026-07-29T17:28:40+0200 | 2026-07-29T17:32:28+0200 | 2026-07-29T17:42:01+0200 |
| Git commit | `571b482` (dirty) | `571b482` (dirty) | `571b482` (dirty) |
| Output size | 3,200,000,584 bytes | 9,920,000,632 bytes | 29,440,000,632 bytes |
| `dataset_merged.npz` sha256 (fast) | `394a69de74d1ec8c651ac78ce99b98edd4f8a7e0d2bba06e3b12a72a4302c717` | `900f3c597312d0ee50b9c3b70a3024400f3b2364f8cdd46c77eee3b11b63451a` | `010dd487a9d76718091d128b7e4bd2ab75efc50e140af5549e179f9c0ed2a3b1` |

`dirty=true` means the working tree had uncommitted local changes at
generation time (in practice, the manual `VERTICES` edit for that run,
made without committing between runs) — the pipeline code itself was at
commit `571b482` for all three.

## Environment

Identical across all three runs:

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
| `jax_x64_enabled` | False |

Source: `results/data_generation/n{4,6,8}/n{4,6,8}_0/reproducibility_manifest.json`.
