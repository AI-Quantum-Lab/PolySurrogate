# Inverse-design settings — PNN at n4 / n6 (n8 pending)

Exact `inverse_design.py` parameters used to produce the production-scale
inverse-design runs under `results/inverse_design/{GHZ,W,LINEAR_CLUSTER}_n{4,6}/`,
pulled directly from each run's own `cfg.json` / `reproducibility_manifest.json`.

**Every setting below is identical across all three target states (GHZ, W,
LINEAR_CLUSTER) and across system sizes (n=4, n=6, and the planned n=8
run)** — only the node-count-dependent fields (`NPHOTONS`/`ARCHITECTURE`/
`MODEL_PATH`) and the target-dependent fields (`TARGET_NAME`/
`CONDITIONED_DATA_PATH`) change between runs. This was verified directly by
diffing all six existing runs' `cfg.json` files: they differ in exactly
those fields and nowhere else (aside from `deterministic_xla`, see the note
below).

To reproduce: edit `NPHOTONS`, `TARGET_NAME`, `ARCHITECTURE`, `MODEL_PATH`,
`CONDITIONED_DATA_PATH` at the top of `inverse_design.py` to the values
below for the target (state, n) pair, leave every other constant as listed,
then run `python inverse_design.py`.

## Shared settings (identical across all runs, all targets, all system sizes)

| Setting | Value |
|---|---|
| `MODEL_TYPE` | PNN |
| `DIMENSIONS` | 2 |
| `GENERATE_DATA` | False (starting graphs loaded from `CONDITIONED_DATA_PATH`) |
| `DATA_SAMPLES` | 1,000 |
| `DATA_BATCH_SIZE` | 5 |
| `DATA_SEED` | 4 |
| `LOW_FIDELITY_THRESHOLD` | 0.999 |
| `NORMED_DATA` | False |
| `NORMALIZE_MODEL_OUTPUT` | False |
| `LAMBDA_L1` | 1e-3 |
| `SEED` | 46 |
| `NUM_STEPS` | 100,000 |
| `MAX_TOTAL_STEPS` | 100,000 |
| `EARLY_STOP_NN_FID` | 0.99999 |
| `LEARNING_RATE` | 1e-2 |
| `MIN_LEARNING_RATE` | 1e-6 |
| `LR_DECAY_STEPS` | 100,000 |
| `LR_EXPONENT` | 1.0 |
| `CLIP_MIN` / `CLIP_MAX` | -1.0 / 1.0 |
| `JITTER_ENABLED` | True |
| `INITIAL_JITTER` | 0.01 |
| `JITTER_SCHEDULE` | [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.00, 1.00, 1.00, 1.00] |
| `JITTER_MAX_EVENTS` | 100,000 |
| `STUCK_PATIENCE_STEPS` | 1,000 |
| `STUCK_MIN_IMPROVEMENT` | 1e-4 |
| `PRUNE_FID_TOLERANCE` | 1e-6 |
| `PRUNE_THRESHOLDS` | [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] |

## Per-system-size settings

| Setting | n4 | n6 | n8 |
|---|---|---|---|
| `NPHOTONS` | 4 | 6 | 8 *(pending — see note below)* |
| `ARCHITECTURE` (must match the trained PNN's `HIDDEN_DIM`) | 400 | 2000 | 15000 |
| `input_dim` | 24 | 60 | 112 |
| `out_dim` | 16 | 64 | 256 |
| `MODEL_PATH` | `results/model_training/PNN/n4/n4_5/best_params.msgpack` | `results/model_training/PNN/n6/n6_0/best_params.msgpack` | `results/model_training/PNN/n8/n8_1/best_params.msgpack` *(training still in progress)* |

## Per-target setting

Only `TARGET_NAME` and `CONDITIONED_DATA_PATH` change between the three
target states, following the same pattern for every system size:

| `TARGET_NAME` | `CONDITIONED_DATA_PATH` |
|---|---|
| `"GHZ"` | `paper_data/inverse_init_fin_data/n{NPHOTONS}/GHZ/paper_like_initial_fidelity_dataset.npz` |
| `"W"` | `paper_data/inverse_init_fin_data/n{NPHOTONS}/W/paper_like_initial_fidelity_dataset.npz` |
| `"LINEAR_CLUSTER"` | `paper_data/inverse_init_fin_data/n{NPHOTONS}/LINEAR_CLUSTER/paper_like_initial_fidelity_dataset.npz` |

(The six completed runs' own `cfg.json` files record an equivalent local
path outside this repository, `paper_like_initial_fidelity_all_targets/n{N}/{TARGET}/...` —
verified byte-identical, via `md5sum`, to the tracked copy at the
`paper_data/inverse_init_fin_data/` path shown above. Use the `paper_data/`
path — it is the one actually shipped with the repository.)

## Per-run provenance

| | GHZ/W/LINEAR_CLUSTER n4 | GHZ/W/LINEAR_CLUSTER n6 |
|---|---|---|
| Git commit | `e00f03f` (dirty) | `e00f03f` (dirty) |
| Trained-model sha256 (fast) | `8a15a1535323077fa828328f105b5ca7ceb3cf39f6d2ed0a065e751eb576e2da` | `8be7cdf7f1d8484b665998d804711e3556479a288cd0b614046fb4b944c05ca5` |
| `deterministic_xla` recorded in `cfg.json` | not present (see note) | `true` |

`dirty=true` for both means the working tree had uncommitted local changes
at generation time — in the n6 case, specifically the `DETERMINISTIC_XLA`
fix added to `inverse_design.py` (see next note), not yet committed at the
time these runs were made.

### Note on determinism

The n4 runs were made **before** `inverse_design.py` gained its
`DETERMINISTIC_XLA = True` fix (which sets `--xla_gpu_deterministic_ops=true`
via `XLA_FLAGS`) — their `cfg.json` has no `deterministic_xla` key at all.
The n6 runs were made **after** that fix, hence `deterministic_xla: true`.
This means the n4 and n6 production inverse-design runs are not directly
comparable for exact reproducibility purposes, even though every other
listed setting is identical. See `README.md` → Troubleshooting for what
`DETERMINISTIC_XLA` does and does not guarantee.

### Note on n8

Not yet run at the time of writing. `ml_model.py`'s PNN n8 training
(`results/model_training/PNN/n8/n8_1/`, `HIDDEN_DIM=15000`) is still in
progress (currently past epoch 2,470 of a 20,000-epoch cap, still
improving) — `MODEL_PATH` above will need updating once that run finishes
or is otherwise finalised. All shared settings and the per-target
`CONDITIONED_DATA_PATH` pattern above already apply to n8 unchanged; only
`NPHOTONS=8`, `ARCHITECTURE=15000`, `input_dim=112`, `out_dim=256`, and the
final `MODEL_PATH` need to be set once training completes.

## Environment

Identical across all six completed runs:

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

Source: `results/inverse_design/{GHZ,W,LINEAR_CLUSTER}_n{4,6}/..._0/cfg.json`
and `reproducibility_manifest.json`.
