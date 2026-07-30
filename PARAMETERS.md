# Parameter reference

Comprehensive, per-stage reference for every tunable parameter in the pipeline. Each of the three numbered scripts keeps its parameters as top-of-file constants — no external config file. `notebooks/sample_workflow.ipynb` only exposes a minimal subset (system size, seeds, sample/epoch/step counts); everything else uses the defaults documented here. Edit the actual script constants, or override any key via the `cfg` dict argument to `generate_dataset()` / `train_surrogate_model()` / `run_optimisation()`, if you need to change something not exposed in the notebook.

## Table of contents

- [Stage 1 — `01_data_generate.py`](#stage-1--01_data_generatepy)
- [Stage 2 — `02_ml_model.py`](#stage-2--02_ml_modelpy)
- [Stage 3 — `inverse_design.py`](#stage-3--inverse_designpy)
- [Cross-stage consistency requirements](#cross-stage-consistency-requirements)

---

## Stage 1 — `01_data_generate.py`

Generates random dense graph weights and the corresponding quantum amplitude vectors, via a PyTheus perfect-matching catalogue.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `VERTICES` | `4` | Graph nodes / photons / modes. Determines `num_edges = 2 * VERTICES * (VERTICES - 1)` and `n_kets = 2 ** VERTICES`. Must match Stage 2's `NODES` and Stage 3's `NPHOTONS`. |
| `DIMENSIONS` | `2` | Local Hilbert-space dimension per mode. `2` = qubit-like (photon present/absent per path). Higher values model qudits. |
| `N_SAMPLES` | `1000` | Total number of random graph samples to generate. |
| `BATCH_SIZE` | `100` | Samples per JAX batch during generation (a performance knob — doesn't affect the generated values themselves; verified empirically that changing it doesn't change the resulting `weights`). |
| `SHARD_SIZE` | `1000` | Max samples per intermediate `.npz` shard file, before merging into one `dataset_merged.npz`. |
| `SEED` | `0` | Seed for `np.random.default_rng(seed)`, which draws all edge weights uniformly from `[-1, 1]`. Bit-identical across machines given the same seed and matching `numpy`/`pytheus` versions — see the reproducibility note at the bottom of the file. |
| `SAVE_DATA` | `True` | If `True`, writes shards + merged `.npz` + `metadata.json` + `reproducibility_manifest.json` to an auto-numbered folder. If `False`, nothing is written to disk — only the in-memory arrays are returned. |
| `DATA_ROOT` | `"results/data_generation"` | Parent folder for output. The actual output folder is `{DATA_ROOT}/n{VERTICES}/n{VERTICES}_{i}`, where `{i}` is auto-computed (next free integer) — never chosen by hand, and never collides with a previous run. |
| `NORMED_DATA` | `False` | `False` (default) → store raw **unnormalised** amplitude vectors — what the surrogate is trained to predict. `True` → L2-normalise (`‖ψ‖₂ = 1`) before saving. Must match `NORMED_DATA` used in Stage 3's starting-sample generation. |

**Return contract:** `generate_dataset(...)` always returns `(weights, amps, out_dir)`. The arrays are always populated in memory regardless of `SAVE_DATA`; `out_dir` is the auto-numbered `Path` when `SAVE_DATA=True`, or `None` when `SAVE_DATA=False`.

**Mechanics** (see [utils.py](utils.py)'s `build_pytheus_catalog`, `generate_random_dense_weights`, `compute_amplitudes`/`compute_amplitudes_no_norms`):
1. `build_pytheus_catalog(vertices, dimensions)` — PyTheus enumerates every graph edge and every perfect-matching path contributing to each output basis state (ket); packed into dense `tensor`/`mask` arrays. Deterministic, no randomness.
2. `generate_random_dense_weights(rng, n_samples, num_edges)` — draws the actual random edge weights, uniform in `[-1, 1]`.
3. `compute_amplitudes` (or `_no_norms`) — for each sample, multiplies weights along each matching, sums across matchings per ket, optionally L2-normalises.

**Reproducibility caveat:** `weights` are bit-identical on any machine given the same seed and matching `numpy`/`pytheus` versions, but this is **not independent of batch/subset structure** — sample *i*'s draw depends on how many samples were drawn before it in the same RNG stream (sequential consumption, not per-sample fold_in). `amps` (computed via JAX) can differ by ~1e-5 across different hardware/XLA backends, since floating-point reduction isn't associative.

---

## Stage 2 — `02_ml_model.py`

Trains a PNN or FNN surrogate to map graph weights → unnormalised amplitude vectors.

<details>
<summary><b>Bootstrap (before Parameters)</b></summary>

| Constant | Default | Meaning |
|----------|---------|---------|
| `DETERMINISTIC_XLA` | `True` | Sets `XLA_FLAGS=--xla_gpu_deterministic_ops=true` in-process, requesting deterministic GPU reduction order. Best-effort only — must run before the first `import jax` (including transitively via `import utils`), which is why this bootstrap stays at the very top of the file rather than in `utils.py`. Fully reliable only if `XLA_FLAGS` is exported in the shell/sbatch script before Python launches at all. |

</details>

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `NODES` | `4` | Node count. Must match the dataset (`VERTICES` from Stage 1) and Stage 3's `NPHOTONS`. |
| `DIMENSIONS` | `2` | Must match Stage 1. |
| `DATE` | `"smoke_test"` | Informational label stored in the resolved config; not used for any path or logic. |
| `MODEL_NAME` | `"PNN"` | `"PNN"` (Polynomial Neural Network — one hidden layer, monomial activation `x^(n/2)`) or `"FNN"` (Feedforward NN — multi-layer, GELU activations). |
| `HIDDEN_DIM` | `400` | PNN: integer hidden dimension. FNN: tuple, e.g. `(2000, 2000, 2000)`. Must match Stage 3's `ARCHITECTURE`. |
| `DATA_PATH` | `"results/data_generation/n4/n4_0/dataset_merged.npz"` | Path to the merged dataset from Stage 1. Edit to point at whichever `n{v}_{i}` run to train on. |
| `DATA_SIZE` | `None` | Cap the number of samples used; `None` uses the full dataset. |
| `NORMALIZE_MODEL_OUTPUT` | `False` | Whether model output is L2-normalised **inside the loss function** before computing MAE. Independent of `NORMED_DATA` (see [Cross-stage consistency](#cross-stage-consistency-requirements)). Must match Stage 3's `NORMALIZE_MODEL_OUTPUT`. |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | `0.8` / `0.1` / `0.1` | Dataset split ratios; must sum to `1.0`. |
| `LEARNING_RATE` | `1e-3` | Initial Adam learning rate. |
| `LR_AFTER_DECAY` | `1e-5` | Learning rate at the end of the cosine-decay schedule. |
| `LR_DECAY_UNTIL_EPOCH` | `20` | Cosine-decay schedule length, in epochs. |
| `BATCH_SIZE` | `100` | Training mini-batch size. |
| `NUM_EPOCHS` | `20` | Training epochs. |
| `PATIENCE` | `10` | Early-stop patience, in epochs with no validation-loss improvement. |
| `TOLERANCE` | `1e-7` | Minimum val-loss improvement to reset the patience counter. |
| `SEED` | `159` | Master seed. Every PRNG key (model init, data split, epoch shuffling, and a reserved slot for dropout/future stochastic layers) is independently derived from this one value via `fold_in` (`utils.derive_root_keys`) — never sequential `split()` — so a resumed run reproduces exactly the future key sequence an uninterrupted run would have used. |
| `SPLIT_MODE` | `"contiguous"` | `"contiguous"` — first `TRAIN_SPLIT` fraction → train, next `VAL_SPLIT` → val, last `TEST_SPLIT` → test; no randomness. `"shuffled"` — indices permuted via a seed-derived key before splitting. |
| `PRECISION` | `"float32"` | `"float32"` (default) or `"float64"` (`jax_enable_x64`, slower / more memory). |
| `DATASET_HASH_MODE` | `"fast"` | Controls how `DATA_PATH` is fingerprinted in the manifest via `utils.hash_file`. `"fast"` — size + mtime + hash of first/last 1MB (cheap, catches accidental file swaps). `"full"` — SHA-256 of the entire file (strong, slow on multi-GB files). `"none"` — skip hashing. |
| `DEBUG_TRACE` | `False` | If `True`, writes `<run_dir>/trace.jsonl` with one record per epoch (batch-index hash, LR, losses, params/opt_state hashes, PRNG key hash). |
| `RESUME_FULL_STATE` | `False` | Set `True` to resume from `CKPT_DIR_RESTORE`. |
| `CKPT_DIR_RESTORE` | `"results/model_training/PNN/n4/n4_0/checkpoints"` | Path to an existing `checkpoints/` folder to resume from. |
| `ROOT_FOLDER` | `"results/model_training"` | Output root. The actual output folder is `{ROOT_FOLDER}/{MODEL_NAME}/n{NODES}/n{NODES}_{i}`, auto-numbered — never chosen by hand, never collides. |
| `LOSS_NAME` | `"mae"` | Informational label used in plot titles / `model_info.json`. |

**Return value:** `train_surrogate_model(cfg)` returns the auto-numbered run directory (a `Path`).

**No collision handling needed:** because the output folder is always freshly auto-numbered, rerunning the script never reuses a folder, so there is no `orbax` "destination already exists" risk from simply running the script again with the same settings (only relevant if you manually point `CKPT_DIR_RESTORE` at the wrong place while `RESUME_FULL_STATE=True`).

**Reproducibility manifest** (`reproducibility_manifest.json`, written to the run folder) contains: the full resolved config (including `SEED`), hashes of every derived PRNG key, git commit/dirty flag/branch, environment versions (JAX/jaxlib/flax/optax/pytheus/numpy, devices, backend, x64 mode), the train/val/test split info, and — critically — a `data_generation` block copied from Stage 1's own manifest (found next to `DATA_PATH`), so this manifest alone documents both how the model was trained *and* how to regenerate the data it was trained on.

---

## Stage 3 — `inverse_design.py`

Uses the trained surrogate as a differentiable proxy for PyTheus: gradient descent on graph weights toward a target quantum state, verified against the exact PyTheus simulator, then pruned to a sparse graph.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `NPHOTONS` | `4` | Node count. Must match the trained model. |
| `TARGET_NAME` | `"GHZ"` | `"GHZ"`, `"W"`, `"LINEAR_CLUSTER"` (alias `"CLUSTER"`/`"LINEAR"`), `"SINGLE"`, or `"ZERO"` — see `get_target_state()` for the exact state definitions. |
| `MODEL_TYPE` | `"PNN"` | Must match the trained model (`"PNN"` or `"FNN"`). |
| `GENERATE_DATA` | `True` | `True` → generate fresh starting graphs on the fly, via Stage 1's own `generate_dataset(save_data=False)` (nothing written to disk, no entry added to Stage 1's dataset registry). `False` → load starting samples from `CONDITIONED_DATA_PATH` instead. |
| `CONDITIONED_DATA_PATH` | `"results/data_generation_conditioned/conditioned_dataset.npz"` | Used only when `GENERATE_DATA=False`. Must be an `.npz` with `weights`/`amps` keys. |
| `MAX_INITIAL_SAMPLES` | `None` | When loading from `CONDITIONED_DATA_PATH`, optionally cap how many samples are read. |
| `DATA_SAMPLES` | `5` | How many starting samples to optimise (when `GENERATE_DATA=True`). |
| `DATA_BATCH_SIZE` | `5` | JAX batch size while generating starting samples. |
| `DATA_SEED` | `4` | Seed for starting-sample generation. Each "round" (if more samples are needed to pass the fidelity filter) uses `DATA_SEED + round_id - 1`. |
| `LOW_FIDELITY_THRESHOLD` | `0.999` | Freshly generated samples already this close to the target are discarded and replaced — ensures the optimiser always starts from a genuinely hard case. |
| `NORMED_DATA` | `False` | Must match Stage 1's `NORMED_DATA`. |
| `ARCHITECTURE` | `400` | Must match Stage 2's `HIDDEN_DIM` (int for PNN, tuple for FNN). |
| `MODEL_PATH` | `"results/model_training/PNN/n4/n4_0/params.msgpack"` | Path to the trained model's `params.msgpack`. **Edit after training** to point at the run you want. |
| `NORMALIZE_MODEL_OUTPUT` | `False` | Must match Stage 2's `NORMALIZE_MODEL_OUTPUT`. |
| `LAMBDA_L1` | `1e-3` | L1 penalty weight in `loss = (1 - fidelity) + LAMBDA_L1 * sum(abs(weights))`. Higher values push toward sparser (fewer-edge) solutions. |
| `SEED` | `46` | Master seed. Drives all reproducible jitter noise via `utils`'s fold_in key plan — every draw is a pure function of `(SEED, sample_id, event_index)`, independent of run order or which subset a sample belongs to. |
| `NUM_STEPS` | `20` | Smoke-test default; typically increased to `10_000+` for production. |
| `EARLY_STOP_NN_FID` | `0.99999` | Surrogate-fidelity stopping threshold. Stopping condition is a **single** threshold: `nn_fid >= EARLY_STOP_NN_FID OR step >= MAX_TOTAL_STEPS` — deliberately not a multi-stage escalation. |
| `LEARNING_RATE` / `MIN_LEARNING_RATE` | `1e-2` / `1e-6` | Cosine-decay Adam schedule bounds. |
| `LR_DECAY_STEPS` | `20` | Should match `NUM_STEPS`. |
| `LR_EXPONENT` | `1.0` | Cosine-decay schedule exponent. |
| `CLIP_MIN` / `CLIP_MAX` | `-1.0` / `1.0` | Graph weights are clipped to this range after every optimisation step. |
| `JITTER_ENABLED` | `True` | Master on/off switch for stall-detection jitter. |
| `INITIAL_JITTER` | `0.01` | Sigma of a one-time perturbation applied before step 1 (0 = no-op). |
| `JITTER_SCHEDULE` | `[0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.00, 1.00, 1.00, 1.00]` | Escalating sigma used for each successive stall-triggered jitter event. |
| `JITTER_MAX_EVENTS` | `12` | Give up re-perturbing after this many jitter events per sample. |
| `STUCK_PATIENCE_STEPS` | `500` | Steps with fidelity improvement below `STUCK_MIN_IMPROVEMENT` before a jitter event fires. |
| `STUCK_MIN_IMPROVEMENT` | `1e-4` | Minimum fidelity gain over `STUCK_PATIENCE_STEPS` to *not* be considered stuck. |
| `MAX_TOTAL_STEPS` | `20` | Hard ceiling on total steps; falls back to `NUM_STEPS` if not set separately. |
| `PRINT_EVERY` | `1` | How often (in steps) to print a log line. |
| `VERIFY_EVERY` | `1` | How often (in steps) to check exact PyTheus fidelity during the loop (vs. the surrogate's own fidelity estimate). |
| `STORE_STEP_VECTORS` | `False` | If `True`, stores gradient/update vectors for every step — large JSON files for long runs; keep `False` outside debugging. |
| `PRUNE_FID_TOLERANCE` | `1e-4` | After optimisation, a pruning threshold is only kept if the resulting fidelity drop is within this tolerance. |
| `PRUNE_THRESHOLDS` | `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]` | Progressive weight-zeroing thresholds tried in order (`progressive_threshold_prune`). |
| `RESULTS_ROOT` | `"results/inverse_design"` | Output root. The actual output folder is `{RESULTS_ROOT}/{TARGET_NAME}_n{NPHOTONS}/{TARGET_NAME}_n{NPHOTONS}_{i}`, auto-numbered — never chosen by hand, never collides. |

**Return value:** `run_optimisation(cfg)` returns the auto-numbered run directory (a `Path`).

**Reproducibility manifest** (`reproducibility_manifest.json`) contains: the full resolved config/seeds, git/environment info, a hash of `MODEL_PATH`, and a `model_training` block copied from Stage 2's own manifest (found next to `MODEL_PATH`) — which itself already contains Stage 1's `data_generation` block. So **Stage 3's manifest alone contains the full three-stage recipe**: optimisation config → training config/seeds → data-generation config/seed.

---

## Cross-stage consistency requirements

Two independent flag pairs must each match internally across stages (they are *not* related to each other):

| Flag | Controls | Must match between |
|------|---------|---------------------|
| `NORMED_DATA` | Whether amplitude vectors are L2-normalised **in the dataset** before saving (Stage 1) / in freshly generated starting samples (Stage 3). | Stage 1 `NORMED_DATA` ↔ Stage 3 `NORMED_DATA` |
| `NORMALIZE_MODEL_OUTPUT` | Whether model output is L2-normalised **inside the loss / fidelity evaluation** at training (Stage 2) / optimisation time (Stage 3). | Stage 2 `NORMALIZE_MODEL_OUTPUT` ↔ Stage 3 `NORMALIZE_MODEL_OUTPUT` |

Default: both pairs are `False` everywhere — the model is trained to predict, and evaluated on, raw unnormalised amplitude vectors.

Also required to match across all three stages: node count (Stage 1 `VERTICES` = Stage 2 `NODES` = Stage 3 `NPHOTONS`), model architecture (Stage 2 `HIDDEN_DIM` = Stage 3 `ARCHITECTURE`), and model type (Stage 2 `MODEL_NAME` = Stage 3 `MODEL_TYPE`).
