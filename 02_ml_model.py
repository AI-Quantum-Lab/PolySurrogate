"""
02_ml_model.py -- Stage 2: train a PNN/FNN surrogate model.

Imports shared model definitions and reproducibility primitives from
utils.py (also used by 03_inverse_design.py); everything stage-specific
(parameters, data loading/splitting, the training loop, checkpoint I/O)
lives in this file. Edit the constants in the "Parameters" section below,
then run:

    python 02_ml_model.py

Reads DATA_PATH (the `dataset_merged.npz` written by 01_data_generate.py --
connected only through that file, not through a Python import). Writes
checkpoints, plots, `params.msgpack`, `model_info.json`, and a
`reproducibility_manifest.json` under an auto-numbered run folder:
    <ROOT_FOLDER>/<MODEL_NAME>/n<NODES>/n<NODES>_<i>/
where `i` is the next free integer for that (model, node-count) pair --
never user-typed, computed the same way 01_data_generate.py numbers its
own output folders. Stage 3 (03_inverse_design.py) reads `params.msgpack`
from whatever folder train_surrogate_model() actually returns.
"""

# =============================================================================
# Bootstrap -- MUST run before the first `import jax` in this process
# (including transitively, via `import utils`, which itself imports jax).
# XLA backend init is lazy (triggered by the first device query/op), so
# setting XLA_FLAGS here still works as long as nothing above this point has
# touched jax -- but the only fully reliable way to guarantee this across
# all launch scenarios is to export XLA_FLAGS in the shell/sbatch script
# before launching python at all. See REPRODUCIBILITY_TRAINING.md. This is
# the one piece of logic that legitimately cannot move into utils.py:
# importing utils.py would itself trigger `import jax` before this flag is set.
# =============================================================================

import json
import os

DETERMINISTIC_XLA = True  # best-effort request; see note above

if DETERMINISTIC_XLA:
    _flag = "--xla_gpu_deterministic_ops=true"
    _current = os.environ.get("XLA_FLAGS", "")
    if _flag not in _current:
        os.environ["XLA_FLAGS"] = (_current + " " + _flag).strip()

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from flax.training import train_state, checkpoints

from utils import (
    build_manifest,
    configure_precision,
    create_model,
    derive_root_keys,
    deterministic_xla_env_is_set,
    dtype_for_precision,
    epoch_key_for,
    hash_array,
    hash_file,
    save_params_msgpack,
    write_json,
    write_manifest,
)


# =============================================================================
# Parameters -- edit these directly, no external config file involved.
# =============================================================================

NODES = 4
DIMENSIONS = 2
DATE = "smoke_test"

MODEL_NAME = "PNN"      # "FNN" or "PNN"
# PNN (Polynomial Neural Network): integer hidden dimension.
# FNN (Feedforward Neural Network): tuple, e.g. (2000, 2000, 2000).
HIDDEN_DIM = 400

# Path to the merged .npz produced by 01_data_generate.py.
DATA_PATH = "results/data_generation/n4/n4_0/dataset_merged.npz"
DATA_SIZE = None        # None -> use the full dataset

# Controls whether the model output is L2-normalised *inside the loss
# function* before computing MAE. Must match `normalize_model_output` used
# in 03_inverse_design.py for correct fidelity evaluation with this model.
NORMALIZE_MODEL_OUTPUT = False

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

LEARNING_RATE = 1e-3
LR_AFTER_DECAY = 1e-5
LR_DECAY_UNTIL_EPOCH = 20    # smoke test: matches NUM_EPOCHS

BATCH_SIZE = 100
NUM_EPOCHS = 20
PATIENCE = 10
TOLERANCE = 1e-7

# SEED is the single master seed for the entire run. Every PRNG key used
# anywhere (model init, dataset split, epoch shuffling, and a reserved slot
# for dropout/other stochastic layers if any are added later) is
# independently derived from this one value via fold_in (see
# utils.derive_root_keys) -- never via sequential split(), so resuming a run
# reproduces exactly the same future key sequence an uninterrupted run
# would have used.
SEED = 159

# "contiguous" (default): first TRAIN_SPLIT fraction -> train, next VAL_SPLIT
# -> val, last TEST_SPLIT -> test. No randomness, always reproducible.
# "shuffled": indices permuted via a seed-derived key before splitting.
SPLIT_MODE = "contiguous"

# "float32" (default) or "float64" (jax_enable_x64, slower/more memory).
PRECISION = "float32"

# "fast" (default, cheap size/mtime + first/last 1MB hash), "full" (sha256
# of the entire file, slow for multi-GB datasets), or "none".
DATASET_HASH_MODE = "fast"

# If True, writes <run_dir>/trace.jsonl with one compact record per epoch
# (batch-index hash, LR, losses, params/opt_state hashes, PRNG key hash).
DEBUG_TRACE = False

# Set RESUME_FULL_STATE=True and point CKPT_DIR_RESTORE at an existing
# checkpoints/ folder to resume a previous run.
RESUME_FULL_STATE = False
CKPT_DIR_RESTORE = "results/model_training/PNN/n4/n4_0/checkpoints"

# Outputs are written to <ROOT_FOLDER>/<MODEL_NAME>/n<NODES>/n<NODES>_<i>/,
# where <i> is auto-numbered (see _next_run_dir) -- no user-typed run name.
ROOT_FOLDER = "results/model_training"
LOSS_NAME = "mae"

CHECKPOINT_SCHEMA_VERSION = 2
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

configure_precision(PRECISION)


# =============================================================================
# Run-folder / logging / plotting
# =============================================================================


def _next_run_dir(root_folder: str, model_name: str, nodes: int) -> Path:
    parent = Path(root_folder) / model_name / f"n{nodes}"
    parent.mkdir(parents=True, exist_ok=True)
    i = 0
    while (parent / f"n{nodes}_{i}").exists():
        i += 1
    return parent / f"n{nodes}_{i}"


def create_run_dirs(root_folder: str, model_name: str, nodes: int):
    run_dir = _next_run_dir(root_folder, model_name, nodes)
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = (run_dir / "checkpoints").resolve()
    plot_dir = run_dir / "plots"
    snap_dir = run_dir / "params_snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = run_dir / "run_log.txt"
    log_file_path.write_text("Training about to start!\n")
    return run_dir, ckpt_dir, plot_dir, snap_dir, log_file_path


def log_file(path: Path, txt):
    if not isinstance(txt, str):
        txt = np.array2string(np.asarray(txt))
    with path.open("a") as f:
        f.write(txt + "\n")


def save_training_plot(epochs, train_losses, val_losses, plot_path, loss_name, title):
    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel(f"Loss ({loss_name})")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_test_plot(fid_list, plot_path, title):
    plt.figure()
    plt.plot(range(len(fid_list)), fid_list, label="Fidelity")
    plt.xlabel("Test sample")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Data loading
# =============================================================================


def load_npz_dataset(dataset_path: str, n_samples=None):
    with np.load(dataset_path, mmap_mode="r") as data:
        X = data["weights"] if "weights" in data else data["X"]
        Y = data["amps"] if "amps" in data else data["Y"]
        if n_samples is not None:
            X = X[:n_samples]
            Y = Y[:n_samples]
        return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def load_and_split_dataset(dataset_path, n_samples, train_split, val_split, test_split,
                            split_mode="contiguous", split_key=None):
    if abs((train_split + val_split + test_split) - 1.0) > 1e-8:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    X, Y = load_npz_dataset(dataset_path, n_samples=n_samples)
    n = X.shape[0]
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    if split_mode == "contiguous":
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
        split_info = {
            "mode": "contiguous", "n_total": n, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "train_range": [0, n_train], "val_range": [n_train, n_train + n_val],
            "test_range": [n_train + n_val, n_train + n_val + n_test],
        }
    elif split_mode == "shuffled":
        if split_key is None:
            raise ValueError("split_mode='shuffled' requires split_key to be provided.")
        perm = np.asarray(jax.random.permutation(split_key, n))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:n_train + n_val + n_test]
        split_info = {
            "mode": "shuffled", "n_total": n, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "perm_hash": hash_array(perm),
        }
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}. Use 'contiguous' or 'shuffled'.")

    return (X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], X[test_idx], Y[test_idx], split_info)


def load_data_generation_info(data_path: str) -> dict:
    """Read 01_data_generate.py's reproducibility_manifest.json from next to
    DATA_PATH (same directory) and pull out its `resolved_config` -- the
    vertices/n_samples/seed/etc. needed to regenerate this exact dataset from
    scratch. Stored under this run's own manifest so a lost/hash-mismatched
    dataset file can still be regenerated, not just detected as changed."""
    manifest_path = Path(data_path).parent / "reproducibility_manifest.json"
    if not manifest_path.exists():
        return {"available": False, "manifest_path": str(manifest_path),
                "note": "No Stage-1 reproducibility_manifest.json found next to DATA_PATH."}
    with open(manifest_path) as f:
        upstream = json.load(f)
    return {
        "available": True,
        "manifest_path": str(manifest_path),
        "resolved_config": upstream.get("resolved_config"),
        "git": upstream.get("git"),
        "environment": upstream.get("environment"),
    }


# =============================================================================
# JAX training / evaluation
# =============================================================================


def make_epoch_perm(n_samples: int, batch_size: int, rng_key):
    perm = jax.random.permutation(rng_key, n_samples)
    n_batches = n_samples // batch_size
    perm = perm[: n_batches * batch_size]
    return perm.reshape(n_batches, batch_size)


def normalize_state_vectors(y, eps: float = 1e-12):
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    norm = jnp.where(norm > eps, norm, 1.0)
    return y / norm


def mae_loss(params, x, y_target, apply_fn, normalize_output: bool):
    y_pred = apply_fn(params, x)
    if normalize_output:
        y_pred = normalize_state_vectors(y_pred)
        y_target = normalize_state_vectors(y_target)
    return jnp.mean(jnp.abs(y_pred - y_target))


def create_train_step(normalize_output: bool):
    @jax.jit
    def train_step(state, x, y_target):
        loss, grad = jax.value_and_grad(mae_loss)(
            state.params, x, y_target, state.apply_fn, normalize_output,
        )
        state = state.apply_gradients(grads=grad)
        return state, loss
    return train_step


def create_train_epoch(train_step):
    @jax.jit
    def train_epoch(state, X_train, Y_train, perm_batches):
        def body_fn(carry, idx):
            xb = X_train[idx]
            yb = Y_train[idx]
            new_state, loss = train_step(carry, xb, yb)
            return new_state, loss
        state, losses = jax.lax.scan(body_fn, state, perm_batches)
        return state, jnp.mean(losses)
    return train_epoch


def create_eval_step(apply_fn, normalize_output=False):
    @jax.jit
    def eval_step(params, xb, yb):
        pred = apply_fn(params, xb)
        if normalize_output:
            pred = normalize_state_vectors(pred)
            yb = normalize_state_vectors(yb)
        return jnp.mean(jnp.abs(pred - yb))
    return eval_step


def eval_epoch(params, X, Y, eval_step, batch_size):
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size
    losses = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        losses.append(eval_step(params, X[start:end], Y[start:end]))
    return float(jnp.mean(jnp.asarray(losses)))


def create_test_step(apply_fn, normalize_output=False):
    @jax.jit
    def test_step(params, xb, yb):
        pred = apply_fn(params, xb)
        if normalize_output:
            pred = normalize_state_vectors(pred)
            yb = normalize_state_vectors(yb)

        mae_s = jnp.mean(jnp.abs(pred - yb), axis=-1)
        mse_s = jnp.mean((pred - yb) ** 2, axis=-1)

        pred_f = pred.reshape((pred.shape[0], -1))
        yb_f = yb.reshape((yb.shape[0], -1))
        numerator = jnp.abs(jnp.sum(jnp.conj(yb_f) * pred_f, axis=-1)) ** 2
        denominator = (
            jnp.sum(jnp.abs(yb_f) ** 2, axis=-1) * jnp.sum(jnp.abs(pred_f) ** 2, axis=-1) + 1e-12
        )
        fid_s = numerator / denominator
        loss_s = mae_s
        return (jnp.mean(loss_s), jnp.mean(fid_s), jnp.mean(mse_s), jnp.mean(mae_s),
                loss_s, fid_s, mse_s, mae_s)
    return test_step


def test_epoch(params, X, Y, apply_fn, test_step, batch_size: int = 20):
    n = X.shape[0]
    total_loss = total_fid = total_mse = total_mae = 0.0
    total_n = 0
    loss_sample, fid_sample, mse_sample, mae_sample = [], [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb, yb = X[start:end], Y[start:end]
        (loss_b, fid_b, mse_b, mae_b, loss_s, fid_s, mse_s, mae_s) = test_step(params, xb, yb)
        bs = xb.shape[0]
        total_loss += float(loss_b) * bs
        total_fid += float(fid_b) * bs
        total_mse += float(mse_b) * bs
        total_mae += float(mae_b) * bs
        total_n += bs
        loss_sample.append(jax.device_get(loss_s))
        fid_sample.append(jax.device_get(fid_s))
        mse_sample.append(jax.device_get(mse_s))
        mae_sample.append(jax.device_get(mae_s))

    return (
        total_loss / total_n, total_fid / total_n, total_mse / total_n, total_mae / total_n,
        np.concatenate(loss_sample), np.concatenate(fid_sample),
        np.concatenate(mse_sample), np.concatenate(mae_sample),
    )


def create_cosine_adamw_optimizer(learning_rate, lr_after_decay, decay_until_epoch,
                                   steps_per_epoch, weight_decay=1e-4):
    total_decay_steps = int(decay_until_epoch * steps_per_epoch)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=total_decay_steps, alpha=lr_after_decay / learning_rate,
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    return tx, lr_schedule


# =============================================================================
# Checkpoint save / restore (schema v2, with legacy fallback)
# =============================================================================


def _checkpoint_target(state, seed: int, keys: dict, precision: str, split_mode: str) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "state": state, "epoch": -1,
        "best_val": np.inf, "best_epoch": -1, "wait": 0, "seed": int(seed),
        "key_model_init": keys["model_init"], "key_data_split": keys["data_split"],
        "key_epoch_root": keys["epoch_root"], "key_batch_root": keys["batch_root"],
        "key_dropout_root": keys["dropout_root"], "precision": precision, "split_mode": split_mode,
    }


def _save_checkpoint(*, ckpt_dir, state, epoch, best_val, best_epoch, wait, keys, seed, precision, split_mode):
    payload = _checkpoint_target(state, seed, keys, precision, split_mode)
    payload.update({"epoch": epoch, "best_val": float(best_val), "best_epoch": int(best_epoch), "wait": int(wait)})
    checkpoints.save_checkpoint(ckpt_dir, payload, step=epoch, keep=5)


def _restore_checkpoint_with_fallback(*, ckpt_dir_restore, fresh_state, keys, log_file_path,
                                       seed, precision, split_mode):
    raw = checkpoints.restore_checkpoint(ckpt_dir_restore, target=None)
    if raw is None:
        raise FileNotFoundError(f"No checkpoint found to restore in: {ckpt_dir_restore}")

    schema_version = raw.get("schema_version") if isinstance(raw, dict) else None

    if schema_version is not None and int(schema_version) >= 2:
        full_target = _checkpoint_target(fresh_state, seed, keys, precision, split_mode)
        restored = checkpoints.restore_checkpoint(ckpt_dir_restore, target=full_target)

        restored_seed = int(restored["seed"])
        if restored_seed != int(seed):
            log_file(log_file_path,
                      f"WARNING: checkpoint was saved with seed={restored_seed}, but the "
                      f"current SEED={seed} differs. Exact resumed-trajectory reproduction "
                      f"is NOT guaranteed -- only params/optimizer-state continuation is.")
        else:
            log_file(log_file_path,
                      f"Resumed from schema_version={schema_version} checkpoint; seed matches ({seed}).")

        state = restored["state"]
        start_epoch = int(restored["epoch"]) + 1
        best_val = float(restored["best_val"])
        best_epoch = int(restored["best_epoch"])
        wait = int(restored["wait"])
        log_file(log_file_path, f"Resuming from epoch: {start_epoch}")
        return state, start_epoch, best_val, best_epoch, wait

    log_file(log_file_path,
              "WARNING: legacy checkpoint detected (no schema_version >= 2). Params and "
              "optimizer state will still restore correctly; exact resumed-trajectory "
              "reproduction is NOT guaranteed by this path.")
    legacy_target = {"state": fresh_state, "epoch": -1, "best_val": np.inf, "best_epoch": -1, "wait": 0}
    restored = checkpoints.restore_checkpoint(ckpt_dir_restore, target=legacy_target)
    state = restored["state"]
    start_epoch = int(restored["epoch"]) + 1
    best_val = float(restored["best_val"])
    best_epoch = int(restored["best_epoch"])
    wait = int(restored["wait"])
    log_file(log_file_path, f"Resuming from epoch: {start_epoch}")
    return state, start_epoch, best_val, best_epoch, wait


# =============================================================================
# Main training function
# =============================================================================


def _default_cfg() -> dict:
    """Snapshot of this file's top-of-file constants, used when
    train_surrogate_model() is called with no override."""
    return dict(
        NODES=NODES, DIMENSIONS=DIMENSIONS, DATE=DATE, MODEL_NAME=MODEL_NAME,
        HIDDEN_DIM=HIDDEN_DIM, DATA_PATH=DATA_PATH, DATA_SIZE=DATA_SIZE,
        NORMALIZE_MODEL_OUTPUT=NORMALIZE_MODEL_OUTPUT, TRAIN_SPLIT=TRAIN_SPLIT,
        VAL_SPLIT=VAL_SPLIT, TEST_SPLIT=TEST_SPLIT, LEARNING_RATE=LEARNING_RATE,
        LR_AFTER_DECAY=LR_AFTER_DECAY, LR_DECAY_UNTIL_EPOCH=LR_DECAY_UNTIL_EPOCH,
        BATCH_SIZE=BATCH_SIZE, NUM_EPOCHS=NUM_EPOCHS, PATIENCE=PATIENCE, TOLERANCE=TOLERANCE,
        SEED=SEED, SPLIT_MODE=SPLIT_MODE, PRECISION=PRECISION,
        DATASET_HASH_MODE=DATASET_HASH_MODE, DEBUG_TRACE=DEBUG_TRACE,
        RESUME_FULL_STATE=RESUME_FULL_STATE, CKPT_DIR_RESTORE=CKPT_DIR_RESTORE,
        ROOT_FOLDER=ROOT_FOLDER, LOSS_NAME=LOSS_NAME,
        DETERMINISTIC_XLA=DETERMINISTIC_XLA,
    )


def train_surrogate_model(cfg: dict | None = None):
    """Train the surrogate model. cfg overrides this file's top-of-file
    constants; pass None (default) to use them directly -- no external
    config file is read either way."""
    cfg = dict(_default_cfg()) if cfg is None else dict(cfg)

    nodes = cfg["NODES"]
    dimensions = cfg["DIMENSIONS"]
    model_name = cfg["MODEL_NAME"]
    hidden_dim = cfg["HIDDEN_DIM"]
    data_path = cfg["DATA_PATH"]
    data_size = cfg["DATA_SIZE"]
    normalize_model_output = cfg["NORMALIZE_MODEL_OUTPUT"]
    train_split = cfg["TRAIN_SPLIT"]
    val_split = cfg["VAL_SPLIT"]
    test_split = cfg["TEST_SPLIT"]
    learning_rate = cfg["LEARNING_RATE"]
    lr_after_decay = cfg["LR_AFTER_DECAY"]
    lr_decay_until_epoch = cfg["LR_DECAY_UNTIL_EPOCH"]
    batch_size = cfg["BATCH_SIZE"]
    num_epochs = cfg["NUM_EPOCHS"]
    patience = cfg["PATIENCE"]
    tolerance = cfg["TOLERANCE"]
    seed = cfg["SEED"]
    split_mode = cfg.get("SPLIT_MODE", "contiguous")
    precision = cfg.get("PRECISION", "float32")
    dataset_hash_mode = cfg.get("DATASET_HASH_MODE", "fast")
    debug_trace = cfg.get("DEBUG_TRACE", False)
    resume_full_state = cfg["RESUME_FULL_STATE"]
    ckpt_dir_restore = cfg["CKPT_DIR_RESTORE"]
    root_folder = cfg["ROOT_FOLDER"]
    loss_name = cfg["LOSS_NAME"]

    input_dims = 2 * nodes * (nodes - 1)
    output_dims = 2 ** nodes
    plot_title = f"Training loss, {model_name} (n={nodes}, hidden={hidden_dim})"

    run_dir, ckpt_dir, plot_dir, snap_dir, log_file_path = create_run_dirs(root_folder, model_name, nodes)

    model_info = {
        "description": f"{model_name}, {nodes}-node case", "root_folder": root_folder,
        "Folder": run_dir.name, "Nodes": nodes, "Dimensions": dimensions, "model_name": model_name,
        "architecture": hidden_dim, "input_dims": input_dims, "output_dims": output_dims,
        "batch_size": batch_size, "num_epoch": num_epochs, "lr": learning_rate,
        "lr_schedule": {"type": "cosine_decay", "initial_lr": learning_rate,
                        "final_lr": lr_after_decay, "decay_until_epoch": lr_decay_until_epoch},
        "seed": seed, "split_mode": split_mode, "precision": precision,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "patience": patience,
        "tolerance": tolerance, "train_split": train_split, "val_split": val_split,
        "test_split": test_split, "dataset": data_path, "data_size": data_size,
        "normalize_model_output": normalize_model_output, "train_cont_full_state": resume_full_state,
        "CKPT_DIR_restore": ckpt_dir_restore, "loss_name": loss_name, "plot_title": plot_title,
    }

    log_file(log_file_path, f"Run directory: {run_dir}")
    log_file(log_file_path, f"JAX devices: {jax.devices()}")
    log_file(log_file_path, f"Default backend: {jax.default_backend()}")
    log_file(log_file_path, f"Precision: {precision} (x64 enabled: {jax.config.jax_enable_x64})")
    log_file(log_file_path,
             f"Deterministic XLA flag requested: {cfg.get('DETERMINISTIC_XLA', True)} "
             f"(env has it: {deterministic_xla_env_is_set()})")
    if cfg.get("DETERMINISTIC_XLA", True) and not deterministic_xla_env_is_set():
        log_file(log_file_path,
                 "WARNING: DETERMINISTIC_XLA=True but XLA_FLAGS does not contain "
                 "--xla_gpu_deterministic_ops=true in this process's environment. For a "
                 "guaranteed effect, export XLA_FLAGS=--xla_gpu_deterministic_ops=true in "
                 "your shell/sbatch script before launching python.")

    model_info_path = run_dir / "model_info.json"
    write_json(model_info_path, model_info)

    keys = derive_root_keys(seed)

    model = create_model(model_name=model_name, hidden_dims=hidden_dim, out_dims=output_dims, nodes=nodes)
    x_example = jnp.zeros((1, input_dims), dtype=dtype_for_precision(precision))
    params = model.init(keys["model_init"], x_example)

    effective_data_size = data_size if data_size is not None else 1
    steps_per_epoch = max(int((effective_data_size * train_split) // batch_size), 1)

    tx, lr_schedule_fn = create_cosine_adamw_optimizer(
        learning_rate=learning_rate, lr_after_decay=lr_after_decay,
        decay_until_epoch=lr_decay_until_epoch, steps_per_epoch=steps_per_epoch, weight_decay=1e-4,
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if resume_full_state:
        state, start_epoch, best_val, best_epoch, wait = _restore_checkpoint_with_fallback(
            ckpt_dir_restore=ckpt_dir_restore, fresh_state=state, keys=keys,
            log_file_path=log_file_path, seed=seed, precision=precision, split_mode=split_mode,
        )
    else:
        start_epoch, best_val, best_epoch, wait = 0, np.inf, -1, 0

    data_loading_start = time.time()
    dataset_info = hash_file(data_path, mode=dataset_hash_mode)
    data_generation_info = load_data_generation_info(data_path)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, split_info = load_and_split_dataset(
        dataset_path=data_path, n_samples=data_size, train_split=train_split,
        val_split=val_split, test_split=test_split, split_mode=split_mode, split_key=keys["data_split"],
    )

    compute_dtype = dtype_for_precision(precision)
    X_train = jax.device_put(jnp.asarray(X_train, dtype=compute_dtype))
    Y_train = jax.device_put(jnp.asarray(Y_train, dtype=compute_dtype))
    X_val = jax.device_put(jnp.asarray(X_val, dtype=compute_dtype))
    Y_val = jax.device_put(jnp.asarray(Y_val, dtype=compute_dtype))

    data_loading_time = time.time() - data_loading_start
    log_file(log_file_path, f"Time for loading + splitting data: {data_loading_time}")

    model_info["actual_train_size"] = int(X_train.shape[0])
    model_info["actual_val_size"] = int(X_val.shape[0])
    model_info["actual_test_size"] = int(X_test.shape[0])
    write_json(model_info_path, model_info)

    manifest = build_manifest(
        resolved_config=dict(cfg),
        seeds={
            "seed": int(seed),
            "key_model_init": hash_array(keys["model_init"]),
            "key_data_split": hash_array(keys["data_split"]),
            "key_epoch_root": hash_array(keys["epoch_root"]),
            "key_batch_root": hash_array(keys["batch_root"]),
            "key_dropout_root": hash_array(keys["dropout_root"]),
        },
        input_file_info=dataset_info, repo_root=REPO_ROOT,
        extra={"split": split_info, "precision": precision,
               "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
               "data_generation": data_generation_info},
    )
    write_manifest(run_dir / "reproducibility_manifest.json", manifest)

    trace_path = run_dir / "trace.jsonl" if debug_trace else None

    train_step = create_train_step(normalize_output=normalize_model_output)
    train_epoch = create_train_epoch(train_step)
    eval_step = create_eval_step(apply_fn=state.apply_fn, normalize_output=normalize_model_output)
    test_step = create_test_step(apply_fn=state.apply_fn, normalize_output=normalize_model_output)

    train_losses, val_losses, epochs = [], [], []
    early_stop = False
    best_state = state
    training_time_computation = 0.0
    training_time_computation_list = []
    epoch_time_list = []
    full_loop_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_compute_start = time.time()

        epoch_key = epoch_key_for(keys["epoch_root"], epoch)
        perm_batches = make_epoch_perm(X_train.shape[0], batch_size, epoch_key)

        epoch_train_start = time.time()
        state, train_loss = train_epoch(state, X_train, Y_train, perm_batches)
        epoch_train_time = time.time() - epoch_train_start
        epoch_time_list.append(epoch_train_time)

        val_loss = eval_epoch(state.params, X_val, Y_val, eval_step, batch_size=batch_size)

        epoch_compute_time = time.time() - epoch_compute_start
        training_time_computation += epoch_compute_time
        training_time_computation_list.append(training_time_computation)

        num_batch = perm_batches.shape[0]
        avg_time_batch = float(epoch_train_time) / float(num_batch)
        current_lr = float(lr_schedule_fn(int(state.step)))

        if val_loss < best_val - tolerance:
            best_val = float(val_loss)
            best_state = state
            best_epoch = epoch
            wait = 0

            snap_path = snap_dir / f"best_param_epoch_{epoch}.msgpack"
            save_params_msgpack(best_state.params, snap_path)
            _save_checkpoint(ckpt_dir=ckpt_dir, state=state, epoch=epoch, best_val=best_val,
                              best_epoch=best_epoch, wait=wait, keys=keys, seed=seed,
                              precision=precision, split_mode=split_mode)
            log_file(log_file_path, f"New checkpoint saved at Epoch {epoch}")
        else:
            wait += 1
            if wait >= patience:
                log_file(log_file_path, f"EARLY STOP at epoch {epoch}, best at {best_epoch}")
                early_stop = True
                break

        log_file(log_file_path,
                 f"Epoch={epoch:5d} | step={int(state.step):8d} | lr={current_lr:.2e} | "
                 f"train_loss={float(train_loss):.8f} | val_loss={float(val_loss):.6f} | "
                 f"epoch_compute_time={epoch_compute_time:.6f} | train_time_per_epoch={epoch_train_time:.6f} | "
                 f"wait={wait} | patience={patience} | avg_batch_time={avg_time_batch:.6f} | ")

        if trace_path is not None:
            from utils import append_trace_line, hash_pytree
            append_trace_line(trace_path, {
                "global_step": int(state.step), "epoch": epoch,
                "batch_indices_hash": hash_array(perm_batches), "learning_rate": current_lr,
                "train_loss": float(train_loss), "val_loss": float(val_loss),
                "params_hash": hash_pytree(state.params), "opt_state_hash": hash_pytree(state.opt_state),
                "epoch_key_hash": hash_array(epoch_key),
            })

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        epochs.append(epoch)

        if epoch % 100 == 0:
            save_training_plot(epochs, train_losses, val_losses, plot_dir / "training_curves.png",
                                loss_name, plot_title)

        model_info.update({
            "train_loss": train_losses, "val_loss": val_losses, "best_val_loss": best_val,
            "best_epoch": best_epoch, "data_loading_time": data_loading_time,
            "cumulative_compute_time": training_time_computation_list,
            "total_compute_time": training_time_computation, "train_step_time_list": epoch_time_list,
        })
        write_json(model_info_path, model_info)

    full_loop_time = time.time() - full_loop_start
    log_file(log_file_path, f"Total compute time: {training_time_computation / 60:.6f} mins")
    log_file(log_file_path, f"Total wall time: {full_loop_time / 60:.6f} mins")

    if best_state is not None:
        state = best_state
        log_file(log_file_path, f"Restored best params from epoch {best_epoch}")
    else:
        log_file(log_file_path, "No best state saved, using last state")

    params_path = run_dir / "params.msgpack"
    save_params_msgpack(state.params, params_path)
    log_file(log_file_path, f"Saved params to {params_path}")

    X_test = jax.device_put(jnp.asarray(X_test, dtype=compute_dtype))
    Y_test = jax.device_put(jnp.asarray(Y_test, dtype=compute_dtype))

    (loss_mean, fidelity_mean, mse_mean, mae_mean,
     loss_list, fid_list, mse_list, mae_list) = test_epoch(
        state.params, X_test, Y_test, state.apply_fn, test_step, batch_size=20,
    )

    log_file(log_file_path,
             f"Test Loss: {loss_mean} | fidelity: {fidelity_mean} | mae: {mae_mean} | mse: {mse_mean}")

    save_test_plot(fid_list, plot_dir / "test_fidelity_curve.png",
                    title=f"Test Fidelity (n={nodes}, {loss_name})")
    save_training_plot(epochs, train_losses, val_losses, plot_dir / "training_curves.png",
                        loss_name, plot_title)

    np.savez(run_dir / "test_metrics.npz", loss=loss_list, fidelity=fid_list, mse=mse_list, mae=mae_list)

    model_info.update({
        "early_stop": early_stop, "train_loss": train_losses, "val_loss": val_losses,
        "best_loss": best_val, "best_epoch": best_epoch, "data_loading_time": data_loading_time,
        "cumulative_compute_time": training_time_computation_list,
        "total_compute_time": training_time_computation, "total_wall_time": full_loop_time,
        "train_step_time_list": epoch_time_list, "test_loss": float(loss_mean),
        "test_fidelity": float(fidelity_mean), "test_mse": float(mse_mean), "test_mae": float(mae_mean),
    })
    write_json(model_info_path, model_info)
    log_file(log_file_path, "Reached end of script!")

    return run_dir


if __name__ == "__main__":
    run_dir = train_surrogate_model()
    print(f"Training complete. Results saved in: {run_dir}")
