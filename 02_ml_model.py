"""Reproducible JAX surrogate training. Validation every epoch. No plotting."""

import os

DETERMINISTIC_XLA = True
if DETERMINISTIC_XLA:
    flag = "--xla_gpu_deterministic_ops=true"
    current = os.environ.get("XLA_FLAGS", "")
    if flag not in current:
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()

import json
import logging
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state

from utils import (
    build_manifest,
    configure_precision,
    create_model,
    dtype_for_precision,
    hash_array,
    hash_file,
    save_params_msgpack,
    write_json,
    write_manifest,
)


# =============================================================================
# 1. PARAMETERS
# =============================================================================

NODES = 8
MODEL_NAME = "PNN"                 # "PNN" or "FNN"
HIDDEN_DIM = 15000   # FNN example: (2000, 2000, 2000)
NORMALIZE_MODEL_OUTPUT = False

DATA_PATH = "/home/bo48god/quick_tests/data_unnormalised/n8_20M/dataset_merged.npz"
DATA_SIZE = 10_000_000     # reduced from full 20M -- full size OOM'd on 40GB A100 (needed >40GB actual GPU mem vs ~26.5GB estimate); 10M gives ~13.2GB train+val on GPU, safe margin
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# "shuffled" (default, original behavior of this file): seeded
# jax.random.permutation split. "contiguous": first TRAIN_SPLIT fraction ->
# train, next VAL_SPLIT -> val, rest -> test, no randomness at all. Added
# specifically to A/B test how much the historical unseeded-shuffle
# split (which can't be recovered) actually matters for the loss
# trajectory, by comparing against this seeded code with everything else
# held fixed.
SPLIT_MODE = "contiguous"    # "shuffled" or "contiguous"

LEARNING_RATE = 1e-3
FINAL_LEARNING_RATE = 1e-5
LR_DECAY_UNTIL_EPOCH = 2000
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8000
NUM_EPOCHS = 10
PATIENCE = 100000
TOLERANCE = 1e-7

SEED = 158
PRECISION = "float32"              # "float32" or "float64"
DATASET_HASH_MODE = "fast"         # "full", "fast", or "none"

ROOT_FOLDER = "results/model_training"
RESUME_RUN_DIR = None              # Existing run directory, or None
CHECKPOINT_EVERY = 1               # 1 = latest checkpoint after every epoch
KEEP_LATEST_CHECKPOINTS = 1        # overwrite/remove older latest checkpoint

CONFIG = {
    "nodes": NODES,
    "model_name": MODEL_NAME,
    "hidden_dim": HIDDEN_DIM,
    "normalize_model_output": NORMALIZE_MODEL_OUTPUT,
    "data_path": DATA_PATH,
    "data_size": DATA_SIZE,
    "train_split": TRAIN_SPLIT,
    "val_split": VAL_SPLIT,
    "test_split": TEST_SPLIT,
    "split_mode": SPLIT_MODE,
    "learning_rate": LEARNING_RATE,
    "final_learning_rate": FINAL_LEARNING_RATE,
    "lr_decay_until_epoch": LR_DECAY_UNTIL_EPOCH,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "patience": PATIENCE,
    "tolerance": TOLERANCE,
    "seed": SEED,
    "precision": PRECISION,
    "dataset_hash_mode": DATASET_HASH_MODE,
    "deterministic_xla": DETERMINISTIC_XLA,
    "checkpoint_every": CHECKPOINT_EVERY,
    "keep_latest_checkpoints": KEEP_LATEST_CHECKPOINTS,
}

configure_precision(PRECISION)

# Explicitly request the fastest matmul precision mode for float32 (allows
# TF32 tensor-core execution on Ampere+ GPUs). Investigated because the
# float32 6FNN run was empirically SLOWER than the float64 one on A100 with
# --xla_gpu_deterministic_ops=true set -- suspected cause: without an
# explicit precision request, XLA's kernel selection under deterministic
# mode was not taking the fast TF32 path for float32 matmuls, while A100's
# native FP64 tensor cores gave float64 a real hardware advantage instead.
# Scoped to this file only (not utils.py), since utils.py is shared by
# 02_ml_model.py, whose established reproducibility comparisons (fold_in
# vs split, CPU vs GPU) should not have their numerics touched by this.
if PRECISION == "float32":
    jax.config.update("jax_default_matmul_precision", "default")


# =============================================================================
# 2. RUN DIRECTORY, LOGGER, AND NPY HISTORY
# =============================================================================

HISTORY_DTYPE = np.dtype([
    ("epoch", np.int64),
    ("step", np.int64),
    ("learning_rate", np.float64),
    ("train_loss", np.float64),
    ("validation_loss", np.float64),
    ("best_validation_loss", np.float64),
    ("best_epoch", np.int64),
    ("wait", np.int64),
    ("shuffle_seconds", np.float64),
    ("train_seconds", np.float64),
    ("validation_seconds", np.float64),
    ("best_checkpoint_seconds", np.float64),
    ("latest_checkpoint_seconds", np.float64),
    ("metrics_flush_seconds", np.float64),
    ("epoch_compute_seconds", np.float64),
    ("epoch_wall_seconds", np.float64),
    ("latest_checkpoint_saved", np.bool_),
    ("best_checkpoint_saved", np.bool_),
])


def next_run_dir() -> Path:
    parent = Path(ROOT_FOLDER) / MODEL_NAME / f"n{NODES}"
    parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while (parent / f"n{NODES}_{index}").exists():
        index += 1
    return (parent / f"n{NODES}_{index}").resolve()


def prepare_run() -> tuple[Path, bool]:
    if RESUME_RUN_DIR is None:
        run_dir = next_run_dir()
        run_dir.mkdir(parents=True)
        return run_dir, False

    run_dir = Path(RESUME_RUN_DIR).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir, True


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_resume_config(run_dir: Path) -> None:
    path = run_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing resume configuration: {path}")
    with path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    current = json.loads(json.dumps(CONFIG))
    if saved != current:
        differences = {
            key: {"saved": saved.get(key), "current": current.get(key)}
            for key in sorted(set(saved) | set(current))
            if saved.get(key) != current.get(key)
        }
        raise ValueError(f"Resume configuration differs: {differences}")


def initialise_history(history) -> None:
    history[:] = np.zeros(history.shape, dtype=HISTORY_DTYPE)
    for name in ("epoch", "step", "best_epoch", "wait"):
        history[name] = -1
    for name in HISTORY_DTYPE.names:
        if np.issubdtype(HISTORY_DTYPE[name], np.floating):
            history[name] = np.nan
    history.flush()


def open_history(path: Path, resuming: bool):
    if resuming and path.exists():
        history = np.lib.format.open_memmap(path, mode="r+")
        if history.dtype != HISTORY_DTYPE or history.shape != (NUM_EPOCHS,):
            raise ValueError("training_history.npy is incompatible with this configuration")
        return history

    history = np.lib.format.open_memmap(
        path, mode="w+", dtype=HISTORY_DTYPE, shape=(NUM_EPOCHS,)
    )
    initialise_history(history)
    return history


def clear_history_after(history, start_epoch: int) -> None:
    if start_epoch >= NUM_EPOCHS:
        return
    initialise = np.zeros(NUM_EPOCHS - start_epoch, dtype=HISTORY_DTYPE)
    initialise["epoch"] = -1
    initialise["step"] = -1
    initialise["best_epoch"] = -1
    initialise["wait"] = -1
    for name in HISTORY_DTYPE.names:
        if np.issubdtype(HISTORY_DTYPE[name], np.floating):
            initialise[name] = np.nan
    history[start_epoch:] = initialise
    history.flush()


def update_history(history, epoch: int, values: dict) -> None:
    for name, value in values.items():
        history[name][epoch] = value


# =============================================================================
# 3. DATA
# =============================================================================

def load_dataset(path: str, n_samples: int | None):
    target_dtype = np.float64 if PRECISION == "float64" else np.float32
    with np.load(path, mmap_mode="r") as data:
        if "weights" in data and "amps" in data:
            x_source, y_source = data["weights"], data["amps"]
        elif "X" in data and "Y" in data:
            x_source, y_source = data["X"], data["Y"]
        else:
            raise KeyError("Dataset must contain weights/amps or X/Y")

        source_dtypes = {
            "input_dtype": str(x_source.dtype),
            "target_dtype": str(y_source.dtype),
        }
        if n_samples is not None:
            x_source, y_source = x_source[:n_samples], y_source[:n_samples]
        x = np.asarray(x_source, dtype=target_dtype)
        y = np.asarray(y_source, dtype=target_dtype)

    return x, y, source_dtypes


def random_reproducible_split(x, y, split_key):
    if not np.isclose(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT, 1.0):
        raise ValueError("TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1")

    n = x.shape[0]
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    if SPLIT_MODE == "contiguous":
        permutation = np.arange(n)
    elif SPLIT_MODE == "shuffled":
        permutation = jax.random.permutation(split_key, n)
        permutation.block_until_ready()
        permutation = np.asarray(jax.device_get(permutation))
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE!r}. Use 'shuffled' or 'contiguous'.")

    return (
        x[permutation[:n_train]],
        y[permutation[:n_train]],
        x[permutation[n_train:n_train + n_val]],
        y[permutation[n_train:n_train + n_val]],
        x[permutation[n_train + n_val:]],
        y[permutation[n_train + n_val:]],
        permutation,
    )


# =============================================================================
# 4. JITTED TRAINING AND VALIDATION
# =============================================================================

def normalize_vectors(values, eps=1e-12):
    norm = jnp.linalg.norm(values, axis=-1, keepdims=True)
    return values / jnp.where(norm > eps, norm, 1.0)


def loss_fn(params, x, y, apply_fn):
    prediction = apply_fn(params, x)
    if NORMALIZE_MODEL_OUTPUT:
        prediction = normalize_vectors(prediction)
        y = normalize_vectors(y)
    return jnp.mean(jnp.abs(prediction - y))


def train_step(state, x_batch, y_batch):
    loss, gradients = jax.value_and_grad(loss_fn)(
        state.params, x_batch, y_batch, state.apply_fn
    )
    return state.apply_gradients(grads=gradients), loss


@partial(jax.jit, donate_argnums=(0,))
def train_one_epoch(state, x_train, y_train, batch_indices):
    def body(current_state, indices):
        next_state, batch_loss = train_step(
            current_state, x_train[indices], y_train[indices]
        )
        return next_state, batch_loss

    state, losses = jax.lax.scan(body, state, batch_indices)
    return state, jnp.mean(losses)


@partial(jax.jit, static_argnames=("n_samples", "batch_size"))
def make_epoch_batches(key, n_samples: int, batch_size: int):
    return jax.random.permutation(key, n_samples).reshape(-1, batch_size)


def make_evaluation_batches(n_samples: int, batch_size: int):
    n_batches = (n_samples + batch_size - 1) // batch_size
    raw = np.arange(n_batches * batch_size, dtype=np.int64)
    mask = raw < n_samples
    indices = np.minimum(raw, n_samples - 1)
    shape = (n_batches, batch_size)
    return (
        jnp.asarray(indices.reshape(shape)),
        jnp.asarray(mask.reshape(shape), dtype=dtype_for_precision(PRECISION)),
    )


def create_validation_epoch(apply_fn):
    @jax.jit
    def validation_epoch(params, x, y, batch_indices, batch_mask):
        zero = jnp.asarray(0.0, dtype=x.dtype)

        def body(carry, batch):
            loss_sum, count = carry
            indices, mask = batch
            target = y[indices]
            prediction = apply_fn(params, x[indices])
            if NORMALIZE_MODEL_OUTPUT:
                prediction = normalize_vectors(prediction)
                target = normalize_vectors(target)
            sample_loss = jnp.mean(
                jnp.abs(prediction - target),
                axis=tuple(range(1, prediction.ndim)),
            )
            return (
                loss_sum + jnp.sum(sample_loss * mask),
                count + jnp.sum(mask),
            ), None

        (loss_sum, count), _ = jax.lax.scan(
            body, (zero, zero), (batch_indices, batch_mask)
        )
        return loss_sum / count

    return validation_epoch


def make_optimizer(steps_per_epoch: int):
    schedule = optax.cosine_decay_schedule(
        init_value=LEARNING_RATE,
        decay_steps=max(LR_DECAY_UNTIL_EPOCH * steps_per_epoch, 1),
        alpha=FINAL_LEARNING_RATE / LEARNING_RATE,
    )
    return optax.adamw(schedule, weight_decay=WEIGHT_DECAY), schedule


# =============================================================================
# 5. CHECKPOINTS AND TESTING
# =============================================================================

def checkpoint_payload(state, epoch, best_val, best_epoch, wait):
    return {
        "state": state,
        "epoch": int(epoch),
        "best_validation_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "wait": int(wait),
        "seed": int(SEED),
    }


def save_checkpoint(directory: Path, payload, epoch: int, keep: int):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints.save_checkpoint(
        str(directory), payload, step=epoch, keep=keep, overwrite=True
    )


def restore_latest(directory: Path, fresh_state):
    target = checkpoint_payload(fresh_state, -1, np.inf, -1, 0)
    restored = checkpoints.restore_checkpoint(str(directory), target=target)
    if int(restored["epoch"]) < 0:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    if int(restored["seed"]) != SEED:
        raise ValueError("Checkpoint seed does not match SEED")
    return (
        restored["state"],
        int(restored["epoch"]) + 1,
        float(restored["best_validation_loss"]),
        int(restored["best_epoch"]),
        int(restored["wait"]),
    )


def create_test_step(apply_fn):
    @jax.jit
    def test_step(params, x, y):
        prediction = apply_fn(params, x)
        if NORMALIZE_MODEL_OUTPUT:
            prediction = normalize_vectors(prediction)
            y = normalize_vectors(y)

        mae = jnp.mean(jnp.abs(prediction - y), axis=-1)
        mse = jnp.mean((prediction - y) ** 2, axis=-1)
        prediction = prediction.reshape((prediction.shape[0], -1))
        y = y.reshape((y.shape[0], -1))
        fidelity = jnp.abs(jnp.sum(jnp.conj(y) * prediction, axis=-1)) ** 2
        fidelity /= (
            jnp.sum(jnp.abs(y) ** 2, axis=-1)
            * jnp.sum(jnp.abs(prediction) ** 2, axis=-1)
            + 1e-12
        )
        return mae, mse, fidelity

    return test_step


def test_model(params, x, y, test_step, dtype):
    mae_all, mse_all, fidelity_all = [], [], []
    for start in range(0, x.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, x.shape[0])
        mae, mse, fidelity = test_step(
            params,
            jnp.asarray(x[start:end], dtype=dtype),
            jnp.asarray(y[start:end], dtype=dtype),
        )
        mae.block_until_ready()
        mae_all.append(np.asarray(jax.device_get(mae)))
        mse_all.append(np.asarray(jax.device_get(mse)))
        fidelity_all.append(np.asarray(jax.device_get(fidelity)))
    return (
        np.concatenate(mae_all),
        np.concatenate(mse_all),
        np.concatenate(fidelity_all),
    )


# =============================================================================
# 6. MAIN
# =============================================================================

def train() -> Path:
    total_wall_start = time.perf_counter()
    run_dir, resuming = prepare_run()
    latest_dir = run_dir / "checkpoints" / "latest"
    best_dir = run_dir / "checkpoints" / "best"
    history_path = run_dir / "training_history.npy"
    logger = make_logger(run_dir / "run.log")
    dtype = dtype_for_precision(PRECISION)

    logger.info("Run directory: %s", run_dir)
    logger.info("Backend: %s | devices: %s", jax.default_backend(), jax.devices())
    logger.info("Precision=%s | validation every epoch | plotting disabled", PRECISION)

    if resuming:
        validate_resume_config(run_dir)
    else:
        write_json(run_dir / "config.json", CONFIG)

    master_key = jax.random.PRNGKey(SEED)
    init_key = jax.random.fold_in(master_key, 0)
    split_key = jax.random.fold_in(master_key, 1)
    shuffle_root = jax.random.fold_in(master_key, 2)

    start = time.perf_counter()
    dataset_info = hash_file(DATA_PATH, mode=DATASET_HASH_MODE)
    dataset_hash_seconds = time.perf_counter() - start

    start = time.perf_counter()
    x, y, source_dtypes = load_dataset(DATA_PATH, DATA_SIZE)
    dataset_load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    x_train, y_train, x_val, y_val, x_test, y_test, split_perm = (
        random_reproducible_split(x, y, split_key)
    )
    dataset_split_seconds = time.perf_counter() - start

    if x_train.shape[0] % BATCH_SIZE != 0:
        raise ValueError("Training size must be divisible by BATCH_SIZE")

    start = time.perf_counter()
    x_train = jax.device_put(jnp.asarray(x_train, dtype=dtype))
    y_train = jax.device_put(jnp.asarray(y_train, dtype=dtype))
    x_val = jax.device_put(jnp.asarray(x_val, dtype=dtype))
    y_val = jax.device_put(jnp.asarray(y_val, dtype=dtype))
    y_val.block_until_ready()
    host_to_device_seconds = time.perf_counter() - start

    input_dim = 2 * NODES * (NODES - 1)
    output_dim = 2 ** NODES

    start = time.perf_counter()
    model = create_model(
        model_name=MODEL_NAME,
        hidden_dims=HIDDEN_DIM,
        out_dims=output_dim,
        nodes=NODES,
    )
    params = model.init(init_key, jnp.zeros((1, input_dim), dtype=dtype))
    jax.tree_util.tree_leaves(params)[0].block_until_ready()
    model_initialisation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    steps_per_epoch = x_train.shape[0] // BATCH_SIZE
    optimizer, lr_schedule = make_optimizer(steps_per_epoch)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    optimizer_initialisation_seconds = time.perf_counter() - start

    val_indices, val_mask = make_evaluation_batches(x_val.shape[0], BATCH_SIZE)
    validation_epoch = create_validation_epoch(model.apply)
    test_step = create_test_step(model.apply)
    history = open_history(history_path, resuming)

    if resuming:
        state, start_epoch, best_val, best_epoch, wait = restore_latest(
            latest_dir, state
        )
        clear_history_after(history, start_epoch)
        logger.info("Resuming from epoch %d", start_epoch)
    else:
        start_epoch, best_val, best_epoch, wait = 0, np.inf, -1, 0

    start = time.perf_counter()
    warm_key = jax.random.fold_in(shuffle_root, start_epoch)
    warm_batches = make_epoch_batches(
        warm_key, n_samples=x_train.shape[0], batch_size=BATCH_SIZE
    )
    warm_batches.block_until_ready()
    compiled_train = train_one_epoch.lower(
        state, x_train, y_train, warm_batches
    ).compile()
    compiled_validation = validation_epoch.lower(
        state.params, x_val, y_val, val_indices, val_mask
    ).compile()
    jit_compilation_seconds = time.perf_counter() - start
    del warm_batches

    manifest = build_manifest(
        resolved_config=CONFIG,
        seeds={
            "master_key": hash_array(master_key),
            "model_initialisation_key": hash_array(init_key),
            "data_split_key": hash_array(split_key),
            "epoch_shuffle_root": hash_array(shuffle_root),
        },
        input_file_info=dataset_info,
        repo_root=str(Path(__file__).resolve().parent),
        extra={
            "source_dtypes": source_dtypes,
            "split_permutation_hash": hash_array(split_perm),
            "train_size": int(x_train.shape[0]),
            "validation_size": int(x_val.shape[0]),
            "test_size": int(x_test.shape[0]),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
    )
    write_manifest(run_dir / "reproducibility_manifest.json", manifest)

    logger.info(
        "Setup seconds | hash=%.2f load=%.2f split=%.2f transfer=%.2f "
        "model_init=%.2f optimizer_init=%.2f jit_compile=%.2f",
        dataset_hash_seconds,
        dataset_load_seconds,
        dataset_split_seconds,
        host_to_device_seconds,
        model_initialisation_seconds,
        optimizer_initialisation_seconds,
        jit_compilation_seconds,
    )

    training_loop_start = time.perf_counter()
    early_stopped = False

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_wall_start = time.perf_counter()

        start = time.perf_counter()
        epoch_key = jax.random.fold_in(shuffle_root, epoch)
        batches = make_epoch_batches(
            epoch_key, n_samples=x_train.shape[0], batch_size=BATCH_SIZE
        )
        batches.block_until_ready()
        shuffle_seconds = time.perf_counter() - start

        start = time.perf_counter()
        state, train_loss_device = compiled_train(
            state, x_train, y_train, batches
        )
        train_loss_device.block_until_ready()
        train_seconds = time.perf_counter() - start
        train_loss = float(train_loss_device)

        start = time.perf_counter()
        val_loss_device = compiled_validation(
            state.params, x_val, y_val, val_indices, val_mask
        )
        val_loss_device.block_until_ready()
        validation_seconds = time.perf_counter() - start
        val_loss = float(val_loss_device)

        improved = val_loss < best_val - TOLERANCE
        if improved:
            best_val, best_epoch, wait = val_loss, epoch, 0
        else:
            wait += 1

        step = int(state.step)
        learning_rate = float(lr_schedule(max(step - 1, 0)))
        latest_saved = (
            (epoch + 1) % CHECKPOINT_EVERY == 0
            or epoch == NUM_EPOCHS - 1
            or wait >= PATIENCE
        )
        payload = checkpoint_payload(state, epoch, best_val, best_epoch, wait)

        update_history(history, epoch, {
            "epoch": epoch,
            "step": step,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "wait": wait,
            "shuffle_seconds": shuffle_seconds,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "best_checkpoint_seconds": 0.0,
            "latest_checkpoint_seconds": 0.0,
            "metrics_flush_seconds": 0.0,
            "epoch_compute_seconds": (
                shuffle_seconds + train_seconds + validation_seconds
            ),
            "epoch_wall_seconds": 0.0,
            "latest_checkpoint_saved": False,
            "best_checkpoint_saved": False,
        })

        start = time.perf_counter()
        history.flush()
        metrics_flush_seconds = time.perf_counter() - start

        latest_checkpoint_seconds = 0.0
        if latest_saved:
            start = time.perf_counter()
            save_checkpoint(
                latest_dir, payload, epoch, keep=KEEP_LATEST_CHECKPOINTS
            )
            latest_checkpoint_seconds = time.perf_counter() - start

        best_checkpoint_seconds = 0.0
        if improved:
            start = time.perf_counter()
            save_checkpoint(best_dir, payload, epoch, keep=1)
            save_params_msgpack(state.params, run_dir / "best_params.msgpack")
            best_checkpoint_seconds = time.perf_counter() - start

        epoch_wall_seconds = time.perf_counter() - epoch_wall_start
        update_history(history, epoch, {
            "best_checkpoint_seconds": best_checkpoint_seconds,
            "latest_checkpoint_seconds": latest_checkpoint_seconds,
            "metrics_flush_seconds": metrics_flush_seconds,
            "epoch_wall_seconds": epoch_wall_seconds,
            "latest_checkpoint_saved": latest_saved,
            "best_checkpoint_saved": improved,
        })
        history.flush()

        logger.info(
            "epoch=%d | train=%.8e | validation=%.8e | shuffle=%.2fs | "
            "train_time=%.2fs | validation_time=%.2fs | epoch_wall=%.2fs | "
            "latest_ckpt=%s | best_ckpt=%s | best_epoch=%d | wait=%d/%d",
            epoch,
            train_loss,
            val_loss,
            shuffle_seconds,
            train_seconds,
            validation_seconds,
            epoch_wall_seconds,
            latest_saved,
            improved,
            best_epoch,
            wait,
            PATIENCE,
        )

        if wait >= PATIENCE:
            early_stopped = True
            logger.info("Early stopping at epoch %d", epoch)
            break

    training_loop_seconds = time.perf_counter() - training_loop_start
    valid_history = history[history["epoch"] >= 0]

    start = time.perf_counter()
    best_target = checkpoint_payload(state, -1, np.inf, -1, 0)
    best_payload = checkpoints.restore_checkpoint(str(best_dir), target=best_target)
    if int(best_payload["epoch"]) < 0:
        raise FileNotFoundError("No best-validation checkpoint was found")
    best_state = best_payload["state"]
    save_params_msgpack(best_state.params, run_dir / "params.msgpack")
    best_restore_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mae, mse, fidelity = test_model(
        best_state.params, x_test, y_test, test_step, dtype
    )
    test_seconds = time.perf_counter() - start
    np.savez(
        run_dir / "test_metrics.npz",
        mae=mae,
        mse=mse,
        fidelity=fidelity,
    )

    summary = {
        "early_stopped": early_stopped,
        "completed_epochs": int(valid_history.shape[0]),
        "best_epoch": int(best_payload["best_epoch"]),
        "best_validation_loss": float(best_payload["best_validation_loss"]),
        "dataset_source_dtypes": source_dtypes,
        "dataset_hash_seconds": dataset_hash_seconds,
        "dataset_load_seconds": dataset_load_seconds,
        "dataset_split_seconds": dataset_split_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "model_initialisation_seconds": model_initialisation_seconds,
        "optimizer_initialisation_seconds": optimizer_initialisation_seconds,
        "jit_compilation_seconds": jit_compilation_seconds,
        "training_loop_seconds": training_loop_seconds,
        "total_shuffle_seconds": float(np.nansum(valid_history["shuffle_seconds"])),
        "total_train_seconds": float(np.nansum(valid_history["train_seconds"])),
        "total_validation_seconds": float(
            np.nansum(valid_history["validation_seconds"])
        ),
        "total_best_checkpoint_seconds": float(
            np.nansum(valid_history["best_checkpoint_seconds"])
        ),
        "total_latest_checkpoint_seconds": float(
            np.nansum(valid_history["latest_checkpoint_seconds"])
        ),
        "mean_epoch_wall_seconds": float(
            np.nanmean(valid_history["epoch_wall_seconds"])
        ),
        "best_restore_seconds": best_restore_seconds,
        "test_seconds": test_seconds,
        "total_wall_seconds": time.perf_counter() - total_wall_start,
        "test_mae": float(np.mean(mae)),
        "test_mse": float(np.mean(mse)),
        "test_fidelity": float(np.mean(fidelity)),
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Complete | %s", summary)

    history.flush()
    del history
    return run_dir


if __name__ == "__main__":
    train()