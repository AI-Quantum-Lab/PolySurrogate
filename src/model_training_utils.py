"""
Utility functions for surrogate model training.

Keeps Python-side helper code separate from the main training script.
The speed-critical JAX functions remain JIT compiled.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.serialization import to_bytes
from flax.training import checkpoints


# -----------------------------------------------------------------------------
# Folder / logging utilities
# -----------------------------------------------------------------------------

def create_run_dirs(root_folder: str, run_name: str):
    """Create run, checkpoint, plot, and snapshot directories."""
    runs_dir = Path(root_folder)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = runs_dir / f"run_{run_name}"
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
    """Append a line to a log file."""
    if not isinstance(txt, str):
        txt = np.array2string(np.asarray(txt))
    with path.open("a") as f:
        f.write(txt + "\n")


def write_json(file_path: Path, payload: dict):
    """Write dictionary to JSON."""
    with file_path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_training_plot(epochs, train_losses, val_losses, plot_path, loss_name, title):
    """Save training/validation curve."""
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
    """Save per-sample test fidelity curve."""
    plt.figure()
    plt.plot(range(len(fid_list)), fid_list, label="Fidelity")
    plt.xlabel("Test sample")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_params_msgpack(params, params_path: Path):
    """Save Flax parameters as msgpack."""
    with params_path.open("wb") as f:
        f.write(to_bytes(params))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_npz_dataset(dataset_path: str, n_samples: int | None = None):
    """
    Load generated dataset.

    Expected keys from your data generator:
        weights -> X
        amps    -> Y

    Also accepts X/Y as fallback names.
    """
    with np.load(dataset_path, mmap_mode="r") as data:
        if "weights" in data:
            X = data["weights"]
        elif "X" in data:
            X = data["X"]
        else:
            raise KeyError("Dataset must contain 'weights' or 'X'.")

        if "amps" in data:
            Y = data["amps"]
        elif "Y" in data:
            Y = data["Y"]
        else:
            raise KeyError("Dataset must contain 'amps' or 'Y'.")

        if n_samples is not None:
            X = X[:n_samples]
            Y = Y[:n_samples]

        # Materialize arrays because JAX needs real arrays for device_put.
        return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def load_and_split_dataset(
    dataset_path: str,
    n_samples: int | None,
    train_split: float,
    val_split: float,
    test_split: float,
):
    """Load one saved dataset and split contiguously into train/val/test."""
    if abs((train_split + val_split + test_split) - 1.0) > 1e-8:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    X, Y = load_npz_dataset(dataset_path, n_samples=n_samples)
    n = X.shape[0]

    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[n_train:n_train + n_val]
    Y_val = Y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:n_train + n_val + n_test]
    Y_test = Y[n_train + n_val:n_train + n_val + n_test]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# -----------------------------------------------------------------------------
# JAX training / evaluation utilities
# -----------------------------------------------------------------------------

def make_epoch_perm(n_samples: int, batch_size: int, rng_key):
    """
    Make per-sample shuffled batch indices.

    Same behavior as your original code: random permutation of samples,
    then reshaped into batches.
    """
    perm = jax.random.permutation(rng_key, n_samples)
    n_batches = n_samples // batch_size
    perm = perm[: n_batches * batch_size]
    return perm.reshape(n_batches, batch_size)


def normalize_state_vectors(y, eps: float = 1e-12):
    """Normalize vectors along the last axis."""
    norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    norm = jnp.where(norm > eps, norm, 1.0)
    return y / norm


def mae_loss(params, x, y_target, apply_fn, normalize_output: bool):
    """
    MAE training loss.

    If normalize_output=True, both prediction and target are normalized before
    the loss. Use this when the generated dataset contains normalized state
    vectors. If False, the model learns raw/unnormalized amplitudes.
    """
    y_pred = apply_fn(params, x)

    if normalize_output:
        y_pred = normalize_state_vectors(y_pred)
        y_target = normalize_state_vectors(y_target)

    return jnp.mean(jnp.abs(y_pred - y_target))


def create_train_step(normalize_output: bool):
    """Create JIT-compiled train step."""

    @jax.jit
    def train_step(state, x, y_target):
        loss, grad = jax.value_and_grad(mae_loss)(
            state.params,
            x,
            y_target,
            state.apply_fn,
            normalize_output,
        )
        state = state.apply_gradients(grads=grad)
        return state, loss

    return train_step


def create_train_epoch(train_step):
    """Create JIT-compiled training epoch using lax.scan."""

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

        loss = jnp.mean(jnp.abs(pred - yb))
        return loss

    return eval_step


def eval_epoch(params, X, Y, eval_step, batch_size):
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    losses = []

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        xb = X[start:end]
        yb = Y[start:end]

        loss_b = eval_step(params, xb, yb)
        losses.append(loss_b)

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
            jnp.sum(jnp.abs(yb_f) ** 2, axis=-1)
            * jnp.sum(jnp.abs(pred_f) ** 2, axis=-1)
            + 1e-12
        )

        fid_s = numerator / denominator
        loss_s = mae_s

        return (
            jnp.mean(loss_s),
            jnp.mean(fid_s),
            jnp.mean(mse_s),
            jnp.mean(mae_s),
            loss_s,
            fid_s,
            mse_s,
            mae_s,
        )

    return test_step


def test_epoch(params, X, Y, apply_fn, test_step, batch_size: int = 20):
    """Run testing and return mean plus per-sample metrics."""
    n = X.shape[0]

    total_loss = 0.0
    total_fid = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_n = 0

    loss_sample = []
    fid_sample = []
    mse_sample = []
    mae_sample = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]
        yb = Y[start:end]

        (
            loss_b,
            fid_b,
            mse_b,
            mae_b,
            loss_s,
            fid_s,
            mse_s,
            mae_s,
        ) = test_step(params, xb, yb)

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

    loss_sample = np.concatenate(loss_sample)
    fid_sample = np.concatenate(fid_sample)
    mse_sample = np.concatenate(mse_sample)
    mae_sample = np.concatenate(mae_sample)

    return (
        total_loss / total_n,
        total_fid / total_n,
        total_mse / total_n,
        total_mae / total_n,
        loss_sample,
        fid_sample,
        mse_sample,
        mae_sample,
    )


# -----------------------------------------------------------------------------
# Optimizer / LR helpers
# -----------------------------------------------------------------------------

def create_cosine_adamw_optimizer(
    learning_rate: float,
    lr_after_decay: float,
    decay_until_epoch: int,
    steps_per_epoch: int,
    weight_decay: float = 1e-4,
):
    """AdamW with cosine decay matching your original code."""
    total_decay_steps = int(decay_until_epoch * steps_per_epoch)

    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_decay_steps,
        alpha=lr_after_decay / learning_rate,
    )

    return optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
    )


def get_lr_for_epoch(epoch: int, learning_rate: float, lr_after_decay: float, decay_until_epoch: int):
    """Compute LR value for logging."""
    progress = min(epoch / decay_until_epoch, 1.0)
    alpha = lr_after_decay / learning_rate
    return learning_rate * (
        alpha + (1.0 - alpha) * 0.5 * (1.0 + np.cos(np.pi * progress))
    )


# Need optax imported after function docs to keep file imports explicit.
import optax  # noqa: E402
