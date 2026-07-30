"""
inverse_design.py -- Stage 3: inverse-design optimisation.

Imports shared PyTheus catalog/amplitude code, model definitions, and
reproducibility primitives from utils.py (also used by data_generate.py
and ml_model.py); everything stage-specific (parameters, target states,
the jitter mechanism, the optimisation loop, pruning) lives in this file.
Edit the constants in the "Parameters" section below, then run:

    python inverse_design.py

Reads MODEL_PATH (the `params.msgpack` written by ml_model.py -- connected
only through that file, not through a Python import). Uses the trained
surrogate as a differentiable proxy for PyTheus: gradient descent on graph
weights minimises loss = (1 - fidelity) + lambda_l1 * ||x||_1, with
reproducible stall-detection jitter, verified against the exact PyTheus
simulator, then pruned to a sparse graph. Writes results under an
auto-numbered run folder:
    <RESULTS_ROOT>/<TARGET>_n<NPHOTONS>/<TARGET>_n<NPHOTONS>_<i>/
where `i` is the next free integer for that (target, node-count) pair --
never user-typed, same scheme data_generate.py / ml_model.py use.

Stopping condition is a single threshold: nn_fid >= EARLY_STOP_NN_FID OR
step >= MAX_TOTAL_STEPS.

All jitter noise is drawn through a fold_in-based PRNG key plan (see
utils.py's "Reproducibility: PRNG key plan" section) -- every draw is a pure
function of (SEED, sample_id, event_index), independent of any other
sample, of run order, or of which subset a sample is part of, as long as it
keeps the same sample_id (position in its dataset).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import data_generate

from utils import (
    build_manifest,
    build_pytheus_catalog,
    compute_amplitudes,
    create_model,
    fidelity_np,
    hash_file,
    load_trained_model,
    optimiser_jitter_event_key_for,
    optimiser_jitter_root_key,
    optimiser_sample_key_for,
    pytheus_state_from_x,
    save_json,
    write_manifest,
)


# =============================================================================
# Inverse-design configuration
# Edit values here directly — no external config file is used.
# =============================================================================


# -----------------------------------------------------------------------------
# Problem
# -----------------------------------------------------------------------------

NPHOTONS = 4
TARGET_NAME = "GHZ"       # "GHZ" | "W" | "LINEAR_CLUSTER" | "SINGLE" | "ZERO"
MODEL_TYPE = "PNN"        # "PNN" | "FNN"

INPUT_DIM = 2 * NPHOTONS * (NPHOTONS - 1)
OUT_DIM = 2**NPHOTONS


# -----------------------------------------------------------------------------
# Starting samples
# -----------------------------------------------------------------------------
# True  → generate fresh random starting graphs
# False → load starting graphs from CONDITIONED_DATA_PATH

GENERATE_DATA = False
CONDITIONED_DATA_PATH = (
    "paper_like_initial_fidelity_all_targets/n4/GHZ/paper_like_initial_fidelity_dataset.npz"
)
MAX_INITIAL_SAMPLES = None

DATA_SAMPLES = 1000
DATA_BATCH_SIZE = 5
DATA_SEED = 4

# Discard generated samples already too close to the target.
LOW_FIDELITY_THRESHOLD = 0.999

# Must match NORMED_DATA in 01_data_generate.py.
NORMED_DATA = False


# -----------------------------------------------------------------------------
# Trained surrogate
# -----------------------------------------------------------------------------

# PNN: integer hidden dimension
# FNN: tuple matching HIDDEN_DIM in ml_model.py
ARCHITECTURE = 400

MODEL_PATH = "results/model_training/PNN/n4/n4_5/best_params.msgpack"

# Must match NORMALIZE_MODEL_OUTPUT in ml_model.py.
NORMALIZE_MODEL_OUTPUT = False


# -----------------------------------------------------------------------------
# Optimisation objective
# -----------------------------------------------------------------------------
# loss = (1 - fidelity) + LAMBDA_L1 * sum(abs(x))
#
# The L1 penalty encourages sparse graphs with fewer active edges.

LAMBDA_L1 = 1e-3


# -----------------------------------------------------------------------------
# Optimisation
# -----------------------------------------------------------------------------

SEED = 46
NUM_STEPS = 100000                 # Smoke test; use 10_000+ for production
MAX_TOTAL_STEPS = 100000           # Hard ceiling; defaults to NUM_STEPS if unset
EARLY_STOP_NN_FID = 0.99999

LEARNING_RATE = 1e-2
MIN_LEARNING_RATE = 1e-6
LR_DECAY_STEPS = 100000            # For smoke tests, normally match NUM_STEPS
LR_EXPONENT = 1.0

CLIP_MIN = -1.0
CLIP_MAX = 1.0


# -----------------------------------------------------------------------------
# Jittering
# -----------------------------------------------------------------------------
# Jittering perturbs stalled trajectories but does not change the stopping
# criteria: EARLY_STOP_NN_FID or MAX_TOTAL_STEPS.
#
# Noise is reproducible and determined by:
#     (SEED, sample_id, event_index)

JITTER_ENABLED = True

# One-time perturbation before the first optimisation step.
INITIAL_JITTER = 0.01

# Noise scale for each successive stall-triggered jitter event.
JITTER_SCHEDULE = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.70,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
]

JITTER_MAX_EVENTS = 100000
STUCK_PATIENCE_STEPS = 1000
STUCK_MIN_IMPROVEMENT = 1e-4


# -----------------------------------------------------------------------------
# Verification and logging
# -----------------------------------------------------------------------------

PRINT_EVERY = 1
VERIFY_EVERY = 1

# Storing every gradient and update vector can produce very large output files.
STORE_STEP_VECTORS = False


# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------
# Thresholds are tested sequentially. A pruning step is accepted only when the
# resulting fidelity loss is no greater than PRUNE_FID_TOLERANCE.

PRUNE_FID_TOLERANCE = 1e-6
PRUNE_THRESHOLDS = [
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
]


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
# Output structure:
#
#   <RESULTS_ROOT>/
#       <TARGET>_n<NPHOTONS>/
#           <TARGET>_n<NPHOTONS>_<run_index>/
#
# The run index is assigned automatically by _next_run_dir.

RESULTS_ROOT = "results/inverse_design"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# =============================================================================
# Target quantum states
# =============================================================================


def ghz_state(n: int) -> np.ndarray:
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.float32)
    state[0] = 1.0 / np.sqrt(2.0)
    state[-1] = 1.0 / np.sqrt(2.0)
    return state


def w_state(n: int) -> np.ndarray:
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.float32)
    norm = 1.0 / np.sqrt(float(n))
    for i in range(n):
        bitstring = ["0"] * n
        bitstring[i] = "1"
        idx = int("".join(bitstring), 2)
        state[idx] = norm
    return state


def linear_cluster_state(n: int) -> np.ndarray:
    dim = 2 ** n
    norm = 1.0 / np.sqrt(float(dim))
    state = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        bitstring = format(i, f"0{n}b")
        bits = np.array(list(bitstring), dtype=np.int32)
        exponent = int(np.sum(bits[:-1] * bits[1:]))
        state[i] = ((-1) ** exponent) * norm
    return state


def single_state(n: int) -> np.ndarray:
    dim = 2 ** n
    if dim <= 10:
        raise ValueError("single_state requires 2**n > 10, e.g. n >= 4.")
    state = np.zeros(dim, dtype=np.float32)
    state[5] = 1.0 / np.sqrt(2.0)
    state[10] = 1.0 / np.sqrt(2.0)
    return state


def zero_state(n: int) -> np.ndarray:
    return np.zeros(2 ** n, dtype=np.float32)


def get_target_state(target_name: str, n: int) -> np.ndarray:
    target_name = target_name.upper()
    if target_name == "GHZ":
        return ghz_state(n)
    if target_name == "W":
        return w_state(n)
    if target_name in {"LINEAR_CLUSTER", "CLUSTER", "LINEAR"}:
        return linear_cluster_state(n)
    if target_name == "SINGLE":
        return single_state(n)
    if target_name == "ZERO":
        return zero_state(n)
    raise ValueError(f"Unknown target state: {target_name}")


# =============================================================================
# JSON / misc helpers
# =============================================================================


def scalar_k_value(k_value: Any) -> int:
    arr = np.asarray(k_value)
    if arr.ndim == 0:
        return int(arr)
    return int(arr.reshape(-1)[0])


# =============================================================================
# Data preparation
# =============================================================================


def generate_low_fidelity_dataset(cfg: Dict[str, Any], target_state: np.ndarray):
    """Generate initial samples in-memory (via data_generate.py's own
    generate_dataset(), save_data=False -- nothing written to disk, no
    entry added to Stage 1's dataset registry), keeping only samples whose
    fidelity to the target is below cfg['low_fidelity_threshold']."""
    n_samples = int(cfg["data_samples"])
    batch_size = int(cfg["data_batch_size"])
    threshold = float(cfg["low_fidelity_threshold"])
    seed = int(cfg["data_seed"])

    X_kept, Y_kept = [], []
    total_kept = 0
    total_generated = 0
    round_id = 0

    while total_kept < n_samples:
        round_id += 1
        round_seed = seed + round_id - 1
        remaining = n_samples - total_kept
        current_batch = max(batch_size, remaining)

        X_batch, Y_batch, _ = data_generate.generate_dataset(
            vertices=cfg["n"], dimensions=cfg.get("dimensions", 2), n_samples=current_batch,
            batch_size=cfg.get("generation_gpu_batch_size", current_batch), seed=round_seed,
            normed_data=cfg.get("normed_data", True), save_data=False,
        )

        fids = np.array([fidelity_np(y, target_state) for y in Y_batch], dtype=np.float32)
        keep_mask = fids < threshold
        X_good, Y_good = X_batch[keep_mask], Y_batch[keep_mask]

        if len(X_good) > 0:
            X_kept.append(X_good)
            Y_kept.append(Y_good)
            total_kept += len(X_good)
        total_generated += int(current_batch)

        print(f"round={round_id} | generated={current_batch} | kept={len(X_good)} | "
              f"discarded={(~keep_mask).sum()} | total_kept={total_kept}/{n_samples}")

    X_final = np.concatenate(X_kept, axis=0)[:n_samples].astype(np.float32)
    Y_final = np.concatenate(Y_kept, axis=0)[:n_samples].astype(np.float32)
    final_fids = np.array([fidelity_np(y, target_state) for y in Y_final], dtype=np.float32)

    print("\nFinal low-fidelity dataset check:")
    print("total generated =", total_generated)
    print("final dataset size =", len(X_final))
    print("max fidelity =", float(final_fids.max()))
    print("all fidelities < threshold =", bool(np.all(final_fids < threshold)))

    return X_final, Y_final


def load_model_training_info(model_path: str) -> dict:
    """Read ml_model.py's reproducibility_manifest.json from next to
    MODEL_PATH (same directory) and pull out its resolved_config/seeds --
    the training hyperparameters needed to retrain this exact model. That
    manifest already embeds Stage 1's own data-generation recipe (see
    ml_model.py's load_data_generation_info), so this closes the full
    chain: optimisation -> model -> training recipe -> data recipe, all
    reachable from just this run's own manifest."""
    manifest_path = Path(model_path).parent / "reproducibility_manifest.json"
    if not manifest_path.exists():
        return {"available": False, "manifest_path": str(manifest_path),
                "note": "No Stage-2 reproducibility_manifest.json found next to MODEL_PATH."}
    with open(manifest_path) as f:
        upstream = json.load(f)
    return {
        "available": True,
        "manifest_path": str(manifest_path),
        "resolved_config": upstream.get("resolved_config"),
        "seeds": upstream.get("seeds"),
        "data_generation": upstream.get("data_generation"),
        "git": upstream.get("git"),
        "environment": upstream.get("environment"),
    }


def prepare_input_data(cfg: Dict[str, Any]):
    """Returns [(k_value, X, Y)] -- fresh samples if cfg['generate_data'],
    else loaded from cfg['conditioned_data_path']."""
    target_state = get_target_state(cfg["target_name"], cfg["n"])

    if cfg.get("generate_data", False):
        X_low, Y_low = generate_low_fidelity_dataset(cfg, target_state)
        return [(0, X_low, Y_low)]

    data_path = Path(cfg["conditioned_data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Conditioned data path not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    if "weights" not in data or "amps" not in data:
        raise KeyError(f"Expected keys 'weights' and 'amps' in {data_path}. Found: {list(data.keys())}")

    X = data["weights"].astype(np.float32)
    Y = data["amps"].astype(np.float32)

    max_samples = cfg.get("max_initial_samples", None)
    if max_samples is not None:
        X = X[:max_samples]
        Y = Y[:max_samples]

    print("Loaded initial/conditioned dataset:")
    print("X:", X.shape, " Y:", Y.shape)
    return [(0, X, Y)]


# =============================================================================
# Optimisation functions
# =============================================================================


def normalize_vector_jax(y, eps=1e-12):
    norm = jnp.linalg.norm(y)
    norm = jnp.where(norm > eps, norm, 1.0)
    return y / norm


def build_optimizer(cfg: Dict[str, Any]):
    alpha = cfg["min_learning_rate"] / cfg["learning_rate"]
    lr_schedule = optax.cosine_decay_schedule(
        init_value=cfg["learning_rate"], decay_steps=cfg["lr_decay_steps"],
        alpha=alpha, exponent=cfg.get("lr_exponent", 1.0),
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    return optimizer, lr_schedule


def build_forward_metrics(apply_fn, params, cfg: Dict[str, Any]):
    """loss = 1 - fidelity + lambda_l1 * sum(abs(x)). Fidelity always computed
    from L2-normalised vectors so fid in [0,1] regardless of
    cfg['normalize_model_output'] (an unnormalised model output with
    ||y_pred||>1 could otherwise trigger a false early-stop)."""

    def _forward_metrics(x, y_target):
        y_pred = apply_fn(params, x)
        y_pred_norm = normalize_vector_jax(y_pred)
        y_target_norm = normalize_vector_jax(y_target)

        if cfg.get("normalize_model_output", True):
            y_pred_for_mae, y_target_for_mae = y_pred_norm, y_target_norm
        else:
            y_pred_for_mae, y_target_for_mae = y_pred, y_target

        l1 = cfg["lambda_l1"] * jnp.sum(jnp.abs(x))
        fid = jnp.abs(jnp.vdot(y_pred_norm, y_target_norm)) ** 2
        mae = jnp.mean(jnp.abs(jnp.abs(y_pred_for_mae) - jnp.abs(y_target_for_mae)))
        loss = (1.0 - fid) + l1
        return loss, y_pred, fid, mae, l1

    return _forward_metrics


def build_dream_step(forward_metrics, optimizer, clip_min: float = -1.0, clip_max: float = 1.0):
    @jax.jit
    def dream_step(x, opt_state, y_target):
        def loss_fn(z):
            loss, _, _, _, _ = forward_metrics(z, y_target)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(x)
        updates, opt_state = optimizer.update(grads, opt_state)
        x_updated = optax.apply_updates(x, updates)
        x_clipped = jnp.clip(x_updated, clip_min, clip_max)

        raw_update = updates
        applied_update_before_clip = x_updated - x
        applied_update_after_clip = x_clipped - x

        loss_new, y_pred, fid, mae, l1 = forward_metrics(x_clipped, y_target)
        return (x_clipped, opt_state, loss_new, y_pred, fid, mae, l1, grads,
                raw_update, applied_update_before_clip, applied_update_after_clip)

    return dream_step


def compute_grad_norm(grads) -> float:
    return float(np.linalg.norm(np.asarray(grads, dtype=np.float32)))


def is_exact_zero_vector(v) -> bool:
    return bool(np.all(np.asarray(v) == 0.0))


# =============================================================================
# Reproducible jitter (stall-detection + escalating re-perturbation)
# =============================================================================
#
# Root cause this fixes: the in-loop stop condition (nn_fid < early_stop_nn_fid)
# exits the instant surrogate fidelity crosses the threshold, leaving near-
# zero margin to absorb the real surrogate-vs-PyTheus noise gap -- this used
# to leave a meaningful fraction of samples below the target PyTheus fidelity
# even though the surrogate said "done." Jitter (this section) + measuring
# PyTheus fidelity BEFORE pruning (compute_pytheus_fidelity_before_pruning)
# are the validated mitigations. See utils.py for the underlying fold_in key
# plan (optimiser_jitter_root_key / optimiser_sample_key_for /
# optimiser_jitter_event_key_for).


def derive_sample_jitter_key(cfg: Dict[str, Any], sample_id: int):
    jitter_root = optimiser_jitter_root_key(int(cfg["seed"]))
    return optimiser_sample_key_for(jitter_root, sample_id)


def draw_jitter_noise(sample_key, event_index: int, shape):
    event_key = optimiser_jitter_event_key_for(sample_key, event_index)
    return jax.random.normal(event_key, shape=shape, dtype=jnp.float32)


def apply_initial_jitter(x0_np: np.ndarray, sample_key, sigma: float) -> np.ndarray:
    """sigma=0.0 is a no-op (returns x0_np unchanged), safe to call unconditionally."""
    if sigma <= 0.0:
        return np.asarray(x0_np, dtype=np.float32)
    noise = draw_jitter_noise(sample_key, event_index=0, shape=x0_np.shape)
    x_start = np.asarray(x0_np, dtype=np.float32) + sigma * np.asarray(noise, dtype=np.float32)
    return np.clip(x_start, -1.0, 1.0)


def maybe_apply_stall_jitter(
    *, x, nn_fid: float, target_nn_fid: float, sample_key, step: int,
    patience_counter: int, last_patience_fid: float, jitter_count: int, jitter_sched_idx: int,
    stuck_patience_steps: int, stuck_min_improvement: float, jitter_schedule: List[float],
    jitter_max_events: int, optimizer, forward_metrics, target_state_jax,
):
    """Once every stuck_patience_steps steps, check whether nn_fid improved
    by stuck_min_improvement; if not (and budget/threshold allow), inject
    reproducible noise (event_index = jitter_count + 1) at an escalating
    sigma, and reset the optimiser state."""
    patience_counter += 1
    log_entry = None

    if patience_counter >= stuck_patience_steps:
        if (nn_fid < last_patience_fid + stuck_min_improvement
                and jitter_count < jitter_max_events and nn_fid < target_nn_fid):
            sigma = jitter_schedule[min(jitter_sched_idx, len(jitter_schedule) - 1)]
            jitter_sched_idx += 1

            noise = draw_jitter_noise(sample_key, event_index=jitter_count + 1, shape=x.shape)
            x_np = np.asarray(jax.block_until_ready(x), dtype=np.float32)
            x_np = np.clip(x_np + sigma * np.asarray(noise, dtype=np.float32), -1.0, 1.0)
            x = jnp.asarray(x_np, dtype=jnp.float32)
            opt_state = optimizer.init(x)

            _, _, nn_fid_after, _, _ = forward_metrics(x, target_state_jax)
            nn_fid_after = float(nn_fid_after)

            log_entry = dict(step=int(step), event_index=int(jitter_count + 1), sigma=float(sigma),
                              nn_fid_before=float(nn_fid), nn_fid_after=nn_fid_after)

            nn_fid = nn_fid_after
            jitter_count += 1
            last_patience_fid = 0.0
        else:
            opt_state = None
            last_patience_fid = nn_fid
        patience_counter = 0
    else:
        opt_state = None

    return dict(x=x, opt_state=opt_state, nn_fid=nn_fid, patience_counter=patience_counter,
                last_patience_fid=last_patience_fid, jitter_count=jitter_count,
                jitter_sched_idx=jitter_sched_idx, jitter_log_entry=log_entry)


def compute_pytheus_fidelity_before_pruning(last_x, target_state_np, tensor_jax, mask_jax):
    """PyTheus-verified fidelity on the graph exactly as the optimiser left
    it, BEFORE any pruning is applied."""
    py_state_before = pytheus_state_from_x(last_x, tensor_jax, mask_jax)
    py_fid_before_pruning = fidelity_np(py_state_before, target_state_np)
    edges_before_pruning = int(np.sum(np.abs(last_x) > 1e-4))
    return py_state_before, float(py_fid_before_pruning), edges_before_pruning


def verify_with_pytheus_from_pred(x, step, loss, mae, l1, nn_state, nn_fid, y_target_np,
                                   write_fn, tensor_jax, mask_jax, grad_norm=0.0,
                                   update_norm=0.0, print_lines=True):
    x_np = np.asarray(x, dtype=np.float32).reshape(-1)
    nn_state_np = np.asarray(nn_state, dtype=np.float32).reshape(-1)
    y_target_np = np.asarray(y_target_np, dtype=np.float32).reshape(-1)

    py_state = np.asarray(pytheus_state_from_x(x_np, tensor_jax, mask_jax), dtype=np.float32).reshape(-1)

    py_fid = fidelity_np(py_state, y_target_np)
    py_nn = fidelity_np(nn_state_np, py_state)

    if print_lines:
        write_fn(f"[step {step:6d}] Loss = {float(loss):.8f} | MAE = {float(mae):.8f} | "
                  f"L1 = {float(l1):.8f} | NN fid(target) = {float(nn_fid):.6f} | "
                  f"Py fid(target) = {py_fid:.6f} | Py-NN fidelity = {py_nn:.6f} | "
                  f"grad_norm = {grad_norm:.8e} | update_norm = {update_norm:.8e}")

    return float(nn_fid), py_fid, nn_state_np, py_state


def progressive_threshold_prune(x_best, y_target_np, write, apply_fn, params, tensor_jax, mask_jax,
                                 normalize_model_output=True, fid_tol=1e-4, thresholds=None):
    if thresholds is None:
        thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    thresholds = list(thresholds)

    x_current = np.array(x_best, dtype=np.float32)
    nn_state = np.asarray(apply_fn(params, jnp.asarray(x_current, dtype=jnp.float32)), dtype=np.float32)
    if normalize_model_output:
        nn_norm = np.linalg.norm(nn_state)
        if nn_norm > 1e-12:
            nn_state = nn_state / nn_norm

    fid_best = fidelity_np(nn_state, y_target_np)
    pruning_info = []

    write("\n=== Starting threshold pruning ===")
    write(f"Initial fidelity: {fid_best}")

    for threshold in thresholds:
        x_new = x_current.copy()
        x_new[np.abs(x_new) < threshold] = 0.0

        nn_state_new = np.asarray(apply_fn(params, jnp.asarray(x_new, dtype=jnp.float32)), dtype=np.float32)
        if normalize_model_output:
            nn_norm = np.linalg.norm(nn_state_new)
            if nn_norm > 1e-12:
                nn_state_new = nn_state_new / nn_norm

        fid_new = fidelity_np(nn_state_new, y_target_np)
        entry = {"threshold": float(threshold), "fidelity_before": float(fid_best),
                 "fidelity_after": float(fid_new),
                 "num_weights_zeroed": int(np.sum(np.abs(x_current) < threshold))}

        if abs(fid_new - fid_best) <= fid_tol:
            write(f"Threshold {threshold} accepted | fid = {fid_new}")
            x_current, fid_best, nn_state = x_new, fid_new, nn_state_new
            entry["accepted"] = True
        else:
            write(f"Threshold {threshold} rejected | fid dropped to {fid_new}")
            entry["accepted"] = False
            pruning_info.append(entry)
            break
        pruning_info.append(entry)

    py_state = pytheus_state_from_x(x_current, tensor_jax, mask_jax)
    write("Final pruned graph:")
    write(x_current)
    write("Final quantum state by neural network:")
    write(nn_state)
    write("Final quantum state by PyTheus:")
    write(py_state)
    write("Number of active weights:")
    write(int(np.sum(np.abs(x_current) > 1e-4)))

    return x_current, nn_state, fid_best, pruning_info


# =============================================================================
# Plotting
# =============================================================================


def save_gradient_plot(sample_dir: Path, grad_norm_history: List[float]) -> None:
    steps = np.arange(len(grad_norm_history))
    plt.figure()
    plt.plot(steps, grad_norm_history)
    plt.xlabel("step")
    plt.ylabel("|| gradient ||")
    plt.title("Gradient norm")
    plt.savefig(sample_dir / "gradient_norm.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_loss_fid_plot(sample_dir, loss_history, fid_history, target_name, nphotons):
    steps = np.arange(len(loss_history))
    fig, ax1 = plt.subplots()
    ax1.plot(steps, loss_history, label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(steps, fid_history, label="fidelity")
    ax2.set_ylabel("fidelity")
    ax2.set_ylim(0, 1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title(f"Inverse design convergence: {target_name} (n={nphotons})")
    fig.tight_layout()
    fig.savefig(sample_dir / "loss_fid_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_edge_fidelity_plot(pt, edge_list, fid_list, cfg):
    if len(edge_list) == 0:
        return
    edges_np, fid_np = np.array(edge_list), np.array(fid_list)
    unique_edges = np.unique(edges_np)
    counts, avg_fid = [], []
    for edge_count in unique_edges:
        m = edges_np == edge_count
        counts.append(np.sum(m))
        avg_fid.append(np.mean(fid_np[m]))
    counts, avg_fid = np.array(counts), np.array(avg_fid)
    sort_idx = np.argsort(unique_edges)[::-1]
    unique_edges, counts, avg_fid = unique_edges[sort_idx], counts[sort_idx], avg_fid[sort_idx]

    fig, ax1 = plt.subplots()
    ax1.bar(unique_edges, counts, width=0.25)
    ax1.set_xlabel("Number of active weights")
    ax1.set_ylabel("Number of samples")
    ax1.set_xlim(max(unique_edges) + 0.5, min(unique_edges) - 0.5)
    ax2 = ax1.twinx()
    ax2.plot(unique_edges, avg_fid, marker="o")
    ax2.set_ylabel("Average PyTheus fidelity")
    ax2.set_ylim(0, 1)
    plt.title(f"Sparsity vs fidelity, n={cfg['n']} | L1={cfg['lambda_l1']}")
    fig.tight_layout()
    fig.savefig(pt / "edge_vs_fidelity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_time_plots(pt, time_per_sample_list, n):
    if len(time_per_sample_list) == 0:
        return
    times = np.array(time_per_sample_list)
    sample_ids = np.arange(len(times))

    fig, ax = plt.subplots()
    ax.plot(sample_ids, times, linewidth=2, marker=".")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Time (seconds)")
    plt.title(f"Time taken per sample, nodes={n}")
    fig.tight_layout()
    fig.savefig(pt / "time_per_sample.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cumulative_time = np.cumsum(times)
    fig, ax = plt.subplots()
    ax.plot(sample_ids, cumulative_time, linewidth=2)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Total runtime (seconds)")
    plt.title(f"Cumulative optimisation time, nodes={n}")
    fig.tight_layout()
    fig.savefig(pt / "cumulative_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================


def _next_run_dir(results_root: str, target_name: str, nphotons: int) -> Path:
    parent = Path(results_root) / f"{target_name}_n{nphotons}"
    parent.mkdir(parents=True, exist_ok=True)
    i = 0
    while (parent / f"{target_name}_n{nphotons}_{i}").exists():
        i += 1
    return parent / f"{target_name}_n{nphotons}_{i}"


def _default_cfg() -> dict:
    return dict(
        n=NPHOTONS, dimensions=2, target_name=TARGET_NAME, model_type=MODEL_TYPE,
        generate_data=GENERATE_DATA, conditioned_data_path=CONDITIONED_DATA_PATH,
        max_initial_samples=MAX_INITIAL_SAMPLES,
        data_samples=DATA_SAMPLES, data_batch_size=DATA_BATCH_SIZE, data_seed=DATA_SEED,
        low_fidelity_threshold=LOW_FIDELITY_THRESHOLD, normed_data=NORMED_DATA,
        generation_gpu_batch_size=DATA_BATCH_SIZE, data_shard_size=DATA_BATCH_SIZE,
        architecture=ARCHITECTURE, model_path=MODEL_PATH, normalize_model_output=NORMALIZE_MODEL_OUTPUT,
        input_dim=INPUT_DIM, out_dim=OUT_DIM, lambda_l1=LAMBDA_L1,
        seed=SEED, num_steps=NUM_STEPS, early_stop_nn_fid=EARLY_STOP_NN_FID,
        learning_rate=LEARNING_RATE, min_learning_rate=MIN_LEARNING_RATE,
        lr_decay_steps=LR_DECAY_STEPS, lr_exponent=LR_EXPONENT,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX,
        jitter_enabled=JITTER_ENABLED, initial_jitter=INITIAL_JITTER,
        jitter_schedule=JITTER_SCHEDULE, jitter_max_events=JITTER_MAX_EVENTS,
        stuck_patience_steps=STUCK_PATIENCE_STEPS, stuck_min_improvement=STUCK_MIN_IMPROVEMENT,
        max_total_steps=MAX_TOTAL_STEPS,
        print_every=PRINT_EVERY, verify_every=VERIFY_EVERY, store_step_vectors=STORE_STEP_VECTORS,
        prune_fid_tolerance=PRUNE_FID_TOLERANCE, prune_thresholds=PRUNE_THRESHOLDS,
        results_root=RESULTS_ROOT,
    )


def run_optimisation(cfg: dict | None = None):
    """Run inverse-design optimisation. cfg overrides this file's top-of-file
    constants; pass None (default) to use them directly -- no external
    config file is read either way."""
    cfg = dict(_default_cfg()) if cfg is None else dict(cfg)

    target_state_np = get_target_state(cfg["target_name"], cfg["n"])
    target_state_jax = jnp.asarray(target_state_np, dtype=jnp.float32)

    data = prepare_input_data(cfg)

    edges, kets, tensor, mask = build_pytheus_catalog(cfg["n"])
    tensor_jax = jnp.asarray(tensor, dtype=jnp.int32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    apply_fn, params = load_trained_model(
        model_type=cfg["model_type"], architecture=cfg["architecture"], input_dim=cfg["input_dim"],
        out_dim=cfg["out_dim"], nodes=cfg["n"], model_path=cfg["model_path"],
    )

    optimizer, lr_schedule = build_optimizer(cfg)
    forward_metrics = build_forward_metrics(apply_fn, params, cfg)
    dream_step = build_dream_step(forward_metrics, optimizer,
                                   clip_min=cfg.get("clip_min", -1.0), clip_max=cfg.get("clip_max", 1.0))

    output_dir = _next_run_dir(cfg["results_root"], cfg["target_name"], cfg["n"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "cfg.json", cfg)

    log_fh = (output_dir / "log.txt").open("w", buffering=1)

    def write(*args, sep=" ", end="\n"):
        log_fh.write(sep.join(str(a) for a in args) + end)

    write("=== Inverse optimisation started ===")
    write(f"JAX devices: {jax.devices()}")
    write(f"Target: {cfg['target_name']}, n={cfg['n']}")
    write(f"Model path: {cfg['model_path']}")
    write(f"Output directory: {output_dir}")

    zero_state_samples = []
    best_solution = {"sample_id": None, "num_edges": np.inf, "graph_vector": None,
                      "nn_state": None, "pytheus_state": None, "fidelity": None}

    initial_fidelities, final_fidelities, final_pytheus_fidelities = [], [], []
    time_per_sample_list, optimisation_time_list, avg_time_per_step_list, optimisation_steps_list = [], [], [], []
    edge_list, py_fid_list = [], []
    py_fid_before_pruning_list, edges_before_pruning_list, jitter_count_list = [], [], []

    store_step_vectors = bool(cfg.get("store_step_vectors", True))

    jitter_enabled = bool(cfg.get("jitter_enabled", False))
    initial_jitter_sigma = float(cfg.get("initial_jitter", 0.0)) if jitter_enabled else 0.0
    jitter_schedule = list(cfg.get(
        "jitter_schedule", [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.00, 1.00, 1.00, 1.00],
    ))
    jitter_max_events = int(cfg.get("jitter_max_events", 12)) if jitter_enabled else 0
    stuck_patience_steps = int(cfg.get("stuck_patience_steps", 500))
    stuck_min_improvement = float(cfg.get("stuck_min_improvement", 1e-4))
    max_total_steps = int(cfg.get("max_total_steps", cfg["num_steps"]))

    try:
        for k_value, dataset_X, dataset_Y in data:
            write(f"\n=== DATASET k={k_value} ===")

            for run_id, x0_np in enumerate(dataset_X):
                write(f"\n===== SAMPLE {run_id} =====")
                sample_start = time.perf_counter()

                x0_jax = jnp.asarray(x0_np, dtype=jnp.float32)

                sample_key = derive_sample_jitter_key(cfg, run_id)
                x_start_np = apply_initial_jitter(x0_np, sample_key, initial_jitter_sigma)
                x = jnp.asarray(x_start_np, dtype=jnp.float32)
                opt_state = optimizer.init(x)

                loss_history, fid_history, mae_history, l1_history = [], [], [], []
                update_norm_history, grad_norm_history, x_history, lr_history = [], [], [], []
                grad_vector_history, raw_update_history = [], []
                update_before_clip_history, update_after_clip_history = [], []

                jitter_log = []
                jitter_count = jitter_sched_idx = patience_counter = 0
                last_patience_fid = 0.0

                step = 0
                init_loss, init_pred, init_fid, init_mae, init_l1 = forward_metrics(x0_jax, target_state_jax)
                init_loss = jax.block_until_ready(init_loss)
                init_pred = jax.block_until_ready(init_pred)
                init_fid = jax.block_until_ready(init_fid)
                init_mae = jax.block_until_ready(init_mae)
                init_l1 = jax.block_until_ready(init_l1)

                nn_fid, py_fid, nn_state, py_state = verify_with_pytheus_from_pred(
                    x=x0_jax, step=step, loss=init_loss, mae=init_mae, l1=init_l1,
                    nn_state=init_pred, nn_fid=init_fid, y_target_np=target_state_np,
                    write_fn=write, tensor_jax=tensor_jax, mask_jax=mask_jax,
                )
                initial_fidelities.append(float(nn_fid))

                sample_info = {
                    "k_value": scalar_k_value(k_value), "sample_id": int(run_id),
                    "x_init": np.asarray(x0_np, dtype=np.float32), "x_start": x_start_np,
                    "y_target": target_state_np, "init_loss": float(init_loss),
                    "init_mae": float(init_mae), "init_l1": float(init_l1),
                    "nn_state_init": nn_state, "py_state_init": py_state,
                    "nn_fid_init": float(nn_fid), "py_fid_init": float(py_fid),
                    "sample_time": 0.0, "optimisation_time": 0.0,
                    "avg_time_per_step": 0.0, "optimisation_steps": 0,
                }

                _, _, nn_fid_start_jax, _, _ = forward_metrics(x, target_state_jax)
                nn_fid = float(jax.block_until_ready(nn_fid_start_jax))

                last_x = x_start_np
                optim_start = time.perf_counter()
                step = 0

                while step < max_total_steps and nn_fid < cfg["early_stop_nn_fid"]:
                    step += 1
                    current_lr = float(lr_schedule(step - 1))
                    lr_history.append(current_lr)

                    x_before = np.asarray(x, dtype=np.float32)
                    (x, opt_state, loss_p, y_pred, nn_fid_jax, mae_p, l1_p, grads_jax,
                     raw_update_jax, update_before_clip_jax, update_after_clip_jax) = dream_step(
                        x, opt_state, target_state_jax)

                    loss_p = jax.block_until_ready(loss_p)
                    y_pred = jax.block_until_ready(y_pred)
                    nn_fid_jax = jax.block_until_ready(nn_fid_jax)
                    mae_p = jax.block_until_ready(mae_p)
                    l1_p = jax.block_until_ready(l1_p)
                    x = jax.block_until_ready(x)
                    grads_jax = jax.block_until_ready(grads_jax)
                    raw_update_jax = jax.block_until_ready(raw_update_jax)
                    update_before_clip_jax = jax.block_until_ready(update_before_clip_jax)
                    update_after_clip_jax = jax.block_until_ready(update_after_clip_jax)

                    x_after = np.asarray(x, dtype=np.float32)
                    update_norm = float(np.linalg.norm(x_after - x_before))
                    grad_norm = compute_grad_norm(grads_jax)

                    grad_norm_history.append(grad_norm)
                    update_norm_history.append(update_norm)
                    loss_history.append(float(loss_p))
                    mae_history.append(float(mae_p))
                    l1_history.append(float(l1_p))
                    fid_history.append(float(nn_fid_jax))

                    if store_step_vectors:
                        grad_vector_history.append(np.asarray(grads_jax, dtype=np.float32).copy())
                        raw_update_history.append(np.asarray(raw_update_jax, dtype=np.float32).copy())
                        update_before_clip_history.append(np.asarray(update_before_clip_jax, dtype=np.float32).copy())
                        update_after_clip_history.append(np.asarray(update_after_clip_jax, dtype=np.float32).copy())

                    if step % cfg["verify_every"] == 0 or step == max_total_steps:
                        nn_fid, py_fid, nn_state, py_state = verify_with_pytheus_from_pred(
                            x=x, step=step, loss=loss_p, mae=mae_p, l1=l1_p, nn_state=y_pred,
                            nn_fid=nn_fid_jax, y_target_np=target_state_np, write_fn=write,
                            tensor_jax=tensor_jax, mask_jax=mask_jax, grad_norm=grad_norm,
                            update_norm=update_norm, print_lines=(step % cfg["print_every"] == 0),
                        )
                    else:
                        nn_fid = float(nn_fid_jax)

                    if jitter_enabled:
                        jr = maybe_apply_stall_jitter(
                            x=x, nn_fid=nn_fid, target_nn_fid=cfg["early_stop_nn_fid"],
                            sample_key=sample_key, step=step, patience_counter=patience_counter,
                            last_patience_fid=last_patience_fid, jitter_count=jitter_count,
                            jitter_sched_idx=jitter_sched_idx, stuck_patience_steps=stuck_patience_steps,
                            stuck_min_improvement=stuck_min_improvement, jitter_schedule=jitter_schedule,
                            jitter_max_events=jitter_max_events, optimizer=optimizer,
                            forward_metrics=forward_metrics, target_state_jax=target_state_jax,
                        )
                        x = jr["x"]
                        if jr["opt_state"] is not None:
                            opt_state = jr["opt_state"]
                        nn_fid = jr["nn_fid"]
                        patience_counter = jr["patience_counter"]
                        last_patience_fid = jr["last_patience_fid"]
                        jitter_count = jr["jitter_count"]
                        jitter_sched_idx = jr["jitter_sched_idx"]
                        if jr["jitter_log_entry"] is not None:
                            jitter_log.append(jr["jitter_log_entry"])

                    last_x = np.asarray(x, dtype=np.float32)
                    x_history.append(last_x.copy())

                _, py_fid_before_pruning, edges_before_pruning = compute_pytheus_fidelity_before_pruning(
                    last_x, target_state_np, tensor_jax, mask_jax,
                )
                write(f"steps={step} nn_fid_at_exit={nn_fid:.6f} "
                      f"py_fid_before_pruning={py_fid_before_pruning:.6f}")

                py_fid_before_pruning_list.append(float(py_fid_before_pruning))
                edges_before_pruning_list.append(int(edges_before_pruning))
                jitter_count_list.append(int(jitter_count))

                x_pruned, nn_pruned, _, pruning_info = progressive_threshold_prune(
                    x_best=last_x, y_target_np=target_state_np, write=write, apply_fn=apply_fn,
                    params=params, tensor_jax=tensor_jax, mask_jax=mask_jax,
                    normalize_model_output=cfg.get("normalize_model_output", True),
                    fid_tol=cfg.get("prune_fid_tolerance", 1e-4),
                    thresholds=cfg.get("prune_thresholds", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
                )

                x_history.append(np.asarray(x_pruned, dtype=np.float32).copy())
                jax.block_until_ready(jnp.asarray(x))
                optimisation_time = time.perf_counter() - optim_start
                avg_time_per_step = optimisation_time / max(step, 1)

                final_edges = int(np.sum(np.abs(x_pruned) > 1e-4))
                final_nn_state = np.asarray(nn_pruned, dtype=np.float32)
                final_py_state = pytheus_state_from_x(x_pruned, tensor_jax, mask_jax)

                py_fid_pruned = fidelity_np(final_py_state, target_state_np)
                nn_fid_pruned = fidelity_np(final_nn_state, target_state_np)

                edge_list.append(final_edges)
                py_fid_list.append(float(py_fid_pruned))
                final_fidelities.append(float(nn_fid_pruned))
                final_pytheus_fidelities.append(float(py_fid_pruned))
                optimisation_time_list.append(float(optimisation_time))
                avg_time_per_step_list.append(float(avg_time_per_step))
                optimisation_steps_list.append(int(step))

                if final_edges < best_solution["num_edges"]:
                    best_solution.update({
                        "sample_id": int(run_id), "num_edges": int(final_edges),
                        "graph_vector": np.asarray(x_pruned, dtype=np.float32).tolist(),
                        "nn_state": final_nn_state.tolist(),
                        "pytheus_state": np.asarray(final_py_state, dtype=np.float32).tolist(),
                        "fidelity": float(py_fid_pruned),
                    })
                    write("NEW BEST GRAPH FOUND")
                    write(f"sample = {run_id}")

                if is_exact_zero_vector(final_nn_state):
                    zero_state_samples.append({
                        "sample_id": int(run_id), "graph_vector": np.asarray(x_pruned, dtype=np.float32).tolist(),
                        "nn_state": final_nn_state.tolist(),
                        "pytheus_state": np.asarray(final_py_state, dtype=np.float32).tolist(),
                        "nn_norm": float(np.linalg.norm(final_nn_state)),
                        "py_norm": float(np.linalg.norm(final_py_state)), "optimisation_steps": int(step),
                    })
                    write("ZERO VECTOR FOUND")
                    write("sample:", run_id)
                    write("graph:", x_pruned)

                sample_dir = output_dir / f"sample_{run_id}"
                sample_dir.mkdir(exist_ok=True)
                abs_error = np.abs(final_nn_state - target_state_np)

                save_loss_fid_plot(sample_dir, loss_history, fid_history, cfg["target_name"], cfg["n"])
                save_gradient_plot(sample_dir, grad_norm_history)

                sample_time = time.perf_counter() - sample_start
                time_per_sample_list.append(float(sample_time))

                sample_info.update({
                    "x_before_pruning": last_x, "py_fid_before_pruning": float(py_fid_before_pruning),
                    "edges_before_pruning": int(edges_before_pruning), "jitter_log": jitter_log,
                    "jitter_count": int(jitter_count), "x_final": np.asarray(x_pruned, dtype=np.float32),
                    "nn_state_final": final_nn_state, "py_state_final": final_py_state,
                    "loss_history": loss_history, "fid_history": fid_history, "mae_history": mae_history,
                    "l1_history": l1_history, "grad_norm_history": grad_norm_history,
                    "update_norm_history": update_norm_history, "pruning_log": pruning_info,
                    "abs_error": abs_error, "final_edges": int(final_edges),
                    "nn_fid_final": float(nn_fid_pruned), "py_fid_final": float(py_fid_pruned),
                    "sample_time": float(sample_time), "optimisation_time": float(optimisation_time),
                    "avg_time_per_step": float(avg_time_per_step), "optimisation_steps": int(step),
                    "x_history": x_history, "lr_history": lr_history,
                    "grad_vector_history": grad_vector_history, "raw_update_history": raw_update_history,
                    "update_before_clip_history": update_before_clip_history,
                    "update_after_clip_history": update_after_clip_history,
                })
                save_json(sample_dir / "sample_info_file.json", sample_info)

        save_edge_fidelity_plot(output_dir, edge_list, py_fid_list, cfg)
        save_time_plots(output_dir, time_per_sample_list, cfg["n"])
        save_json(output_dir / "zero_state_samples.json", zero_state_samples)

        summary_data = {
            "initial_fidelities": [float(x) for x in initial_fidelities],
            "final_fidelities": [float(x) for x in final_fidelities],
            "final_pytheus_fidelities": [float(x) for x in final_pytheus_fidelities],
            "pytheus_fidelity_before_pruning": [float(x) for x in py_fid_before_pruning_list],
            "edges_before_pruning": [int(x) for x in edges_before_pruning_list],
            "jitter_counts": [int(x) for x in jitter_count_list],
            "time_per_sample": [float(x) for x in time_per_sample_list],
            "optimisation_time_per_sample": [float(x) for x in optimisation_time_list],
            "avg_time_per_step": [float(x) for x in avg_time_per_step_list],
            "optimisation_steps": [int(x) for x in optimisation_steps_list],
            "final_edges_per_sample": [int(x) for x in edge_list],
        }
        save_json(output_dir / "optimisation_summary.json", summary_data)
        save_json(output_dir / "best_graph_solution.json", best_solution)

        model_training_info = load_model_training_info(cfg["model_path"])
        manifest = build_manifest(
            resolved_config=cfg,
            seeds={"seed": int(cfg["seed"]), "data_seed": int(cfg["data_seed"])},
            input_file_info=hash_file(cfg["model_path"]),
            repo_root=REPO_ROOT,
            extra={"model_training": model_training_info},
        )
        write_manifest(output_dir / "reproducibility_manifest.json", manifest)

    finally:
        log_fh.close()

    return output_dir


if __name__ == "__main__":
    output_dir = run_optimisation()
    print(f"Optimisation complete. Results saved in: {output_dir}")
