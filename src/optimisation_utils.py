"""
Utility functions for surrogate inverse optimisation.

This file keeps the optimisation helpers out of optimiser.py so the main script
stays readable while preserving the original workflow:
- load or generate initial samples
- load trained surrogate model
- optimise graph weights with gradient descent
- verify with PyTheus
- prune weak weights
- save plots and JSON summaries
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.serialization import from_bytes
from pytheus import theseus as th

from data_generation import generate_dataset
from models import create_model
from target_states import get_target_state


# =============================================================================
# JSON / logging helpers
# =============================================================================


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, jax.Array):
        return np.asarray(obj).tolist()
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, (jnp.floating, np.floating)):
        return float(obj)
    if isinstance(obj, (jnp.integer, np.integer)):
        return int(obj)
    return obj


def save_json(path: Path, data: Any) -> None:
    path = Path(path)
    with path.open("w") as f:
        json.dump(to_jsonable(data), f, indent=2)


def scalar_k_value(k_value: Any) -> int:
    arr = np.asarray(k_value)
    if arr.ndim == 0:
        return int(arr)
    return int(arr.reshape(-1)[0])


# =============================================================================
# Model loading
# =============================================================================


def load_trained_model(cfg: Dict[str, Any]):
    """
    Load a trained Flax model from params.msgpack.

    Expected cfg keys:
        model_type, architecture, input_dim, out_dim, n, model_path

    Returns:
        apply_fn, params
    """
    model = create_model(
        model_name=cfg["model_type"],
        hidden_dims=cfg["architecture"],
        out_dims=cfg["out_dim"],
        nodes=cfg["n"],
    )

    x_example = jnp.zeros((1, cfg["input_dim"]), dtype=jnp.float32)
    params_template = model.init(jax.random.PRNGKey(0), x_example)

    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model params file not found: {model_path}")

    with model_path.open("rb") as f:
        params = from_bytes(params_template, f.read())

    return model.apply, params


# =============================================================================
# Data preparation
# =============================================================================


def fidelity_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.size != b.size:
        raise ValueError(f"State size mismatch: {a.size} vs {b.size}")

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    a = a / na
    b = b / nb

    return float(np.abs(np.vdot(a, b)) ** 2)


def generate_low_fidelity_dataset(
    cfg: Dict[str, Any],
    target_state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate initial samples by calling generate_dataset() and keeping only
    samples whose fidelity to the target is below cfg['low_fidelity_threshold'].

    This uses the cleaned data_generation.py function. It supports both return
    styles:
        (X, Y)
        (X, Y, levels)
    """
    n_samples = int(cfg["data_samples"])
    batch_size = int(cfg["data_batch_size"])
    threshold = float(cfg["low_fidelity_threshold"])
    seed = int(cfg["data_seed"])

    X_kept: List[np.ndarray] = []
    Y_kept: List[np.ndarray] = []
    levels_kept: List[np.ndarray] = []

    total_kept = 0
    total_generated = 0
    round_id = 0

    while total_kept < n_samples:
        round_id += 1
        round_seed = seed + round_id - 1
        remaining = n_samples - total_kept
        current_batch = max(batch_size, remaining)

        out = generate_dataset(
            vertices=cfg["n"],
            dimensions=cfg.get("dimensions", 2),
            n_samples=current_batch,
            batch_size=cfg.get("generation_gpu_batch_size", current_batch),
            seed=round_seed,
            save_to_file=False,
            out_dir_path=cfg.get("data_out_dir", "data/optimiser_tmp"),
            shard_size=cfg.get("data_shard_size", current_batch),
            normed_data=cfg.get("normed_data", True),
        )

        if isinstance(out, tuple) and len(out) == 3:
            X_batch, Y_batch, levels_batch = out
        elif isinstance(out, tuple) and len(out) == 2:
            X_batch, Y_batch = out
            levels_batch = np.full((len(X_batch),), 5, dtype=np.int8)
        else:
            raise RuntimeError(
                "generate_dataset(save_to_file=False) must return (X, Y) or (X, Y, levels)."
            )

        fids = np.array(
            [fidelity_np(y, target_state) for y in Y_batch],
            dtype=np.float32,
        )

        keep_mask = fids < threshold

        X_good = X_batch[keep_mask]
        Y_good = Y_batch[keep_mask]
        levels_good = levels_batch[keep_mask]

        if len(X_good) > 0:
            X_kept.append(X_good)
            Y_kept.append(Y_good)
            levels_kept.append(levels_good)
            total_kept += len(X_good)

        total_generated += int(current_batch)

        print(
            f"round={round_id} | generated={current_batch} | kept={len(X_good)} | "
            f"discarded={(~keep_mask).sum()} | total_kept={total_kept}/{n_samples}"
        )

    X_final = np.concatenate(X_kept, axis=0)[:n_samples].astype(np.float32)
    Y_final = np.concatenate(Y_kept, axis=0)[:n_samples].astype(np.float32)
    levels_final = np.concatenate(levels_kept, axis=0)[:n_samples]

    final_fids = np.array(
        [fidelity_np(y, target_state) for y in Y_final],
        dtype=np.float32,
    )

    print("\nFinal low-fidelity dataset check:")
    print("total generated =", total_generated)
    print("final dataset size =", len(X_final))
    print("max fidelity =", float(final_fids.max()))
    print("all fidelities < threshold =", bool(np.all(final_fids < threshold)))

    return X_final, Y_final, levels_final, final_fids


def prepare_input_data(cfg: Dict[str, Any]):
    """
    Prepare initial graph samples for inverse optimisation.

    Modes:
    1. cfg['generate_data'] = True
       Generate n samples using data_generation.generate_dataset(), then keep
       low-fidelity samples.

    2. cfg['generate_data'] = False
       Load existing .npz file with keys:
           weights, amps
       optional:
           levels

    Returns:
        list of tuples: [(levels, X, Y)]
    """
    target_state = get_target_state(cfg["target_name"], cfg["n"])

    if cfg.get("generate_data", False):
        X_low, Y_low, levels_low, _ = generate_low_fidelity_dataset(cfg, target_state)
        return [(levels_low, X_low, Y_low)]

    data_path = Path(cfg["conditioned_data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Conditioned data path not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    if "weights" not in data or "amps" not in data:
        raise KeyError(
            f"Expected keys 'weights' and 'amps' in {data_path}. Found: {list(data.keys())}"
        )

    X = data["weights"].astype(np.float32)
    Y = data["amps"].astype(np.float32)

    if "levels" in data:
        levels = data["levels"]
    else:
        levels = np.full((len(X),), 99, dtype=np.int8)

    max_samples = cfg.get("max_initial_samples", None)
    if max_samples is not None:
        X = X[:max_samples]
        Y = Y[:max_samples]
        levels = levels[:max_samples]

    print("Loaded initial/conditioned dataset:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("levels:", levels.shape)

    return [(levels, X, Y)]


# =============================================================================
# PyTheus verification
# =============================================================================


def setup_pytheus_catalog(n: int):
    edges = th.buildAllEdges(
        dimensions=[2] * n,
        string=False,
        imaginary=False,
        loops=False,
    )

    pms = th.findPerfectMatchings(edges)
    catalog = th.stateCatalog(pms)

    edge_to_idx = {e: i for i, e in enumerate(edges)}
    kets = sorted(catalog.keys())
    n_kets = len(kets)

    max_m = max(len(catalog[k]) for k in kets)
    epm = len(catalog[kets[0]][0])
    sentinel = len(edges)

    tensor = np.full((n_kets, max_m, epm), sentinel, dtype=np.int32)
    mask = np.zeros((n_kets, max_m), dtype=np.float32)

    for i, k in enumerate(kets):
        for j, pm in enumerate(catalog[k]):
            mask[i, j] = 1.0
            for l, e in enumerate(pm):
                tensor[i, j, l] = edge_to_idx[e]

    return edges, tensor, mask


@jax.jit
def compute_quantum_states(X, tensor, mask):
    w_ext = jnp.concatenate(
        [X, jnp.ones((X.shape[0], 1), dtype=jnp.float32)],
        axis=1,
    )
    gathered = w_ext[:, tensor]
    prods = jnp.prod(gathered, axis=-1)
    amps = jnp.sum(prods * mask[None, :, :], axis=-1)

    norms = jnp.linalg.norm(amps, axis=-1, keepdims=True)
    norms = jnp.where(norms > 0, norms, 1.0)

    return amps / norms


def pytheus_state_from_x(x_vec, tensor_jax, mask_jax):
    x_vec = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)

    amps = compute_quantum_states(
        jnp.array(x_vec, dtype=jnp.float32),
        tensor_jax,
        mask_jax,
    )

    amps = jax.block_until_ready(amps)
    return np.asarray(amps[0], dtype=np.float32), None


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
        init_value=cfg["learning_rate"],
        decay_steps=cfg["lr_decay_steps"],
        alpha=alpha,
        exponent=cfg.get("lr_exponent", 1.0),
    )

    optimizer = optax.adam(learning_rate=lr_schedule)
    return optimizer, lr_schedule


def build_forward_metrics(apply_fn, params, cfg: Dict[str, Any]):
    """
    Build forward metric function for inverse optimisation.

    Preserved objective:
        loss = 1 - fidelity + lambda_l1 * sum(abs(x))

    If cfg['normalize_model_output'] is True, the surrogate prediction is
    normalized before fidelity/MAE computation.
    """

    def _forward_metrics(x, y_target):
        y_pred = apply_fn(params, x)

        if cfg.get("normalize_model_output", True):
            y_pred = normalize_vector_jax(y_pred)
            y_target_used = normalize_vector_jax(y_target)
        else:
            y_target_used = y_target

        l1 = cfg["lambda_l1"] * jnp.sum(jnp.abs(x))
        fid = jnp.abs(jnp.vdot(y_pred, y_target_used)) ** 2
        mae = jnp.mean(jnp.abs(jnp.abs(y_pred) - jnp.abs(y_target_used)))
        loss = (1.0 - fid) + l1

        return loss, y_pred, fid, mae, l1

    return _forward_metrics


def build_dream_step(forward_metrics, optimizer, clip_min: float = -1.0, clip_max: float = 1.0):
    """
    One JIT-compiled optimiser step on the graph vector x.

    clip_min and clip_max control the allowed graph-weight range after each
    update. They are passed from optimiser_config.py through optimiser.py.
    """

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

        return (
            x_clipped,
            opt_state,
            loss_new,
            y_pred,
            fid,
            mae,
            l1,
            grads,
            raw_update,
            applied_update_before_clip,
            applied_update_after_clip,
        )

    return dream_step


def compute_grad_norm(grads) -> float:
    grads = np.asarray(grads, dtype=np.float32)
    return float(np.linalg.norm(grads))


def is_exact_zero_vector(v) -> bool:
    v = np.asarray(v)
    return bool(np.all(v == 0.0))


def verify_with_pytheus_from_pred(
    x,
    step: int,
    loss,
    mae,
    l1,
    nn_state,
    nn_fid,
    y_target_np,
    write_fn: Callable[..., None],
    tensor_jax,
    mask_jax,
    grad_norm: float = 0.0,
    update_norm: float = 0.0,
    print_lines: bool = True,
):
    x_np = np.asarray(x, dtype=np.float32).reshape(-1)
    nn_state_np = np.asarray(nn_state, dtype=np.float32).reshape(-1)
    y_target_np = np.asarray(y_target_np, dtype=np.float32).reshape(-1)

    py_state, _ = pytheus_state_from_x(x_np, tensor_jax, mask_jax)
    py_state = np.asarray(py_state, dtype=np.float32).reshape(-1)

    py_fid = fidelity_np(py_state, y_target_np)
    py_nn = fidelity_np(nn_state_np, py_state)

    if print_lines:
        write_fn(
            f"[step {step:6d}] "
            f"Loss = {float(loss):.8f} | "
            f"MAE = {float(mae):.8f} | "
            f"L1 = {float(l1):.8f} | "
            f"NN fid(target) = {float(nn_fid):.6f} | "
            f"Py fid(target) = {py_fid:.6f} | "
            f"Py-NN fidelity = {py_nn:.6f} | "
            f"grad_norm = {grad_norm:.8e} | "
            f"update_norm = {update_norm:.8e}"
        )

    return float(nn_fid), py_fid, nn_state_np, py_state


def progressive_threshold_prune(
    x_best,
    y_target_np,
    write,
    apply_fn,
    params,
    tensor_jax,
    mask_jax,
    normalize_model_output: bool = True,
    fid_tol: float = 1e-4,
    thresholds: Optional[Iterable[float]] = None,
):
    if thresholds is None:
        thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    thresholds = list(thresholds)

    x_current = np.array(x_best, dtype=np.float32)

    nn_state = np.asarray(
        apply_fn(params, jnp.asarray(x_current, dtype=jnp.float32)),
        dtype=np.float32,
    )

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

        nn_state_new = np.asarray(
            apply_fn(params, jnp.asarray(x_new, dtype=jnp.float32)),
            dtype=np.float32,
        )

        if normalize_model_output:
            nn_norm = np.linalg.norm(nn_state_new)
            if nn_norm > 1e-12:
                nn_state_new = nn_state_new / nn_norm

        fid_new = fidelity_np(nn_state_new, y_target_np)

        entry = {
            "threshold": float(threshold),
            "fidelity_before": float(fid_best),
            "fidelity_after": float(fid_new),
            "num_weights_zeroed": int(np.sum(np.abs(x_current) < threshold)),
        }

        delta = abs(fid_new - fid_best)

        if delta <= fid_tol:
            write(f"Threshold {threshold} accepted | fid = {fid_new}")
            x_current = x_new
            fid_best = fid_new
            nn_state = nn_state_new
            entry["accepted"] = True
        else:
            write(f"Threshold {threshold} rejected | fid dropped to {fid_new}")
            entry["accepted"] = False
            pruning_info.append(entry)
            break

        pruning_info.append(entry)

    py_state, _ = pytheus_state_from_x(x_current, tensor_jax, mask_jax)

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


def save_loss_fid_plot(
    sample_dir: Path,
    loss_history: List[float],
    fid_history: List[float],
    target_name: str,
    nphotons: int,
) -> None:
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


def save_edge_fidelity_plot(pt: Path, edge_list: List[int], fid_list: List[float], cfg: Dict[str, Any]) -> None:
    if len(edge_list) == 0:
        return

    edges_np = np.array(edge_list)
    fid_np = np.array(fid_list)

    unique_edges = np.unique(edges_np)
    counts = []
    avg_fid = []

    for edge_count in unique_edges:
        mask = edges_np == edge_count
        counts.append(np.sum(mask))
        avg_fid.append(np.mean(fid_np[mask]))

    counts = np.array(counts)
    avg_fid = np.array(avg_fid)

    sort_idx = np.argsort(unique_edges)[::-1]
    unique_edges = unique_edges[sort_idx]
    counts = counts[sort_idx]
    avg_fid = avg_fid[sort_idx]

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


def save_time_plots(pt: Path, time_per_sample_list: List[float], n: int) -> None:
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
