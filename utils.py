"""
utils.py -- shared functions used by 01_data_generate.py, 02_ml_model.py,
and inverse_design.py.

This file holds only genuinely common code: things that would otherwise be
byte-for-byte duplicated across two or more of the numbered stage scripts
(PyTheus catalog/amplitude computation, the FNN/PNN model definitions,
reproducibility primitives, fidelity, JSON I/O). Each stage script keeps its
own parameters and orchestration logic -- this file has no "main" behaviour
of its own and is never run directly.

Reproducibility metadata (see the "Reproducibility" section below): every
stage script writes a `reproducibility_manifest.json` recording the git
commit, environment (JAX/jaxlib/flax/optax versions, devices), the resolved
config it ran with (including all seeds), and a hash of its key input file
(dataset for training, model checkpoint for inverse design) -- everything
needed to know exactly what produced a given run's output later.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from pytheus import theseus as th


# =============================================================================
# JSON I/O
# =============================================================================


def to_jsonable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays, Paths, and scalars into plain
    JSON-serialisable Python types."""
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


def write_json(path: Path, data: dict) -> None:
    """Plain JSON write (no array conversion) -- for dicts that are already
    plain Python types, e.g. training's model_info."""
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# PyTheus catalog + amplitude computation
# =============================================================================
#
# Shared by 01_data_generate.py (fresh random samples), 02_ml_model.py's
# training target (the amplitudes 01 produced), and inverse_design.py
# (both for generating fresh starting samples and for PyTheus verification
# of the optimiser's current graph against the target state).


def build_pytheus_catalog(vertices: int, dimensions: int = 2):
    """Build the PyTheus perfect-matching catalog and convert it into dense
    tensors: tensor[k, m, e] = edge index used in matching m for ket k;
    mask[k, m] = 1 if matching m exists for ket k, else 0.

    Returns (edges, kets, tensor, mask). tensor is int32, mask is float32 --
    both get cast to those dtypes before any JAX use regardless, so storing
    them that way from the start costs nothing.
    """
    edges = th.buildAllEdges(dimensions=[dimensions] * vertices, string=False, imaginary=False, loops=False)
    perfect_matchings = th.findPerfectMatchings(edges)
    catalog = th.stateCatalog(perfect_matchings)

    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    kets = sorted(catalog.keys())

    max_matchings = max(len(catalog[ket]) for ket in kets)
    edges_per_matching = len(catalog[kets[0]][0])

    sentinel = len(edges)
    tensor = np.full((len(kets), max_matchings, edges_per_matching), sentinel, dtype=np.int32)
    mask = np.zeros((len(kets), max_matchings), dtype=np.float32)

    for ket_idx, ket in enumerate(kets):
        for matching_idx, matching in enumerate(catalog[ket]):
            mask[ket_idx, matching_idx] = 1.0
            for edge_pos, edge in enumerate(matching):
                tensor[ket_idx, matching_idx, edge_pos] = edge_to_idx[edge]

    return edges, kets, tensor, mask


@jax.jit
def compute_amplitudes(weights, tensor, mask):
    """Normalised quantum amplitudes from graph weights. weights: (batch, num_edges)."""
    sentinel_column = jnp.ones((weights.shape[0], 1), dtype=jnp.float32)
    weights_extended = jnp.concatenate([weights, sentinel_column], axis=1)
    gathered = weights_extended[:, tensor]
    products = jnp.prod(gathered, axis=-1)
    amplitudes = jnp.sum(products * mask[None, :, :], axis=-1)
    norms = jnp.linalg.norm(amplitudes, axis=-1, keepdims=True)
    norms = jnp.where(norms > 0, norms, 1.0)
    return (amplitudes / norms).astype(jnp.float32)


@jax.jit
def compute_amplitudes_no_norms(weights, tensor, mask):
    """Raw, unnormalised amplitudes from graph weights."""
    sentinel_column = jnp.ones((weights.shape[0], 1), dtype=jnp.float32)
    weights_extended = jnp.concatenate([weights, sentinel_column], axis=1)
    gathered = weights_extended[:, tensor]
    products = jnp.prod(gathered, axis=-1)
    amplitudes = jnp.sum(products * mask[None, :, :], axis=-1)
    return amplitudes.astype(jnp.float32)


def pytheus_state_from_x(x_vec, tensor_jax, mask_jax):
    """Normalised PyTheus quantum state for a single graph-weight vector."""
    x_vec = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)
    amps = compute_amplitudes(jnp.array(x_vec, dtype=jnp.float32), tensor_jax, mask_jax)
    amps = jax.block_until_ready(amps)
    return np.asarray(amps[0], dtype=np.float32)


def generate_random_dense_weights(rng: np.random.Generator, n_samples: int, num_edges: int,
                                   low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Fully random dense graph weights, drawn from a single sequentially-
    consumed RNG stream -- reproducible for a full identical rerun (same
    seed, same n_samples), not independent of batch/subset structure. See
    REPRODUCIBILITY.md."""
    return rng.uniform(low, high, size=(n_samples, num_edges)).astype(np.float32)


def fidelity_np(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2 between two L2-normalised vectors (normalises internally)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"State size mismatch: {a.size} vs {b.size}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    a, b = a / na, b / nb
    return float(np.abs(np.vdot(a, b)) ** 2)


# =============================================================================
# Model definitions
# =============================================================================


class FNN(nn.Module):
    hidden_dims: tuple
    out_dims: int
    nodes: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h, use_bias=True)(x)
            x = nn.gelu(x)
        return nn.Dense(self.out_dims, use_bias=True)(x)


class PNN(nn.Module):
    """One hidden layer of size hidden_dims, monomial activation x^(nodes/2).
    Output is the unnormalised quantum-state amplitude vector."""
    hidden_dims: int
    out_dims: int
    nodes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dims, use_bias=True)(x)
        phi = x ** (self.nodes / 2)
        return nn.Dense(self.out_dims, use_bias=True)(phi)


def create_model(model_name, hidden_dims, out_dims, nodes):
    model_name = model_name.upper()
    if model_name == "HNN":
        model_name = "PNN"

    if model_name == "FNN":
        if isinstance(hidden_dims, int):
            raise ValueError("For FNN, hidden_dims should be a tuple, e.g. (2000, 2000, 2000).")
        return FNN(hidden_dims=tuple(hidden_dims), out_dims=out_dims, nodes=nodes)

    if model_name == "PNN":
        if isinstance(hidden_dims, (tuple, list)):
            if len(hidden_dims) != 1:
                raise ValueError("For PNN, hidden_dims should be an integer, e.g. 1000.")
            hidden_dims = int(hidden_dims[0])
        return PNN(hidden_dims=int(hidden_dims), out_dims=out_dims, nodes=nodes)

    raise ValueError(f"Unknown model_name: {model_name}. Use 'PNN' or 'FNN'.")


def save_params_msgpack(params, path: Path) -> None:
    with Path(path).open("wb") as f:
        f.write(to_bytes(params))


def load_trained_model(model_type: str, architecture, input_dim: int, out_dim: int, nodes: int, model_path):
    """Load a trained Flax model's params from a params.msgpack file.
    Returns (apply_fn, params)."""
    model = create_model(model_name=model_type, hidden_dims=architecture, out_dims=out_dim, nodes=nodes)
    x_example = jnp.zeros((1, input_dim), dtype=jnp.float32)
    params_template = model.init(jax.random.PRNGKey(0), x_example)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model params file not found: {model_path}")
    with model_path.open("rb") as f:
        params = from_bytes(params_template, f.read())
    return model.apply, params


# =============================================================================
# Reproducibility: PRNG key plan (fold_in, not sequential split())
# =============================================================================
#
# root_key = jax.random.PRNGKey(SEED)
# Every top-level purpose gets an independent key via fold_in with a fixed,
# named integer tag -- NOT via sequential jax.random.split() chaining. This
# is what makes resumed/rerun trajectories reproducible: nothing about
# "where we were in the RNG sequence" needs to survive a process restart or
# depend on other samples/epochs, because every key is independently
# re-derivable from (SEED, purpose-tag, index) alone.

KEY_TAG_MODEL_INIT = 0
KEY_TAG_DATA_SPLIT = 1
KEY_TAG_EPOCH_ROOT = 2
KEY_TAG_BATCH_ROOT = 3
KEY_TAG_DROPOUT_ROOT = 4
KEY_TAG_OPTIMISER_JITTER = 5


def derive_root_keys(seed: int) -> dict:
    """Used by 02_ml_model.py. Independent keys for model init, dataset
    split, epoch shuffling, and a reserved dropout/batch-level slot."""
    root_key = jax.random.PRNGKey(seed)
    return {
        "root": root_key,
        "model_init": jax.random.fold_in(root_key, KEY_TAG_MODEL_INIT),
        "data_split": jax.random.fold_in(root_key, KEY_TAG_DATA_SPLIT),
        "epoch_root": jax.random.fold_in(root_key, KEY_TAG_EPOCH_ROOT),
        "batch_root": jax.random.fold_in(root_key, KEY_TAG_BATCH_ROOT),
        "dropout_root": jax.random.fold_in(root_key, KEY_TAG_DROPOUT_ROOT),
    }


def epoch_key_for(key_epoch_root, epoch: int):
    """Pure function of (key_epoch_root, epoch) -- resume-safe."""
    return jax.random.fold_in(key_epoch_root, int(epoch))


def optimiser_jitter_root_key(seed: int):
    """Used by inverse_design.py. Root key for all jitter randomness,
    independent of every other key derived from the same SEED."""
    return jax.random.fold_in(jax.random.PRNGKey(seed), KEY_TAG_OPTIMISER_JITTER)


def optimiser_sample_key_for(jitter_root_key, sample_id: int):
    """Pure function of (jitter_root_key, sample_id) only -- deliberately NOT
    of any dataset metadata, so a sample re-supplied at the same position
    always gets identical noise regardless of what metadata travels with it."""
    return jax.random.fold_in(jitter_root_key, int(sample_id))


def optimiser_jitter_event_key_for(sample_key, event_index: int):
    """event_index=0 is the initial pre-optimisation jitter; 1, 2, 3, ... are
    stall-triggered events in firing order."""
    return jax.random.fold_in(sample_key, int(event_index))


# =============================================================================
# Reproducibility: precision + XLA-determinism helpers
# =============================================================================


def configure_precision(precision: str) -> None:
    """Must be called before any array/param creation."""
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    elif precision == "float32":
        jax.config.update("jax_enable_x64", False)
    else:
        raise ValueError(f"Unknown precision: {precision!r}. Use 'float32' or 'float64'.")


def dtype_for_precision(precision: str):
    return jnp.float64 if precision == "float64" else jnp.float32


def deterministic_xla_env_is_set() -> bool:
    return "--xla_gpu_deterministic_ops=true" in __import__("os").environ.get("XLA_FLAGS", "")


# =============================================================================
# Reproducibility: git / environment / file hashing / manifest
# =============================================================================


def get_git_info(repo_root) -> dict:
    """Best-effort git commit hash + dirty flag. Never raises."""
    repo_root = Path(repo_root)

    def _run(args):
        return subprocess.run(args, cwd=repo_root, capture_output=True, text=True, timeout=10)

    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        if commit.returncode != 0:
            return {"available": False, "reason": commit.stderr.strip()}
        status = _run(["git", "status", "--porcelain"])
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        branch_name = branch.stdout.strip() if branch.returncode == 0 else None
        return {"available": True, "commit": commit.stdout.strip(), "dirty": dirty, "branch": branch_name}
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "reason": str(e)}


def get_environment_info() -> dict:
    """JAX/jaxlib/flax/optax/numpy versions, devices, backend, precision mode."""
    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
    except Exception:
        jaxlib_version = None
    try:
        import flax
        flax_version = flax.__version__
    except Exception:
        flax_version = None
    try:
        optax_version = optax.__version__
    except Exception:
        optax_version = None
    try:
        import pytheus
        pytheus_version = pytheus.__version__
    except Exception:
        pytheus_version = None
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception:
        devices = []
    try:
        backend = jax.default_backend()
    except Exception:
        backend = None

    return {
        "python_version": sys.version, "platform": platform.platform(),
        "jax_version": jax.__version__, "jaxlib_version": jaxlib_version,
        "flax_version": flax_version, "optax_version": optax_version,
        "pytheus_version": pytheus_version,
        "numpy_version": np.__version__, "jax_default_backend": backend,
        "jax_devices": devices, "jax_x64_enabled": bool(jax.config.jax_enable_x64),
    }


def hash_file(path, mode: str = "fast") -> dict:
    """Identity-hash any file (dataset or model checkpoint).

    mode="fast" (default): hashes file size + mtime + first/last 1MB. Cheap
      regardless of file size -- adequate to catch "wrong file" mistakes.
    mode="full": sha256 of the entire file. Exact but slow for large files.
    mode="none": skip hashing, just record path/size/mtime.
    """
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}

    size = path.stat().st_size
    info = {"path": str(path), "exists": True, "size_bytes": size,
            "mtime": path.stat().st_mtime, "hash_mode": mode}
    if mode == "none":
        return info

    h = hashlib.sha256()
    if mode == "full":
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
        return info

    chunk_size = 1 << 20
    with path.open("rb") as f:
        h.update(f.read(chunk_size))
        if size > chunk_size:
            f.seek(max(size - chunk_size, 0))
            h.update(f.read(chunk_size))
    h.update(str(size).encode())
    info["sha256_fast"] = h.hexdigest()
    return info


def hash_array(arr) -> str:
    arr = np.asarray(jax.device_get(arr))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def hash_pytree(tree) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(jax.device_get(leaf))
        h.update(arr.tobytes())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
    return h.hexdigest()


def build_manifest(*, resolved_config: dict, seeds: dict, input_file_info: dict,
                    repo_root, extra: dict | None = None) -> dict:
    """Assemble a reproducibility manifest: everything needed to know what
    produced a given run's output later -- resolved config (including every
    seed), git commit, environment, and a hash of the run's key input file
    (dataset for training, model checkpoint for inverse design)."""
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "resolved_config": resolved_config,
        "seeds": seeds,
        "input_file": input_file_info,
        "git": get_git_info(repo_root),
        "environment": get_environment_info(),
        "launch_command": " ".join(sys.argv),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path, manifest: dict) -> None:
    with Path(path).open("w") as f:
        json.dump(manifest, f, indent=2, default=str)


def append_trace_line(path, record: dict) -> None:
    with Path(path).open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
