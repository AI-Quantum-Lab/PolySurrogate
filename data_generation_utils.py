"""
Utility functions for GPU-accelerated dataset generation of graph-defined
quantum states using PyTheus perfect matchings and JAX.

This file contains:
- logging utilities
- PyTheus catalog construction
- JAX amplitude computation
- optional advanced graph-generation utilities from the original curriculum code

The main readable pipeline should stay in `01_data_generation.py`.
"""

from __future__ import annotations

import glob
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from pytheus import theseus as th


# =============================================================================
# Logging
# =============================================================================


def setup_logger(out_dir: Path) -> logging.Logger:
    """Create a logger that writes both to terminal and to a timestamped log file."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"data_generation_{ts}.log"

    logger = logging.getLogger("dataset_generation")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if this function is called multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Log file: {log_path}")
    return logger


def get_basic_logger() -> logging.Logger:
    """Fallback console logger used when no output directory is created."""
    logger = logging.getLogger("dataset_generation_basic")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)

    return logger


# =============================================================================
# Graph / edge indexing helpers
# =============================================================================


def pair_index(i: int, j: int, n_vertices: int) -> int:
    """Index of unordered pair (i, j) in upper-triangular edge ordering."""
    if i > j:
        i, j = j, i
    return i * (2 * n_vertices - i - 1) // 2 + (j - i - 1)


def edge_index(i: int, j: int, n_vertices: int) -> int:
    """Alias for pair_index."""
    return pair_index(i, j, n_vertices)


def weight_index(i: int, j: int, c: int, n_vertices: int) -> int:
    """
    Index into flattened graph-weight vector.

    For local dimension 2, every physical edge has 4 local weights.
    """
    return 4 * edge_index(i, j, n_vertices) + c


# =============================================================================
# PyTheus catalog construction
# =============================================================================


def precompute_catalog(vertices: int, dimensions: int):
    """
    Build PyTheus perfect-matching catalog and convert it into dense tensors.

    Returns:
        edges:
            PyTheus edge list.
        kets:
            Sorted computational basis keys.
        tensor:
            Integer tensor with shape (n_kets, max_matchings_per_ket, edges_per_matching).
            tensor[k, m, e] gives the edge index used in matching m for ket k.
        mask:
            Float mask with shape (n_kets, max_matchings_per_ket).
            mask[k, m] = 1 if matching m exists for ket k, else 0.
    """
    edges = th.buildAllEdges(
        dimensions=[dimensions] * vertices,
        string=False,
        imaginary=False,
        loops=False,
    )
    perfect_matchings = th.findPerfectMatchings(edges)
    catalog = th.stateCatalog(perfect_matchings)

    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    kets = sorted(catalog.keys())

    max_matchings = max(len(catalog[ket]) for ket in kets)
    edges_per_matching = len(catalog[kets[0]][0])

    sentinel = len(edges)
    tensor = np.full(
        (len(kets), max_matchings, edges_per_matching),
        sentinel,
        dtype=np.int64,
    )
    mask = np.zeros((len(kets), max_matchings), dtype=np.float64)

    for ket_idx, ket in enumerate(kets):
        for matching_idx, matching in enumerate(catalog[ket]):
            mask[ket_idx, matching_idx] = 1.0
            for edge_pos, edge in enumerate(matching):
                tensor[ket_idx, matching_idx, edge_pos] = edge_to_idx[edge]

    return edges, kets, tensor, mask


def extract_unique_pms_array(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract unique perfect matchings from the catalog tensor."""
    perfect_matchings = []

    for ket_idx in range(tensor.shape[0]):
        valid_indices = np.where(mask[ket_idx] > 0)[0]
        for matching_idx in valid_indices:
            perfect_matchings.append(tuple(tensor[ket_idx, matching_idx]))

    return np.array(sorted(set(perfect_matchings)), dtype=np.int32)


# =============================================================================
# JAX amplitude computation
# =============================================================================


@jax.jit
def compute_amplitudes_gpu(weights, tensor, mask):
    """
    Compute normalized quantum amplitudes from graph weights.

    Args:
        weights:
            Array with shape (batch_size, num_edges).
        tensor:
            Perfect-matching lookup tensor.
        mask:
            Valid perfect-matching mask.

    Returns:
        Normalized amplitudes with shape (batch_size, n_kets).
    """
    sentinel_column = jnp.ones((weights.shape[0], 1), dtype=jnp.float32)
    weights_extended = jnp.concatenate([weights, sentinel_column], axis=1)

    gathered = weights_extended[:, tensor]
    products = jnp.prod(gathered, axis=-1)
    amplitudes = jnp.sum(products * mask[None, :, :], axis=-1)

    norms = jnp.linalg.norm(amplitudes, axis=-1, keepdims=True)
    norms = jnp.where(norms > 0, norms, 1.0)

    return (amplitudes / norms).astype(jnp.float32)


@jax.jit
def compute_amplitudes_gpu_no_norms(weights, tensor, mask):
    """
    Compute raw, unnormalized amplitudes from graph weights.
    """
    sentinel_column = jnp.ones((weights.shape[0], 1), dtype=jnp.float32)
    weights_extended = jnp.concatenate([weights, sentinel_column], axis=1)

    gathered = weights_extended[:, tensor]
    products = jnp.prod(gathered, axis=-1)
    amplitudes = jnp.sum(products * mask[None, :, :], axis=-1)

    return amplitudes.astype(jnp.float32)


# =============================================================================
# Simple Level-5 generator used by the clean main script
# =============================================================================


def generate_random_dense_weights(
    rng: np.random.Generator,
    n_samples: int,
    num_edges: int,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Generate fully random dense graph weights.

    This corresponds to Level 5 in the original curriculum code.
    """
    return rng.uniform(low, high, size=(n_samples, num_edges)).astype(np.float32)


# =============================================================================
# Optional advanced generators from the original curriculum code
# Kept here so the main data-generation file remains clean.
# =============================================================================


def make_sparse_weights(
    rng: np.random.Generator,
    n_samples: int,
    num_edges: int,
    k_min: int,
    k_max: int,
) -> np.ndarray:
    """Generate sparse graph weights with a random number of active edges per row."""
    if n_samples == 0:
        return np.zeros((0, num_edges), dtype=np.float32)

    raw = rng.uniform(-1.0, 1.0, size=(n_samples, num_edges)).astype(np.float32)
    k_per_row = rng.integers(k_min, k_max + 1, size=n_samples)
    k_per_row = np.minimum(k_per_row, num_edges)

    priorities = rng.random(size=(n_samples, num_edges))
    sorted_priorities = np.sort(priorities, axis=1)[:, ::-1]
    thresholds = sorted_priorities[np.arange(n_samples), k_per_row - 1]
    sparse_mask = (priorities >= thresholds[:, None]).astype(np.float32)

    return raw * sparse_mask


def make_graphs_with_zero_pm_ultrafast(
    n_samples: int,
    num_edges: int,
    pms_arr: np.ndarray,
    rng: np.random.Generator,
    k_min: int = 1,
    k_max: int = 23,
    max_tries: int = 1000,
) -> np.ndarray:
    """Generate graphs designed to have zero active perfect matchings."""
    weights = np.zeros((n_samples, num_edges), dtype=np.float32)

    for sample_idx in range(n_samples):
        success = False

        for _ in range(max_tries):
            k = rng.integers(k_min, k_max + 1)
            chosen_edges = rng.choice(num_edges, size=k, replace=False)

            active_mask = np.zeros(num_edges, dtype=bool)
            active_mask[chosen_edges] = True

            pm_active = np.all(active_mask[pms_arr], axis=1)

            if not np.any(pm_active):
                weights[sample_idx, chosen_edges] = rng.uniform(-1.0, 1.0, size=k)
                success = True
                break

        if not success:
            raise RuntimeError(
                f"Failed to generate zero-perfect-matching graph for sample {sample_idx} "
                f"after {max_tries} attempts."
            )

    return weights


def make_graphs_with_m_pms_fast(
    n_samples: int,
    num_edges: int,
    pms_arr: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate graphs by activating the edges of m selected perfect matchings."""
    if n_samples == 0:
        return np.zeros((0, num_edges), dtype=np.float32)

    if m == 0:
        return np.zeros((n_samples, num_edges), dtype=np.float32)

    num_pms, edges_per_matching = pms_arr.shape

    scores = rng.random((n_samples, num_pms), dtype=np.float32)
    chosen_pm_idx = np.argpartition(scores, kth=m - 1, axis=1)[:, :m]
    chosen_edges = pms_arr[chosen_pm_idx]

    vals = rng.uniform(-1.0, 1.0, size=(n_samples, m, edges_per_matching)).astype(np.float32)
    weights = np.zeros((n_samples, num_edges), dtype=np.float32)

    row_idx = np.repeat(np.arange(n_samples), m * edges_per_matching)
    edge_idx = chosen_edges.reshape(-1)
    val_idx = vals.reshape(-1)

    weights[row_idx, edge_idx] = val_idx
    return weights


def make_graphs_with_zero_pm_fast(
    n_samples: int,
    num_edges: int,
    pms_arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Alternative zero-perfect-matching generator by zeroing one edge per PM."""
    if n_samples == 0:
        return np.zeros((0, num_edges), dtype=np.float32)

    num_pms, edges_per_matching = pms_arr.shape
    weights = rng.uniform(-1.0, 1.0, size=(n_samples, num_edges)).astype(np.float32)

    chosen_pos = rng.integers(0, edges_per_matching, size=(n_samples, num_pms))
    chosen_edges = np.take_along_axis(
        np.broadcast_to(pms_arr[None, :, :], (n_samples, num_pms, edges_per_matching)),
        chosen_pos[..., None],
        axis=2,
    ).squeeze(-1)

    row_idx = np.repeat(np.arange(n_samples), num_pms)
    edge_idx = chosen_edges.reshape(-1)
    weights[row_idx, edge_idx] = 0.0

    return weights


def make_sparse_edge_weights_batch_simple(
    edge_pairs,
    n_samples: int,
    n_vertices: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate sparse weights on specified physical edge pairs."""
    num_edges = 4 * (n_vertices * (n_vertices - 1) // 2)
    output = np.zeros((n_samples, num_edges), dtype=np.float32)

    for row in range(n_samples):
        for i, j in edge_pairs:
            base = edge_index(i, j, n_vertices)
            channel = rng.integers(0, 4)
            idx = 4 * base + channel
            output[row, idx] = rng.uniform(-1.0, 1.0)

    return output


def generate_curriculum_weights(
    rng: np.random.Generator,
    n_samples: int,
    num_edges: int,
    pms_arr: np.ndarray,
    level_ratios,
    sparsity,
    mag_levels,
    vertices: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original curriculum generator.

    Kept as an advanced utility. The clean main script does not use this.
    """
    if len(level_ratios) != 12:
        raise ValueError("level_ratios must contain exactly 12 values.")

    if not np.isclose(sum(level_ratios), 1.0):
        raise ValueError("level_ratios must sum to 1.0.")

    ns = [int(n_samples * ratio) for ratio in level_ratios]
    ns[-1] = n_samples - sum(ns[:-1])

    weights = np.zeros((n_samples, num_edges), dtype=np.float32)
    levels = np.empty(n_samples, dtype=np.int8)
    offset = 0

    # Level 0: very sparse
    n = ns[0]
    weights[offset:offset + n] = make_sparse_weights(
        rng, n, num_edges, sparsity["0_min"], sparsity["0_max"]
    )
    levels[offset:offset + n] = 0
    offset += n

    # Level 1: moderately sparse
    n = ns[1]
    weights[offset:offset + n] = make_sparse_weights(
        rng, n, num_edges, sparsity["1_min"], sparsity["1_max"]
    )
    levels[offset:offset + n] = 1
    offset += n

    # Levels 2, 3, 4: dense with increasing magnitudes
    for level, magnitude in enumerate(mag_levels, start=2):
        n = ns[level]
        weights[offset:offset + n] = rng.uniform(
            -magnitude, magnitude, size=(n, num_edges)
        ).astype(np.float32)
        levels[offset:offset + n] = level
        offset += n

    # Level 5: full random dense weights
    n = ns[5]
    weights[offset:offset + n] = generate_random_dense_weights(rng, n, num_edges)
    levels[offset:offset + n] = 5
    offset += n

    # Level 6: discrete {-1, 0, 1}
    n = ns[6]
    weights[offset:offset + n] = rng.choice(
        [-1.0, 0.0, 1.0], size=(n, num_edges)
    ).astype(np.float32)
    levels[offset:offset + n] = 6
    offset += n

    # Level 7: zero perfect matchings
    n = ns[7]
    weights[offset:offset + n] = make_graphs_with_zero_pm_ultrafast(
        n, num_edges, pms_arr, rng, k_min=1, k_max=2 * vertices * (vertices - 1)
    )
    levels[offset:offset + n] = 7
    offset += n

    # Levels 8-11: m perfect matchings
    for level, m in zip([8, 9, 10, 11], [1, 2, 3, 4]):
        n = ns[level]
        weights[offset:offset + n] = make_graphs_with_m_pms_fast(
            n, num_edges, pms_arr, m, rng
        )
        levels[offset:offset + n] = level
        offset += n

    permutation = rng.permutation(n_samples)
    return weights[permutation], levels[permutation]


# =============================================================================
# Optional save helpers
# =============================================================================


def save_npz_shard(path: Path, weights: np.ndarray, amps: np.ndarray) -> None:
    """Save one data shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=weights, amps=amps)


def merge_shards_to_npz(out_dir: Path, output_name: str = "dataset_merged.npz") -> Path:
    """
    Merge shard files named data_*.npz into a single NPZ file.

    Warning:
        The final np.savez step loads data into a standard .npz container.
        For very large datasets, training directly from shards or memmap files
        is usually safer.
    """
    shard_files = sorted(glob.glob(str(out_dir / "data_*.npz")))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {out_dir}")

    first = np.load(shard_files[0])
    num_edges = first["weights"].shape[1]
    n_kets = first["amps"].shape[1]
    first.close()

    total_samples = 0
    for shard_file in shard_files:
        with np.load(shard_file) as data:
            total_samples += data["weights"].shape[0]

    weights_memmap = np.memmap(
        out_dir / "merged_weights.dat",
        dtype="float32",
        mode="w+",
        shape=(total_samples, num_edges),
    )
    amps_memmap = np.memmap(
        out_dir / "merged_amps.dat",
        dtype="float32",
        mode="w+",
        shape=(total_samples, n_kets),
    )

    offset = 0
    for shard_file in shard_files:
        with np.load(shard_file) as data:
            n = data["weights"].shape[0]
            weights_memmap[offset:offset + n] = data["weights"]
            amps_memmap[offset:offset + n] = data["amps"]
            offset += n

    output_path = out_dir / output_name
    np.savez(output_path, weights=np.array(weights_memmap), amps=np.array(amps_memmap))

    return output_path
