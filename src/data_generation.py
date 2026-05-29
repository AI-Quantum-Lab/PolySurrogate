"""
Clean random graph dataset generation script.

This is the readable main file for GitHub.

Pipeline:
1. Build PyTheus perfect-matching catalog.
2. Generate dense random graph weights in [-1, 1].
3. Compute amplitudes using JAX on GPU.
4. Save data batch-wise as NPZ shards.

This corresponds to Level 5 in the original curriculum code.
Advanced generators are kept in `data_generation_utils.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data_generation_utils import (
    compute_amplitudes_gpu,
    compute_amplitudes_gpu_no_norms,
    generate_random_dense_weights,
    get_basic_logger,
    precompute_catalog,
    setup_logger,
)

from data_config import (
    VERTICES,
    DIMENSIONS,
    N_SAMPLES,
    BATCH_SIZE,
    SEED,
    SAVE_TO_FILE,
    OUT_DIR,
    SHARD_SIZE,
    NORMED_DATA,
)


# =============================================================================
# Dataset generation
# =============================================================================


def generate_dataset(
    vertices: int = VERTICES,
    dimensions: int = DIMENSIONS,
    n_samples: int = N_SAMPLES,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    save_to_file: bool = SAVE_TO_FILE,
    out_dir_path: str = OUT_DIR,
    shard_size: int = SHARD_SIZE,
    normed_data: bool = NORMED_DATA,
):
    """
    Generate random dense graph data and corresponding amplitudes.

    Args:
        vertices:
            Number of graph vertices / photons / modes.
        dimensions:
            Local dimension. Usually 2 for qubit-like systems.
        n_samples:
            Number of samples to generate.
        batch_size:
            Number of samples processed per JAX batch.
        seed:
            Random seed for reproducibility.
        save_to_file:
            If True, save NPZ shards to disk. If False, return arrays in memory.
        out_dir_path:
            Output directory used when save_to_file=True.
        shard_size:
            Maximum number of samples per saved shard.
        normed_data:
            If True, save normalized amplitudes. If False, save raw amplitudes.

    Returns:
        If save_to_file=True:
            Path to output directory.
        If save_to_file=False:
            Tuple (weights, amps).
    """
    if save_to_file:
        out_dir = Path(out_dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(out_dir)
    else:
        out_dir = None
        logger = get_basic_logger()

    logger.info("=== Random Dense Graph Dataset Generation ===")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"vertices={vertices}, dimensions={dimensions}")
    logger.info(f"n_samples={n_samples:,}, batch_size={batch_size:,}")
    logger.info(f"seed={seed}")
    logger.info(f"normed_data={normed_data}")
    logger.info("graph regime=Level 5 random dense weights in [-1, 1]")

    # -------------------------------------------------------------------------
    # 1. Build PyTheus perfect-matching catalog
    # -------------------------------------------------------------------------
    logger.info("Precomputing PyTheus perfect-matching catalog...")
    t0 = time.time()

    edges, kets, tensor, mask = precompute_catalog(vertices, dimensions)

    num_edges = len(edges)
    n_kets = len(kets)

    logger.info(f"num_edges={num_edges}")
    logger.info(f"n_kets={n_kets}")
    logger.info(f"catalog tensor shape={tensor.shape}")
    logger.info(f"catalog time={time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # 2. Transfer catalog to GPU
    # -------------------------------------------------------------------------
    logger.info("Transferring catalog to GPU...")
    tensor_gpu = jax.device_put(jnp.array(tensor, dtype=jnp.int32))
    mask_gpu = jax.device_put(jnp.array(mask, dtype=jnp.float32))

    # -------------------------------------------------------------------------
    # 3. JIT warmup
    # -------------------------------------------------------------------------
    logger.info("JIT warmup...")
    dummy_batch_size = min(1000, batch_size)
    dummy_weights = jnp.ones((dummy_batch_size, num_edges), dtype=jnp.float32)

    if normed_data:
        compute_amplitudes_gpu(dummy_weights, tensor_gpu, mask_gpu).block_until_ready()
    else:
        compute_amplitudes_gpu_no_norms(dummy_weights, tensor_gpu, mask_gpu).block_until_ready()

    del dummy_weights

    # -------------------------------------------------------------------------
    # 4. Metadata
    # -------------------------------------------------------------------------
    if save_to_file:
        metadata = {
            "vertices": vertices,
            "dimensions": dimensions,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "seed": seed,
            "num_edges": num_edges,
            "n_kets": n_kets,
            "input_shape": [n_samples, num_edges],
            "output_shape": [n_samples, n_kets],
            "normed_data": normed_data,
            "graph_regime": "random_dense_level_5",
            "weight_range": [-1.0, 1.0],
        }

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Batch generation and saving
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    all_weights = []
    all_amps = []

    shard_weights = None
    shard_amps = None
    shard_fill = 0
    shard_idx = 0

    if save_to_file:
        shard_weights = np.empty((shard_size, num_edges), dtype=np.float32)
        shard_amps = np.empty((shard_size, n_kets), dtype=np.float32)

    logger.info("Generating data batches...")
    start_time = time.time()

    samples_done = 0
    batch_idx = 0

    while samples_done < n_samples:
        n_current = min(batch_size, n_samples - samples_done)

        # CPU: random dense graph weights, Level 5 from original code.
        weights_np = generate_random_dense_weights(rng, n_current, num_edges)

        # GPU: amplitude computation.
        weights_gpu = jax.device_put(jnp.array(weights_np, dtype=jnp.float32))

        if normed_data:
            amps_gpu = compute_amplitudes_gpu(weights_gpu, tensor_gpu, mask_gpu)
        else:
            amps_gpu = compute_amplitudes_gpu_no_norms(weights_gpu, tensor_gpu, mask_gpu)

        amps_np = np.array(amps_gpu, dtype=np.float32)

        if save_to_file:
            # Fill fixed-size shards safely, even if batch_size > shard_size.
            local_offset = 0
            while local_offset < n_current:
                free_space = shard_size - shard_fill
                take = min(free_space, n_current - local_offset)

                shard_weights[shard_fill:shard_fill + take] = weights_np[local_offset:local_offset + take]
                shard_amps[shard_fill:shard_fill + take] = amps_np[local_offset:local_offset + take]

                shard_fill += take
                local_offset += take

                if shard_fill == shard_size:
                    shard_path = out_dir / f"data_{shard_idx:05d}.npz"
                    np.savez(shard_path, weights=shard_weights, amps=shard_amps)
                    logger.info(f"Saved shard {shard_idx:05d} -> {shard_path}")

                    shard_idx += 1
                    shard_fill = 0
        else:
            all_weights.append(weights_np)
            all_amps.append(amps_np)

        samples_done += n_current
        batch_idx += 1

        if batch_idx % 10 == 0 or samples_done >= n_samples:
            elapsed = time.time() - start_time
            rate = samples_done / max(elapsed, 1e-9)
            logger.info(
                f"batch {batch_idx:>5d} | "
                f"{samples_done:>10,}/{n_samples:,} "
                f"({100 * samples_done / n_samples:5.1f}%) | "
                f"{elapsed:.1f}s | "
                f"{rate:,.0f} samples/s"
            )

    # -------------------------------------------------------------------------
    # 6. Save final partial shard or return arrays
    # -------------------------------------------------------------------------
    if save_to_file:
        if shard_fill > 0:
            shard_path = out_dir / f"data_{shard_idx:05d}.npz"
            np.savez(
                shard_path,
                weights=shard_weights[:shard_fill],
                amps=shard_amps[:shard_fill],
            )
            logger.info(f"Saved final shard {shard_idx:05d} -> {shard_path}")

        total_time = time.time() - start_time
        logger.info("=== DONE ===")
        logger.info(f"Total generation time: {total_time:.1f}s")
        logger.info(f"Output directory: {out_dir}")

        return out_dir

    final_weights = np.concatenate(all_weights, axis=0)
    final_amps = np.concatenate(all_amps, axis=0)

    total_time = time.time() - start_time
    logger.info("=== DONE ===")
    logger.info(f"Total generation time: {total_time:.1f}s")

    return final_weights, final_amps


# =============================================================================
# Simple notebook/script execution block
# =============================================================================


if __name__ == "__main__":
    # Values are read from config.py by default.
    output = generate_dataset()

    print("Done")
    print(output)
