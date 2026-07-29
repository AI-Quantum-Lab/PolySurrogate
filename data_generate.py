"""
01_data_generate.py -- Stage 1: random graph -> quantum amplitude dataset.

Imports shared PyTheus catalog/amplitude code from utils.py (also used by
03_inverse_design.py); everything stage-specific (parameters, shard I/O,
the main generate_dataset() orchestration) lives in this file. Edit the
constants in the "Parameters" section below, then run:

    python 01_data_generate.py

Output path is fully automatic -- no folder ever needs to be defined
externally. When SAVE_DATA=True, data is written to
`DATA_ROOT/n{VERTICES}/n{VERTICES}_{i}` (e.g. `results/data_generation/n4/n4_0`
for the first n=4 run, `n4_1` for the next, and so on -- `{i}` is the next
free integer under that n{VERTICES} folder, found automatically). When
SAVE_DATA=False, nothing is written to disk at all. Either way,
generate_dataset() always returns `(weights, amps, out_dir)`, with
`out_dir` set to the path above when saved, or None when not.

When saved, a `reproducibility_manifest.json` is also written alongside
`dataset_merged.npz` -- git commit, environment (JAX/jaxlib/flax/optax/
pytheus/numpy versions, devices), the resolved config (including the seed),
and a hash of the generated dataset file itself. Stage 2 (02_ml_model.py)
reads `dataset_merged.npz` from whichever `out_dir` this run produced --
there is no direct code import between the numbered stage scripts, only
the generated folder.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils import (
    build_manifest,
    build_pytheus_catalog,
    compute_amplitudes,
    compute_amplitudes_no_norms,
    generate_random_dense_weights,
    hash_file,
    write_manifest,
)

# =============================================================================
# Parameters -- edit these directly.
# =============================================================================

VERTICES = 4          # graph vertices / photons / modes
DIMENSIONS = 2         # local Hilbert-space dimension (2 = qubit-like)

N_SAMPLES = 20_000000       # total samples to generate
BATCH_SIZE = 5000  # matches historical dataset-generation batch structure
SHARD_SIZE = 500_000      # max samples per saved shard file

SEED = 34              # np.random.default_rng seed -- reproducible given the
                       # same seed and a full identical rerun (see note at
                       # the bottom of this file on what this guarantees).

# If True, data is written to disk at an auto-computed path (see module
# docstring) -- no folder name ever needs to be chosen or passed in. If
# False, nothing is written; only the arrays are returned.

SAVE_DATA = True


REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT = REPO_ROOT / "results" / "data_generation"

# False (default) -> raw unnormalised amplitude vectors from the simulator.
#                    This is what the surrogate model is trained to predict.
# True  -> amplitudes are L2-normalised so ||psi||_2 = 1 before saving.
# Must match NORMED_DATA used in 03_inverse_design.py's starting-sample
# generation, if that also generates fresh samples.
NORMED_DATA = False

# Numerical precision for generated weights, amplitudes, shards, and merged data.
# Change only this line to switch the entire data-generation pipeline.
PRECISION = "float32"  # "float32" or "float64", data used for training the model, was produced with float32, so keep it that way for now.
if PRECISION not in {"float32", "float64"}:
    raise ValueError("PRECISION must be 'float32' or 'float64'")

if PRECISION == "float64":
    jax.config.update("jax_enable_x64", True) # Allow actual 64-bit arrays and calculations.
    NP_FLOAT_DTYPE = np.float64
    JAX_FLOAT_DTYPE = jnp.float64
else:
    NP_FLOAT_DTYPE = np.float32
    JAX_FLOAT_DTYPE = jnp.float32


# =============================================================================
# Logging
# =============================================================================


def _setup_logger(out_dir: Path | None) -> logging.Logger:
    """File+console logger if out_dir is given, console-only otherwise."""
    logger = logging.getLogger(f"dataset_generation_{id(out_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if out_dir is not None:
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"data_generation_{ts}.log"

        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if out_dir is not None:
        logger.info(f"Log file: {log_path}")
    return logger


# =============================================================================
# Auto-numbered output directory
# =============================================================================


def _next_run_dir(data_root: str, vertices: int) -> Path:
    """Next free `{data_root}/n{vertices}/n{vertices}_{i}` directory (i
    starting at 0) -- auto-incrementing so back-to-back runs never collide
    and no run name ever needs to be chosen by hand."""
    parent = Path(data_root) / f"n{vertices}"
    parent.mkdir(parents=True, exist_ok=True)

    i = 0
    while (parent / f"n{vertices}_{i}").exists():
        i += 1
    return parent / f"n{vertices}_{i}"


# =============================================================================
# Shard I/O
# =============================================================================


def merge_shards_to_npz(out_dir: Path, output_name: str = "dataset_merged.npz") -> Path:
    """Merge shard files named data_*.npz into a single NPZ file."""
    shard_files = sorted(glob.glob(str(out_dir / "data_*.npz")))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {out_dir}")

    with np.load(shard_files[0]) as first:
        num_edges = first["weights"].shape[1]
        n_kets = first["amps"].shape[1]
        weights_dtype = first["weights"].dtype
        amps_dtype = first["amps"].dtype

    total_samples = 0
    for shard_file in shard_files:
        with np.load(shard_file) as data:
            total_samples += data["weights"].shape[0]

    weights_memmap = np.memmap(
        out_dir / "merged_weights.dat", dtype=weights_dtype, mode="w+",
        shape=(total_samples, num_edges),
    )
    amps_memmap = np.memmap(
        out_dir / "merged_amps.dat", dtype=amps_dtype, mode="w+",
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


# =============================================================================
# Main entry point
# =============================================================================


def generate_dataset(
    vertices: int = VERTICES, dimensions: int = DIMENSIONS,
    n_samples: int = N_SAMPLES, batch_size: int = BATCH_SIZE, seed: int = SEED,
    shard_size: int = SHARD_SIZE, normed_data: bool = NORMED_DATA,
    save_data: bool = SAVE_DATA, data_root: str = DATA_ROOT,
):
    """Generate a dataset. Always returns (weights, amps, out_dir).

    If save_data=True (default), data is additionally written to disk at
    the next free `{data_root}/n{vertices}/n{vertices}_{i}` directory
    (auto-numbered, never needs to be chosen by hand), and out_dir is that
    Path. If save_data=False, nothing is written and out_dir is None.
    Every argument defaults to this file's top-of-file constants -- pass
    explicit values to override any of them per call.
    """
    out_dir = _next_run_dir(data_root, vertices) if save_data else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir)

    logger.info("=== Random Dense Graph Dataset Generation ===")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"vertices={vertices}, dimensions={dimensions}")
    logger.info(f"n_samples={n_samples:,}, batch_size={batch_size:,}")
    logger.info(f"seed={seed}")
    logger.info(f"normed_data={normed_data}")
    logger.info(f"precision={PRECISION}")
    logger.info(f"save_data={save_data}  out_dir={out_dir}")

    logger.info("Precomputing PyTheus perfect-matching catalog...")
    t0 = time.time()
    edges, kets, tensor, mask = build_pytheus_catalog(vertices, dimensions)
    num_edges = len(edges)
    n_kets = len(kets)
    logger.info(f"num_edges={num_edges}  n_kets={n_kets}  catalog time={time.time() - t0:.2f}s")

    tensor_gpu = jax.device_put(jnp.array(tensor, dtype=jnp.int32))
    mask_gpu = jax.device_put(jnp.asarray(mask, dtype=JAX_FLOAT_DTYPE))

    logger.info("JIT warmup...")
    dummy_batch_size = min(1000, batch_size)
    dummy_weights = jnp.ones((dummy_batch_size, num_edges), dtype=JAX_FLOAT_DTYPE)
    if normed_data:
        compute_amplitudes(dummy_weights, tensor_gpu, mask_gpu).block_until_ready()
    else:
        compute_amplitudes_no_norms(dummy_weights, tensor_gpu, mask_gpu).block_until_ready()
    del dummy_weights

    if out_dir is not None:
        metadata = {
            "vertices": vertices, "dimensions": dimensions, "n_samples": n_samples,
            "batch_size": batch_size, "seed": seed, "num_edges": num_edges, "n_kets": n_kets,
            "input_shape": [n_samples, num_edges], "output_shape": [n_samples, n_kets],
            "normed_data": normed_data, "precision": PRECISION,
            "weights_dtype": str(np.dtype(NP_FLOAT_DTYPE)),
            "amps_dtype": str(np.dtype(NP_FLOAT_DTYPE)),
            "graph_regime": "random_dense_level_5",
            "weight_range": [-1.0, 1.0],
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    rng = np.random.default_rng(seed)

    # In-memory accumulation happens regardless of save_data, since the
    # arrays are always returned.
    all_weights: list[np.ndarray] = []
    all_amps: list[np.ndarray] = []

    if out_dir is not None:
        shard_weights = np.empty((shard_size, num_edges), dtype=NP_FLOAT_DTYPE)
        shard_amps = np.empty((shard_size, n_kets), dtype=NP_FLOAT_DTYPE)
        shard_fill = 0
        shard_idx = 0

    logger.info("Generating data batches...")
    start_time = time.time()
    samples_done = 0
    batch_idx = 0

    while samples_done < n_samples:
        n_current = min(batch_size, n_samples - samples_done)

        #weights_np = generate_random_dense_weights(rng, n_current, num_edges)
        # Match the old generator: draw the batch and then shuffle its rows
        weights_np = np.empty(
            (n_current, num_edges),
            dtype=NP_FLOAT_DTYPE,
        )

        weights_np[:] = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(n_current, num_edges),
        )

        perm = rng.permutation(n_current)

        weights_np = weights_np[perm]
        
        weights_gpu = jax.device_put(jnp.asarray(weights_np, dtype=JAX_FLOAT_DTYPE))

        if normed_data:
            amps_gpu = compute_amplitudes(weights_gpu, tensor_gpu, mask_gpu)
        else:
            amps_gpu = compute_amplitudes_no_norms(weights_gpu, tensor_gpu, mask_gpu)
        amps_np = np.asarray(jax.device_get(amps_gpu), dtype=NP_FLOAT_DTYPE)

        all_weights.append(weights_np)
        all_amps.append(amps_np)

        if out_dir is not None:
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

        samples_done += n_current
        batch_idx += 1

        if batch_idx % 10 == 0 or samples_done >= n_samples:
            elapsed = time.time() - start_time
            rate = samples_done / max(elapsed, 1e-9)
            logger.info(
                f"batch {batch_idx:>5d} | {samples_done:>10,}/{n_samples:,} "
                f"({100 * samples_done / n_samples:5.1f}%) | {elapsed:.1f}s | {rate:,.0f} samples/s"
            )

    if out_dir is not None and shard_fill > 0:
        shard_path = out_dir / f"data_{shard_idx:05d}.npz"
        np.savez(shard_path, weights=shard_weights[:shard_fill], amps=shard_amps[:shard_fill])
        logger.info(f"Saved final shard {shard_idx:05d} -> {shard_path}")

    logger.info(f"=== DONE === total time={time.time() - start_time:.1f}s")

    weights_all = np.concatenate(all_weights, axis=0)
    amps_all = np.concatenate(all_amps, axis=0)

    if out_dir is not None:
        logger.info("Merging shards...")
        merged_path = merge_shards_to_npz(out_dir, "dataset_merged.npz")
        logger.info(f"Merged dataset -> {merged_path}")

        manifest = build_manifest(
            resolved_config=dict(
                vertices=vertices, dimensions=dimensions, n_samples=n_samples,
                batch_size=batch_size, seed=seed, shard_size=shard_size,
                normed_data=normed_data, save_data=save_data, data_root=data_root,
                precision=PRECISION,
            ),
            seeds={"seed": int(seed)},
            input_file_info={"note": "Stage 1 has no input file -- it is the data source."},
            repo_root=REPO_ROOT,
            extra={"output_file": hash_file(merged_path)},
        )
        write_manifest(out_dir / "reproducibility_manifest.json", manifest)
        logger.info(f"Wrote reproducibility_manifest.json -> {out_dir / 'reproducibility_manifest.json'}")

    return weights_all, amps_all, out_dir


if __name__ == "__main__":
    weights, amps, out_dir = generate_dataset()
    print(f"Done. weights={weights.shape} amps={amps.shape}")
    print(f"Saved to: {out_dir}" if out_dir is not None else "save_data=False -- nothing written to disk.")

# =============================================================================
# Reproducibility note
# =============================================================================
#
# `weights` are drawn on CPU via np.random.default_rng(SEED) -- bit-identical
# on any machine given the same seed and matching numpy/pytheus versions, for
# a FULL identical rerun (same n_samples, same batch structure). This is NOT
# independent of subset selection or batch/round structure: sample i's draw
# depends on how many samples were drawn before it in the same rng stream.
# If you need to reproduce one specific sample later, save and reuse its
# `weights` row directly rather than regenerating it from the seed alone.
#
# `amps` are computed on whatever JAX backend is available (GPU or CPU
# fallback); floating-point addition/multiplication is not associative, so
# different hardware/XLA versions can still produce tiny floating-point
# differences, though `weights` remain exact. See REPRODUCIBILITY.md.