"""
Configuration for surrogate inverse optimisation.

Edit this file before running:

    python optimiser.py

This file keeps experiment settings separate from the optimisation logic.
"""

from datetime import date
from pathlib import Path


# =============================================================================
# Main optimisation configuration
# =============================================================================

NPHOTONS = 8
TARGET_NAME = "W"          # "GHZ", "W", "LINEAR_CLUSTER", "SINGLE"
MODEL_TYPE = "HNN"         # "HNN" or "FNN"

DATE = date.today()


OPTIMISER_CONFIG = {
    "description": "inverse optimisation using trained surrogate",

    # -------------------------------------------------------------------------
    # Main setup
    # -------------------------------------------------------------------------
    "n": NPHOTONS,
    "dimensions": 2,
    "target_name": TARGET_NAME,
    "model_type": MODEL_TYPE,

    # -------------------------------------------------------------------------
    # Initial sample data
    # -------------------------------------------------------------------------
    # If True, generate starting samples using data_generation.generate_dataset().
    # If False, load starting samples from conditioned_data_path.
    "generate_data": False,

    "conditioned_data_path": (
        f"/home/bo48god/quick_tests/Experiment_data_collection/"
        f"anchor_noise_dataset_{NPHOTONS}_{TARGET_NAME}/"
        f"anchor_noise_conditioned_dataset.npz"
    ),

    # Optional cap when loading existing initial samples.
    # Use None to use all samples in the conditioned dataset.
    "max_initial_samples": None,

    # Used when generate_data=True
    "data_samples": 100,
    "data_batch_size": 100,
    "data_seed": 4,
    "low_fidelity_threshold": 0.5,
    "save_generated_data": False,
    "data_out_dir": "data/optimiser_generated",
    "normed_data": True,
    "generation_gpu_batch_size": 100,
    "data_shard_size": 100,

    # -------------------------------------------------------------------------
    # Trained surrogate model
    # -------------------------------------------------------------------------
    # For HNN use integer architecture, e.g. 15000.
    # For FNN use tuple/list, e.g. (2000, 2000, 2000).
    "architecture": 15000,

    "model_path": (
        "/home/bo48god/quick_tests/Experiment_data_collection/"
        "Models_11_05_26/params_8node_HNN.msgpack"
    ),

    # Use True if training used normalised target states / normalised model output.
    "normalize_model_output": True,

    # Derived model dimensions
    "input_dim": 2 * NPHOTONS * (NPHOTONS - 1),
    "out_dim": 2 ** NPHOTONS,

    # -------------------------------------------------------------------------
    # Optimisation objective
    # -------------------------------------------------------------------------
    # loss = 1 - fidelity + lambda_l1 * sum(abs(x))
    "lambda_l1": 1e-3,

    # -------------------------------------------------------------------------
    # Optimisation schedule
    # -------------------------------------------------------------------------
    "seed": 46,
    "num_steps": 10_000,
    "early_stop_nn_fid": 0.9999,

    "learning_rate": 1e-2,
    "min_learning_rate": 1e-6,
    "lr_decay_steps": 10_000,
    "lr_exponent": 1.0,

    # Clip optimised graph weights after each update.
    "clip_min": -1.0,
    "clip_max": 1.0,

    # -------------------------------------------------------------------------
    # Verification and logging frequency
    # -------------------------------------------------------------------------
    "print_every": 1,
    "verify_every": 1,

    # -------------------------------------------------------------------------
    # Step-history storage
    # -------------------------------------------------------------------------
    # True stores gradients and update vectors for every step.
    # This is useful for debugging, but can create very large JSON files.
    "store_step_vectors": True,

    # -------------------------------------------------------------------------
    # Pruning configuration
    # -------------------------------------------------------------------------
    "prune_fid_tolerance": 1e-4,
    "prune_thresholds": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    "results_root": str(
        Path(f"/home/bo48god/quick_tests/Experiment_data_collection/optimisation/{NPHOTONS}_node")
    ),
    "folder_name": f"{TARGET_NAME}_100_sample_clean",

    # -------------------------------------------------------------------------
    # Optional metadata
    # -------------------------------------------------------------------------
    "date": str(DATE),
    "k_value": 0,
    "noise": 0,
    "CPU_GPU": "GPU",
}
