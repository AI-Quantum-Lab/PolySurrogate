"""
Configuration for surrogate inverse optimisation.

Edit this file before running:

    python src/optimiser.py

The values below are SMOKE-TEST defaults (4 photons, GHZ target, generate_data=True)
that run without any pre-existing dataset file.  The only thing you must edit
is model_path — point it at the params.msgpack produced by model_training.py.

See the commented-out block at the bottom for production / HPC values.
All paths are relative to the project root (surrogate_model_clean/).
"""

from datetime import date


# =============================================================================
# Top-level knobs (change these to switch target / node count / model type)
# =============================================================================

NPHOTONS = 4
TARGET_NAME = "GHZ"           # "GHZ" | "W" | "LINEAR_CLUSTER" | "SINGLE" | "ZERO"
MODEL_TYPE = "HNN"             # "HNN" or "FNN"

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
    # generate_data=True  -> fresh random starting graphs (no file needed)
    # generate_data=False -> load from conditioned_data_path
    "generate_data": True,

    # Only used when generate_data=False.
    # Edit this path to point at your conditioned dataset.
    "conditioned_data_path": "data/optimiser_conditioned/conditioned_dataset.npz",

    # Optional cap on loaded samples when generate_data=False.
    "max_initial_samples": None,

    # Settings used when generate_data=True
    "data_samples": 5,
    "data_batch_size": 5,
    "data_seed": 4,
    "low_fidelity_threshold": 0.999,
    "save_generated_data": False,
    "data_out_dir": "data/optimiser_generated",
    "normed_data": True,
    "generation_gpu_batch_size": 5,
    "data_shard_size": 5,

    # -------------------------------------------------------------------------
    # Trained surrogate model
    # -------------------------------------------------------------------------
    # HNN: integer hidden dim matching training_config HIDDEN_DIM.
    # FNN: tuple/list matching training_config HIDDEN_DIM.
    "architecture": 400,

    # Point this at the params.msgpack produced by model_training.py.
    # Example (smoke-test default):
    "model_path": "Models_smoke_test/run_HNN_4_smoke_test/params.msgpack",

    # Must match NORMALIZE_MODEL_OUTPUT used during training.
    "normalize_model_output": False,

    # Derived automatically from NPHOTONS — do not change unless you change n.
    "input_dim": 2 * NPHOTONS * (NPHOTONS - 1),
    "out_dim": 2 ** NPHOTONS,

    # -------------------------------------------------------------------------
    # Optimisation objective
    # -------------------------------------------------------------------------
    # loss = (1 - fidelity) + lambda_l1 * sum(abs(x))
    "lambda_l1": 1e-3,

    # -------------------------------------------------------------------------
    # Optimisation schedule
    # -------------------------------------------------------------------------
    "seed": 46,
    "num_steps": 20,            # smoke test: increase to 10 000 for real runs
    "early_stop_nn_fid": 0.9999,

    "learning_rate": 1e-2,
    "min_learning_rate": 1e-6,
    "lr_decay_steps": 20,       # smoke test: match num_steps
    "lr_exponent": 1.0,

    # Clip graph weights to this range after each update.
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
    # Creates large JSON files for long runs — set False for production.
    "store_step_vectors": False,

    # -------------------------------------------------------------------------
    # Pruning configuration
    # -------------------------------------------------------------------------
    "prune_fid_tolerance": 1e-4,
    "prune_thresholds": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    # Results are written to <results_root>/<folder_name>/.
    "results_root": "optimiser_notebook_results",
    "folder_name": f"{TARGET_NAME}_{NPHOTONS}_smoke_test",

    # -------------------------------------------------------------------------
    # Optional metadata
    # -------------------------------------------------------------------------
    "date": str(DATE),
    "k_value": 0,
    "noise": 0,
    "CPU_GPU": "CPU",
}

# =============================================================================
# Production / HPC values (uncomment and edit model_path for large runs)
# =============================================================================
# NPHOTONS_PROD = 8
# OPTIMISER_CONFIG.update({
#     "n": NPHOTONS_PROD,
#     "target_name": "GHZ",
#     "generate_data": False,
#     "conditioned_data_path": "data/anchor_noise_dataset_8_GHZ/conditioned_dataset.npz",
#     "architecture": 15000,
#     "model_path": "Models_prod/run_HNN_8/params.msgpack",
#     "normalize_model_output": True,
#     "input_dim": 2 * NPHOTONS_PROD * (NPHOTONS_PROD - 1),
#     "out_dim": 2 ** NPHOTONS_PROD,
#     "num_steps": 10_000,
#     "lr_decay_steps": 10_000,
#     "store_step_vectors": False,
#     "results_root": f"optimiser_results/{NPHOTONS_PROD}_node",
#     "folder_name": "GHZ_100_sample_prod",
#     "CPU_GPU": "GPU",
# })
