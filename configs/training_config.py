"""
Configuration for surrogate model training.

Edit this file before running:

    python src/model_training.py

The values below are SMOKE-TEST defaults (4 nodes, 1 000 samples, 20 epochs)
that run end-to-end in a few minutes on CPU.  See the commented-out block at
the bottom for the production HPC values used in the paper.

All paths are relative to the project root (surrogate_model_clean/).
Run scripts from that root directory, or set PYTHONPATH correctly — see README.
"""

TRAINING_CONFIG = {
    # -------------------------------------------------------------------------
    # Experiment identity
    # -------------------------------------------------------------------------
    "NODES": 4,
    "DIMENSIONS": 2,
    "DATE": "smoke_test",

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    "MODEL_NAME": "PNN",      # "FNN" or "PNN"
    # PNN (Polynomial Neural Network): integer hidden dimension.
    # FNN (Feedforward Neural Network): tuple, e.g. (2000, 2000, 2000).
    "HIDDEN_DIM": 400,

    # -------------------------------------------------------------------------
    # Data configuration
    # -------------------------------------------------------------------------
    # Path to the merged .npz produced by merge_shards_to_npz().
    # Must match the OUT_DIR in data_config.py + "/dataset_merged.npz".
    "DATA_PATH": "data/smoke_test/dataset_merged.npz",
    "DATA_SIZE": None,         # None -> use the full dataset

    # -------------------------------------------------------------------------
    # Target/output normalisation
    # -------------------------------------------------------------------------
    # Controls whether the model output is L2-normalised *inside the loss function*
    # before computing MAE.
    #
    #   False (default) -> training target is the raw unnormalised amplitude vector.
    #                      The model directly learns to predict unnormalised vectors.
    #   True            -> both prediction and target are L2-normalised inside the
    #                      loss before computing MAE (extra step in the loss fn).
    #
    # IMPORTANT: This is a SEPARATE setting from NORMED_DATA in data_config.py.
    # They are NOT required to have the same value.
    #   - NORMED_DATA controls what is stored in the dataset file.
    #   - NORMALIZE_MODEL_OUTPUT controls a step inside the loss function at
    #     training time.
    #
    # Consistency requirement:
    #   NORMALIZE_MODEL_OUTPUT here must match normalize_model_output in
    #   optimiser_config.py (both must be the same for correct fidelity
    #   evaluation using the trained model).
    "NORMALIZE_MODEL_OUTPUT": False,

    # -------------------------------------------------------------------------
    # Dataset split
    # -------------------------------------------------------------------------
    "TRAIN_SPLIT": 0.8,
    "VAL_SPLIT": 0.1,
    "TEST_SPLIT": 0.1,

    # -------------------------------------------------------------------------
    # Optimizer / training schedule
    # -------------------------------------------------------------------------
    "LEARNING_RATE": 1e-3,
    "LR_AFTER_DECAY": 1e-5,
    "LR_DECAY_UNTIL_EPOCH": 20,   # smoke test: matches NUM_EPOCHS

    "BATCH_SIZE": 100,
    "NUM_EPOCHS": 20,
    "PATIENCE": 10,
    "TOLERANCE": 1e-7,
    "INIT_KEY": 159,

    # -------------------------------------------------------------------------
    # Checkpoint / resume configuration
    # -------------------------------------------------------------------------
    # Set RESUME_FULL_STATE=True and point CKPT_DIR_RESTORE at an existing
    # checkpoints/ folder to resume a previous run.
    # NOTE: if the run folder already exists from a previous run, delete it
    # before re-running — orbax will raise ValueError if checkpoints exist.
    "RESUME_FULL_STATE": False,
    "CKPT_DIR_RESTORE": "Models_smoke_test/run_PNN_4_smoke_test/checkpoints",

    # -------------------------------------------------------------------------
    # Output configuration
    # -------------------------------------------------------------------------
    # Outputs are written to <ROOT_FOLDER>/run_<RUN_NAME>/.
    "ROOT_FOLDER": "Models_smoke_test",
    "RUN_NAME": "PNN_4_smoke_test",
    "LOSS_NAME": "mae",
}

# =============================================================================
# Production / HPC values (uncomment and edit DATA_PATH for large runs)
# =============================================================================
# TRAINING_CONFIG.update({
#     "NODES": 10,
#     "DATE": "26_05_26",
#     "HIDDEN_DIM": 45000,
#     "DATA_PATH": "data/node10_test/dataset_merged.npz",
#     "DATA_SIZE": 5_000_000,
#     "LR_DECAY_UNTIL_EPOCH": 5000,
#     "BATCH_SIZE": 5000,
#     "NUM_EPOCHS": 20000,
#     "PATIENCE": 2000,
#     "ROOT_FOLDER": "./Models_26_05_26",
#     "RUN_NAME": "PNN_10_5M",
# })
