"""
Configuration for surrogate model training.

Edit this file before running:

    python model_training.py
"""

TRAINING_CONFIG = {
    # -------------------------------------------------------------------------
    # Experiment identity
    # -------------------------------------------------------------------------
    "NODES": 10,
    "DIMENSIONS": 2,
    "DATE": "26_05_26",

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    "MODEL_NAME": "HNN",      # "FNN" or "HNN"
    "HIDDEN_DIM": 45000,

    # -------------------------------------------------------------------------
    # Data configuration
    # -------------------------------------------------------------------------
    "DATA_PATH": (
        "/home/bo48god/projects/deep_dreaming/"
        "surrogate_model_clean_all_codes_with_config/"
        "surrogate_model_clean/data/node10_test/dataset_merged.npz"
    ),
    "DATA_SIZE": 5_000_000,   # set to None to use the full dataset

    # -------------------------------------------------------------------------
    # Target/output configuration
    # -------------------------------------------------------------------------
    # True  -> normalize prediction and target before MAE/fidelity metrics
    # False -> train on raw/unnormalized amplitudes
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
    "LR_DECAY_UNTIL_EPOCH": 5000,

    "BATCH_SIZE": 5000,
    "NUM_EPOCHS": 20000,
    "PATIENCE": 2000,
    "TOLERANCE": 1e-7,
    "INIT_KEY": 159,

    # -------------------------------------------------------------------------
    # Checkpoint / resume configuration
    # -------------------------------------------------------------------------
    "RESUME_FULL_STATE": False,
    "CKPT_DIR_RESTORE": (
        "/home/bo48god/projects/deep_dreaming/src/projects/src/"
        "projection_network_node_4_13_04_26/"
        "run_FNN4_MAE_2_3L_300Nd_lrn3_20M/checkpoints"
    ),

    # -------------------------------------------------------------------------
    # Output configuration
    # -------------------------------------------------------------------------
    "ROOT_FOLDER": "./Models_26_05_26",
    "RUN_NAME": "HNN_10_normMAE_5M_8",
    "LOSS_NAME": "mae",
}
