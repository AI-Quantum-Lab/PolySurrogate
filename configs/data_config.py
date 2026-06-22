"""
Central configuration file for the surrogate-model pipeline.

Edit these values for quick tests or full data-generation runs.
"""

# =============================================================================
# Data generation configuration
# =============================================================================

# Graph / quantum-state setup
VERTICES = 4
DIMENSIONS = 2

# Dataset size
N_SAMPLES = 1000
BATCH_SIZE = 100

# Reproducibility
SEED = 0

# Saving
SAVE_TO_FILE = True
OUT_DIR = 'data/smoke_test'
SHARD_SIZE = 1000

# Output target type
# False (default) -> raw unnormalised amplitude vectors from the simulator.
#                    This is what PNN is trained to predict.
# True  -> amplitudes are L2-normalised so ||psi||_2 = 1 before saving.
#
# Must match 'normed_data' in optimiser_config.py (they control the same
# type of starting samples used in optimisation).
# Does NOT need to match NORMALIZE_MODEL_OUTPUT in training_config.py
# (that controls a separate normalisation step inside the loss function).
NORMED_DATA = False
