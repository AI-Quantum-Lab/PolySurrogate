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
SHARD_SIZE =  1000

# Output target type
# True  -> amplitudes are normalized: ||psi||_2 = 1
# False -> raw, unnormalized perfect-matching amplitudes
NORMED_DATA = True
