"""
Central configuration file for the surrogate-model pipeline.

Edit these values for quick tests or full data-generation runs.
"""

# =============================================================================
# Data generation configuration
# =============================================================================

# Graph / quantum-state setup
VERTICES = 10
DIMENSIONS = 2

# Dataset size
N_SAMPLES = 20_000_000
BATCH_SIZE = 50000

# Reproducibility
SEED = 59

# Saving
SAVE_TO_FILE = True
OUT_DIR = f"data/node{VERTICES}_test"
SHARD_SIZE =  500_000

# Output target type
# True  -> amplitudes are normalized: ||psi||_2 = 1
# False -> raw, unnormalized perfect-matching amplitudes
NORMED_DATA = False
