"""
Quick end-to-end smoke test for surrogate_model_clean.

Runs the full pipeline with tiny settings (~10 s on CPU):
  1. Data generation   -> data/quick_test/
  2. Shard merging     -> data/quick_test/dataset_merged.npz
  3. Model training    -> Models_quick_test/run_PNN_4_quick_test/
  4. Inverse optimisation -> optimiser_quick_test/GHZ_4_quick_test/

Usage (run from the project root):
    python scripts/run_quick_test.py

PYTHONPATH must include src/ and configs/.  The easiest way is:
    export PYTHONPATH="$PWD/src:$PWD/configs:$PYTHONPATH"
    python scripts/run_quick_test.py

Or activate the virtual environment first:
    source .venv/bin/activate
    PYTHONPATH="$PWD/src:$PWD/configs" python scripts/run_quick_test.py
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ and configs/ are importable when this script is run from root.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "configs"))

# ---------------------------------------------------------------------------
# Quick-test configuration (all values set here; configs/*.py are not used)
# ---------------------------------------------------------------------------
NODES = 4
DATA_DIR = "data/quick_test"
DATA_MERGED = f"{DATA_DIR}/dataset_merged.npz"
MODEL_ROOT = "Models_quick_test"
MODEL_RUN = "PNN_4_quick_test"
MODEL_PATH = f"{MODEL_ROOT}/run_{MODEL_RUN}/params.msgpack"
OPT_ROOT = "optimiser_quick_test"
OPT_FOLDER = "GHZ_4_quick_test"

DATAGEN_CFG = dict(
    vertices=NODES,
    dimensions=2,
    n_samples=200,
    batch_size=100,
    seed=42,
    save_to_file=True,
    out_dir_path=DATA_DIR,
    shard_size=200,
    normed_data=False,   # project trains on unnormalised amplitude vectors
)

TRAINING_CFG = {
    "NODES": NODES,
    "DIMENSIONS": 2,
    "DATE": "quick_test",
    "MODEL_NAME": "PNN",
    "HIDDEN_DIM": 400,
    "DATA_PATH": DATA_MERGED,
    "DATA_SIZE": None,
    "NORMALIZE_MODEL_OUTPUT": False,
    "TRAIN_SPLIT": 0.8,
    "VAL_SPLIT": 0.1,
    "TEST_SPLIT": 0.1,
    "LEARNING_RATE": 1e-3,
    "LR_AFTER_DECAY": 1e-5,
    "LR_DECAY_UNTIL_EPOCH": 10,
    "BATCH_SIZE": 50,
    "NUM_EPOCHS": 10,
    "PATIENCE": 5,
    "TOLERANCE": 1e-7,
    "INIT_KEY": 42,
    "RESUME_FULL_STATE": False,
    "CKPT_DIR_RESTORE": "",
    "ROOT_FOLDER": MODEL_ROOT,
    "RUN_NAME": MODEL_RUN,
    "LOSS_NAME": "mae",
}

OPTIMISER_CFG = {
    "n": NODES,
    "dimensions": 2,
    "target_name": "GHZ",
    "model_type": "PNN",
    "generate_data": True,
    "conditioned_data_path": "",
    "max_initial_samples": None,
    "data_samples": 3,
    "data_batch_size": 3,
    "data_seed": 7,
    "low_fidelity_threshold": 0.999,
    "save_generated_data": False,
    "data_out_dir": "data/optimiser_generated_quick",
    "normed_data": False,   # must match normed_data in data generation
    "generation_gpu_batch_size": 3,
    "data_shard_size": 3,
    "architecture": 400,
    "model_path": MODEL_PATH,
    "normalize_model_output": False,   # must match NORMALIZE_MODEL_OUTPUT in training
    "input_dim": 2 * NODES * (NODES - 1),
    "out_dim": 2 ** NODES,
    "lambda_l1": 1e-3,
    "seed": 46,
    "num_steps": 10,
    "early_stop_nn_fid": 0.9999,
    "learning_rate": 1e-2,
    "min_learning_rate": 1e-6,
    "lr_decay_steps": 10,
    "lr_exponent": 1.0,
    "clip_min": -1.0,
    "clip_max": 1.0,
    "print_every": 5,
    "verify_every": 5,
    "store_step_vectors": False,
    "prune_fid_tolerance": 1e-4,
    "prune_thresholds": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "results_root": OPT_ROOT,
    "folder_name": OPT_FOLDER,
    "date": "",
    "k_value": 0,
    "noise": 0,
    "CPU_GPU": "CPU",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def check(path: str, label: str) -> None:
    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        print(f"  [OK] {label}: {path}  ({size:,} bytes)")
    else:
        print(f"  [MISSING] {label}: {path}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Stage 1: Data generation
# ---------------------------------------------------------------------------

def run_data_generation():
    section("Stage 1 / 4 — Data generation")
    from data_generation import generate_dataset

    t0 = time.perf_counter()
    out_dir = generate_dataset(**DATAGEN_CFG)
    elapsed = time.perf_counter() - t0

    shards = sorted(Path(out_dir).glob("data_*.npz"))
    print(f"  Generated {len(shards)} shard(s) in {elapsed:.1f}s -> {out_dir}")
    for s in shards:
        print(f"    {s.name}")
    return out_dir


# ---------------------------------------------------------------------------
# Stage 2: Merge shards
# ---------------------------------------------------------------------------

def run_merge(out_dir: str):
    section("Stage 2 / 4 — Shard merging")
    from data_generation_utils import merge_shards_to_npz
    import numpy as np

    t0 = time.perf_counter()
    merged = merge_shards_to_npz(Path(out_dir), "dataset_merged.npz")
    elapsed = time.perf_counter() - t0

    d = np.load(merged)
    print(f"  Merged in {elapsed:.1f}s -> {merged}")
    print(f"  weights: {d['weights'].shape}  amps: {d['amps'].shape}")
    check(str(merged), "merged dataset")
    return str(merged)


# ---------------------------------------------------------------------------
# Stage 3: Model training
# ---------------------------------------------------------------------------

def run_training():
    section("Stage 3 / 4 — Model training")

    # Patch the config module so model_training.py picks up our values.
    import training_config
    training_config.TRAINING_CONFIG = TRAINING_CFG

    # model_training reads the module-level config at import time, so we
    # must reload it after patching.
    import importlib
    import model_training
    importlib.reload(model_training)

    t0 = time.perf_counter()
    run_dir = model_training.train_surrogate_model()
    elapsed = time.perf_counter() - t0

    print(f"  Training complete in {elapsed:.1f}s -> {run_dir}")
    check(f"{run_dir}/params.msgpack", "params.msgpack")
    check(f"{run_dir}/model_info.json", "model_info.json")
    check(f"{run_dir}/plots/training_curves.png", "training_curves.png")
    return str(run_dir)


# ---------------------------------------------------------------------------
# Stage 4: Inverse-design optimisation
# ---------------------------------------------------------------------------

def run_optimisation():
    section("Stage 4 / 4 — Inverse-design optimisation")
    from optimiser import run_optimisation as _run_opt

    t0 = time.perf_counter()
    result_dir = _run_opt(OPTIMISER_CFG)
    elapsed = time.perf_counter() - t0

    print(f"  Optimisation complete in {elapsed:.1f}s -> {result_dir}")
    check(f"{result_dir}/best_graph_solution.json", "best_graph_solution.json")
    check(f"{result_dir}/optimisation_summary.json", "optimisation_summary.json")
    check(f"{result_dir}/log.txt", "log.txt")
    return str(result_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()
    print("\nsurrogate_model_clean — quick end-to-end smoke test")
    print(f"Project root: {REPO_ROOT}")

    data_dir = run_data_generation()
    run_merge(data_dir)
    run_training()
    run_optimisation()

    total = time.perf_counter() - wall_start
    section(f"ALL STAGES PASSED  ({total:.1f}s total)")
    print(f"  Data      : {DATA_DIR}/")
    print(f"  Model     : {MODEL_ROOT}/run_{MODEL_RUN}/")
    print(f"  Results   : {OPT_ROOT}/{OPT_FOLDER}/")
    print()


if __name__ == "__main__":
    main()
