"""
Quick end-to-end smoke test for PolySurrogate.

Runs the full 3-stage pipeline with tiny settings (~10s on CPU), writing
everything under one results/ folder, one subfolder per stage:
  1. Data generation      -> results/quick_test/data_generation/n4/n4_{i}/dataset_merged.npz
  2. Model training        -> results/quick_test/model_training/PNN/n4/n4_{i}/
  3. Inverse optimisation  -> results/quick_test/inverse_design/GHZ_n4/GHZ_n4_{i}/

All three stages' exact output folders are auto-numbered by
01_data_generate.py / 02_ml_model.py / 03_inverse_design.py themselves --
this script never predicts those paths, it reads back whatever
generate_dataset() / train_surrogate_model() / run_optimisation() actually
returned.

Usage (run from the project root):
    python notebooks/run_quick_test.py

No PYTHONPATH setup needed -- the repo root is added to sys.path below, then
01_data_generate.py, 02_ml_model.py, and 03_inverse_design.py are loaded via
importlib.import_module (a literal `import 01_data_generate` isn't valid
Python, since identifiers can't start with a digit).
"""

import importlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_stage(module_name: str):
    """Load one of the numbered pipeline scripts by module name (as a
    string). A literal `import 01_data_generate` isn't valid Python
    (identifiers can't start with a digit), but importlib.import_module
    works fine since it takes the name as a string, not bare syntax."""
    return importlib.import_module(module_name)


NODES = 4
QUICK_TEST_ROOT = "results/quick_test"
DATA_ROOT = f"{QUICK_TEST_ROOT}/data_generation"
MODEL_ROOT = f"{QUICK_TEST_ROOT}/model_training"
OPT_ROOT = f"{QUICK_TEST_ROOT}/inverse_design"


def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def check(path: str, label: str) -> None:
    p = Path(path)
    if p.exists():
        print(f"  [OK] {label}: {path}  ({p.stat().st_size:,} bytes)")
    else:
        print(f"  [MISSING] {label}: {path}")
        sys.exit(1)


def run_data_generation():
    section("Stage 1 / 3 -- Data generation")
    stage1 = _load_stage("01_data_generate")

    t0 = time.perf_counter()
    weights, amps, out_dir = stage1.generate_dataset(
        vertices=NODES, dimensions=2, n_samples=200, batch_size=100, seed=42,
        shard_size=200, normed_data=False, save_data=True, data_root=DATA_ROOT,
    )
    elapsed = time.perf_counter() - t0

    merged_path = out_dir / "dataset_merged.npz"
    print(f"  Generated + merged in {elapsed:.1f}s -> {merged_path}")
    print(f"  weights={weights.shape}  amps={amps.shape}")
    check(str(merged_path), "merged dataset")
    return str(merged_path)


def run_training(data_path: str):
    section("Stage 2 / 3 -- Model training")
    stage2 = _load_stage("02_ml_model")

    cfg = dict(
        NODES=NODES, DIMENSIONS=2, DATE="quick_test", MODEL_NAME="PNN", HIDDEN_DIM=400,
        DATA_PATH=data_path, DATA_SIZE=None, NORMALIZE_MODEL_OUTPUT=False,
        TRAIN_SPLIT=0.8, VAL_SPLIT=0.1, TEST_SPLIT=0.1,
        LEARNING_RATE=1e-3, LR_AFTER_DECAY=1e-5, LR_DECAY_UNTIL_EPOCH=10,
        BATCH_SIZE=50, NUM_EPOCHS=10, PATIENCE=5, TOLERANCE=1e-7, SEED=42,
        SPLIT_MODE="contiguous", PRECISION="float32", DATASET_HASH_MODE="fast",
        DEBUG_TRACE=False, RESUME_FULL_STATE=False, CKPT_DIR_RESTORE="",
        ROOT_FOLDER=MODEL_ROOT, LOSS_NAME="mae",
        DETERMINISTIC_XLA=True,
    )

    t0 = time.perf_counter()
    run_dir = stage2.train_surrogate_model(cfg)
    elapsed = time.perf_counter() - t0

    print(f"  Training complete in {elapsed:.1f}s -> {run_dir}")
    check(f"{run_dir}/params.msgpack", "params.msgpack")
    check(f"{run_dir}/model_info.json", "model_info.json")
    check(f"{run_dir}/reproducibility_manifest.json", "reproducibility_manifest.json")
    check(f"{run_dir}/plots/training_curves.png", "training_curves.png")
    return str(run_dir)


def run_inverse_design(model_dir: str):
    section("Stage 3 / 3 -- Inverse-design optimisation")
    stage3 = _load_stage("03_inverse_design")

    model_path = f"{model_dir}/params.msgpack"
    cfg = dict(
        n=NODES, dimensions=2, target_name="GHZ", model_type="PNN",
        generate_data=True, conditioned_data_path="", max_initial_samples=None,
        data_samples=3, data_batch_size=3, data_seed=7, low_fidelity_threshold=0.999,
        normed_data=False, generation_gpu_batch_size=3, data_shard_size=3,
        architecture=400, model_path=model_path, normalize_model_output=False,
        input_dim=2 * NODES * (NODES - 1), out_dim=2 ** NODES, lambda_l1=1e-3,
        seed=46, num_steps=10, early_stop_nn_fid=0.9999,
        learning_rate=1e-2, min_learning_rate=1e-6, lr_decay_steps=10, lr_exponent=1.0,
        clip_min=-1.0, clip_max=1.0,
        jitter_enabled=True, initial_jitter=0.01,
        jitter_schedule=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
        jitter_max_events=6, stuck_patience_steps=500, stuck_min_improvement=1e-4,
        max_total_steps=10,
        print_every=5, verify_every=5, store_step_vectors=False,
        prune_fid_tolerance=1e-4, prune_thresholds=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        results_root=OPT_ROOT,
    )

    t0 = time.perf_counter()
    result_dir = stage3.run_optimisation(cfg)
    elapsed = time.perf_counter() - t0

    print(f"  Optimisation complete in {elapsed:.1f}s -> {result_dir}")
    check(f"{result_dir}/best_graph_solution.json", "best_graph_solution.json")
    check(f"{result_dir}/optimisation_summary.json", "optimisation_summary.json")
    check(f"{result_dir}/reproducibility_manifest.json", "reproducibility_manifest.json")
    check(f"{result_dir}/log.txt", "log.txt")
    return str(result_dir)


def main():
    wall_start = time.perf_counter()
    print("\nPolySurrogate -- quick end-to-end smoke test")
    print(f"Project root: {REPO_ROOT}")

    data_path = run_data_generation()
    model_dir = run_training(data_path)
    result_dir = run_inverse_design(model_dir)

    total = time.perf_counter() - wall_start
    section(f"ALL STAGES PASSED  ({total:.1f}s total)")
    print(f"  Data      : {data_path}")
    print(f"  Model     : {model_dir}/")
    print(f"  Results   : {result_dir}/")
    print()


if __name__ == "__main__":
    main()
