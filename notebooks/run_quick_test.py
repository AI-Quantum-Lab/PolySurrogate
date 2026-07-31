"""
Quick end-to-end smoke test for PolySurrogate.

Runs the full 3-stage pipeline with tiny settings (well under a minute on
CPU), writing everything under one results/quick_test/ folder, one
subfolder per stage:
  1. Data generation      -> results/quick_test/data_generation/n4/n4_{i}/dataset_merged.npz
  2. Model training        -> results/quick_test/model_training/PNN/n4/n4_{i}/
  3. Inverse optimisation  -> results/quick_test/inverse_design/GHZ_n4/GHZ_n4_{i}/

All three stages' exact output folders are auto-numbered by data_generate.py
/ ml_model.py / inverse_design.py themselves -- this script never predicts
those paths, it reads back whatever generate_dataset() / train() /
run_optimisation() actually returned.

Usage (run from the project root, or from anywhere -- the project root is
auto-detected below):
    python notebooks/run_quick_test.py

ml_model.py has no cfg-dict entry point -- it configures itself through
module-level constants (see its own "Parameters" section) and exposes a
no-argument train(). This script sets the handful of constants that matter
for a quick test directly on the imported module, then calls train()
unchanged -- the same pattern notebooks/sample_workflow.ipynb uses.
inverse_design.py does accept a cfg dict override via run_optimisation(cfg).
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_generate
import ml_model
import inverse_design

NODES = 4
QUICK_TEST_ROOT = REPO_ROOT / "results" / "quick_test"
DATA_ROOT = QUICK_TEST_ROOT / "data_generation"
MODEL_ROOT = QUICK_TEST_ROOT / "model_training"
OPT_ROOT = QUICK_TEST_ROOT / "inverse_design"


def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def check(path, label: str) -> None:
    p = Path(path)
    if p.exists():
        print(f"  [OK] {label}: {p}  ({p.stat().st_size:,} bytes)")
    else:
        print(f"  [MISSING] {label}: {p}")
        sys.exit(1)


def run_data_generation():
    section("Stage 1 / 3 -- Data generation")
    t0 = time.perf_counter()
    weights, amps, out_dir = data_generate.generate_dataset(
        vertices=NODES, dimensions=2, n_samples=200, batch_size=100, seed=42,
        shard_size=200, normed_data=False, save_data=True, data_root=str(DATA_ROOT),
    )
    elapsed = time.perf_counter() - t0

    merged_path = out_dir / "dataset_merged.npz"
    print(f"  Generated + merged in {elapsed:.1f}s -> {merged_path}")
    print(f"  weights={weights.shape}  amps={amps.shape}")
    check(merged_path, "merged dataset")
    return merged_path


def run_training(data_path):
    section("Stage 2 / 3 -- Model training")

    ml_model.NODES = NODES
    ml_model.MODEL_NAME = "PNN"
    ml_model.HIDDEN_DIM = 400
    ml_model.DATA_PATH = str(data_path)
    ml_model.DATA_SIZE = None
    ml_model.SPLIT_MODE = "contiguous"
    ml_model.BATCH_SIZE = 40   # 200 samples * 0.8 train split = 160, divisible by 40
    ml_model.NUM_EPOCHS = 10
    ml_model.PATIENCE = 10
    ml_model.LR_DECAY_UNTIL_EPOCH = 10
    ml_model.SEED = 42
    ml_model.ROOT_FOLDER = str(MODEL_ROOT)
    ml_model.RESUME_RUN_DIR = None
    ml_model.CONFIG.update(dict(
        nodes=ml_model.NODES, model_name=ml_model.MODEL_NAME, hidden_dim=ml_model.HIDDEN_DIM,
        data_path=ml_model.DATA_PATH, data_size=ml_model.DATA_SIZE, split_mode=ml_model.SPLIT_MODE,
        batch_size=ml_model.BATCH_SIZE, num_epochs=ml_model.NUM_EPOCHS, patience=ml_model.PATIENCE,
        lr_decay_until_epoch=ml_model.LR_DECAY_UNTIL_EPOCH, seed=ml_model.SEED,
    ))

    t0 = time.perf_counter()
    run_dir = ml_model.train()
    elapsed = time.perf_counter() - t0

    print(f"  Training complete in {elapsed:.1f}s -> {run_dir}")
    check(run_dir / "best_params.msgpack", "best_params.msgpack")
    check(run_dir / "config.json", "config.json")
    check(run_dir / "summary.json", "summary.json")
    check(run_dir / "reproducibility_manifest.json", "reproducibility_manifest.json")
    return run_dir


def run_inverse_design(model_dir):
    section("Stage 3 / 3 -- Inverse-design optimisation")

    model_path = model_dir / "best_params.msgpack"
    cfg = dict(inverse_design._default_cfg())
    cfg.update(
        n=NODES, dimensions=2, target_name="GHZ", model_type="PNN",
        generate_data=True, conditioned_data_path="", max_initial_samples=None,
        data_samples=3, data_batch_size=3, data_seed=7, low_fidelity_threshold=0.999,
        normed_data=False, generation_gpu_batch_size=3, data_shard_size=3,
        architecture=400, model_path=str(model_path), normalize_model_output=False,
        input_dim=2 * NODES * (NODES - 1), out_dim=2 ** NODES,
        seed=46, num_steps=10, max_total_steps=10, early_stop_nn_fid=0.9999,
        lr_decay_steps=10, jitter_schedule=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
        jitter_max_events=6, results_root=str(OPT_ROOT),
    )

    t0 = time.perf_counter()
    result_dir = inverse_design.run_optimisation(cfg)
    elapsed = time.perf_counter() - t0

    print(f"  Optimisation complete in {elapsed:.1f}s -> {result_dir}")
    check(Path(result_dir) / "best_graph_solution.json", "best_graph_solution.json")
    check(Path(result_dir) / "optimisation_summary.json", "optimisation_summary.json")
    check(Path(result_dir) / "reproducibility_manifest.json", "reproducibility_manifest.json")
    check(Path(result_dir) / "log.txt", "log.txt")
    return result_dir


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
