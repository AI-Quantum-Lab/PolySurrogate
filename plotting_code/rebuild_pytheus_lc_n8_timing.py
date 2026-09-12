"""
Rebuilds Data/runtime_data/pytheus_data_time/LC/8n_time_info.json (the file
runtime.ipynb reads for the LC n=8 PyTheus timing bar) from the complete,
verified 50-sample benchmark CSV tracked alongside it in this same directory.

Why this script exists: the JSON previously in place was a stale 46-sample
mid-run progress snapshot, captured while the original SLURM array job
(8438291) was still executing Cluster N=8 and before it hit its 2-day
timeout at sample 11. The job was resumed (8531784, samples 12-49) and
finished on 2026-08-09, producing the complete, correct 50-row CSV -- but
the plotting JSON was never regenerated from it until now (2026-09-12).

Source of truth: Cluster_N8_stage1.csv in this same directory, a verified
byte-identical copy (sha256 96278659fe0791d752cb0063657e41cc812bbdae5bbc0ebe
57d1daffbc06e432) of
Time_pytheus/pytheus_time/PyTheus-main/benchmark/results/full_benchmark/Cluster_N8_stage1.csv.
That original file is never modified by this script or anything else --
copying it in here just makes this repo self-contained and able to rebuild
its own derived data without depending on a sibling project's path existing.

Run this script whenever Cluster_N8_stage1.csv changes (e.g. more/rerun
samples), then re-execute runtime.ipynb to regenerate the figure.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "Data" / "runtime_data" / "pytheus_data_time" / "LC" / "Cluster_N8_stage1.csv"
JSON_PATH = HERE / "Data" / "runtime_data" / "pytheus_data_time" / "LC" / "8n_time_info.json"

EXPECTED_SAMPLES = 50


def load_stage1_times(csv_path: Path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} rows in {csv_path}, found {len(rows)}"
        )

    samples = sorted(int(r["sample"]) for r in rows)
    if samples != list(range(EXPECTED_SAMPLES)):
        raise ValueError(f"Expected samples 0..{EXPECTED_SAMPLES - 1}, got {samples}")

    not_completed = [r["sample"] for r in rows if r["completed"] != "True"]
    if not_completed:
        raise ValueError(f"Samples not marked completed=True: {not_completed}")

    errored = [r["sample"] for r in rows if r["error"]]
    if errored:
        raise ValueError(f"Samples with a non-empty error field: {errored}")

    rows.sort(key=lambda r: int(r["sample"]))
    return [float(r["stage1_total_time"]) for r in rows]


def build_json(times):
    return {
        "Time_pre_opt": times,
        "_source": "Time_pytheus benchmark/results/full_benchmark, Stage 1 only (Stage 2 blocked)",
        "_source_file": "benchmark/results/full_benchmark/Cluster_N8_stage1.csv",
        "_n_samples": len(times),
        "_note": (
            "Time_pre_opt here = per-sample stage1_total_time (genuine per-sample "
            "duration, not cumulative): loss setup + complete-graph optimization + "
            "bulk-threshold deletion + truncated-graph re-optimization + save, timed "
            "via time.perf_counter() per sample. Rebuilt by "
            "rebuild_pytheus_lc_n8_timing.py from the final, complete CSV (job 8438291 "
            "original samples 0-11 + resume job 8531784 samples 12-49, both "
            "stage2=False, same Cluster_8.json config) -- replaces a stale 46-sample "
            "mid-run progress snapshot captured before the original job's 2-day "
            "timeout forced the resume."
        ),
        "_generated": datetime.now(timezone.utc).isoformat(),
    }


def main():
    times = load_stage1_times(CSV_PATH)
    payload = build_json(times)

    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Read {len(times)} samples from {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"  mean = {sum(times) / len(times):.10f}")
    print(f"  min  = {min(times):.10f}")
    print(f"  max  = {max(times):.10f}")


if __name__ == "__main__":
    main()
