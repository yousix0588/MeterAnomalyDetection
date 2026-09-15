"""
Run once, after a Slurm array job (run_train_test_anomaly.py --array_count N)
has finished all tasks, to merge each task's ALL_METERS_anomalies_part*.csv
and _run_summary_part*.csv into single final files. Canonical 15-minute score
parts are merged without recalculating their frozen training percentiles.

Usage
-----
    python aggregate_array_results.py --out_dir ./out_august
"""
import argparse
import glob
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()

part_files = sorted(glob.glob(os.path.join(args.out_dir, "ALL_METERS_anomalies_part*.csv")))
summary_files = sorted(glob.glob(os.path.join(args.out_dir, "_run_summary_part*.csv")))
score_files = sorted(glob.glob(os.path.join(args.out_dir, "ALL_METERS_scores_15min_part*.csv")))

if part_files:
    combined = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
    combined = combined.sort_values(["meter_id", "start"]).reset_index(drop=True)
    combined.to_csv(os.path.join(args.out_dir, "ALL_METERS_anomalies.csv"), index=False)
    print(f"Merged {len(part_files)} part files -> "
          f"{os.path.join(args.out_dir, 'ALL_METERS_anomalies.csv')} ({len(combined)} anomaly periods total)")
else:
    print("No ALL_METERS_anomalies_part*.csv files found -- was the array job run with --array_count > 1?")

if score_files:
    scores = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)
    scores = scores.sort_values(["series_id", "interval_start"]).reset_index(drop=True)
    if scores.duplicated(["model", "series_id", "interval_start"]).any():
        raise ValueError("Duplicate canonical VAE score intervals found across array parts")
    score_path = os.path.join(args.out_dir, "ALL_METERS_scores_15min.csv")
    scores.to_csv(score_path, index=False)
    print(f"Merged {len(score_files)} canonical score parts -> {score_path} ({len(scores)} rows)")
else:
    print("No ALL_METERS_scores_15min_part*.csv files found")

if summary_files:
    summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
    summary.to_csv(os.path.join(args.out_dir, "_run_summary.csv"), index=False)
    n_ok = (summary.status == "ok").sum()
    n_skipped = (summary.status == "skipped/failed").sum()
    print(f"Merged {len(summary_files)} summary files -> {n_ok} meters ok, {n_skipped} skipped/failed")
