#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJECTED_EVAL = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "injected_evaluation"
OUT_DIR = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "injected"
METADATA_FILE = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "metadata.json"
CALIBRATION_FILE = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "calibration_scores.npz"

OUT_DIR.mkdir(parents=True, exist_ok=True)

raw_df = pd.read_csv(INJECTED_EVAL / "injected_scores_15min.csv")
meta = json.loads(METADATA_FILE.read_text())
threshold = float(meta["threshold"])

# Load frozen calibration to compute exact max_percentile and mean_percentile
calib = np.load(CALIBRATION_FILE)
ref_max = np.sort(calib["max_scores"])
ref_mean = np.sort(calib["mean_scores"])

def frozen_ecdf(vals, ref):
    v = np.asarray(vals, dtype=float)
    return np.searchsorted(ref, v, side="right") / ref.size

max_scores = raw_df["raw_score_injected"].to_numpy()
mean_scores = raw_df["mean_score_injected"].to_numpy()

raw_df["model"] = "lstm_autoencoder"
raw_df["raw_score"] = max_scores
raw_df["anomaly_score"] = max_scores
raw_df["max_score"] = max_scores
raw_df["mean_score"] = mean_scores
raw_df["score_std"] = 0.0
raw_df["max_percentile"] = frozen_ecdf(max_scores, ref_max).clip(0.0, 1.0)
raw_df["mean_percentile"] = frozen_ecdf(mean_scores, ref_mean).clip(0.0, 1.0)
raw_df["threshold"] = threshold
raw_df["is_anomaly"] = raw_df["raw_score"] > threshold
raw_df["valid_point_count"] = 3
raw_df["expected_point_count"] = 3
raw_df["coverage_ratio"] = 1.0
raw_df["source_resolution_minutes"] = 5
raw_df["aggregation_method"] = "last_15min_of_causal_24h_window"
raw_df["data_status"] = "native_5m"
raw_df["calibration_version"] = "train_pre_2026-08-01_v1"
raw_df["available"] = True

canonical_cols = [
    "model", "series_id", "interval_start", "interval_end", "raw_score",
    "anomaly_score", "max_score", "mean_score", "score_std", "max_percentile",
    "mean_percentile", "threshold", "is_anomaly", "valid_point_count",
    "expected_point_count", "coverage_ratio", "source_resolution_minutes",
    "aggregation_method", "data_status", "calibration_version", "available"
]
canonical_df = raw_df[canonical_cols].sort_values(["series_id", "interval_start"])
canonical_df.to_csv(OUT_DIR / "scores_15min.csv", index=False)
print(f"Exported canonical LSTM AE scores: {len(canonical_df)} rows to {OUT_DIR / 'scores_15min.csv'}")
