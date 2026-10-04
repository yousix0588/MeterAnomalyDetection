#!/usr/bin/env python3
"""
Compare All Models on Injected Meters vs Original Meters.
Computes Precision, Recall, F1, and Score Amplification for each model and ensemble.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJECTED_DIR = PROJECT_ROOT / "data" / "injected_meters"
OUT_DIR = PROJECT_ROOT / "runs" / "ensemble" / "injected_comparison"


def load_ground_truth(injected_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Ground Truth anomaly flags for each series."""
    gt_map = {}
    for p in sorted(injected_dir.glob("*.csv")):
        if p.name.startswith("_"):
            continue
        sid = p.stem
        df = pd.read_csv(p)
        if "timestamp" not in df.columns or "pRealKw" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if "pRealKw_original" in df.columns:
            diff = (pd.to_numeric(df["pRealKw"], errors="coerce") -
                    pd.to_numeric(df["pRealKw_original"], errors="coerce")).abs()
            is_anomaly = diff > 1e-4
            kw_diff = diff.fillna(0.0)
        else:
            is_anomaly = pd.Series(False, index=df.index)
            kw_diff = pd.Series(0.0, index=df.index)

        gt_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "is_injected": is_anomaly,
            "kw_diff": kw_diff,
        }).set_index("timestamp")
        gt_map[sid] = gt_df
    return gt_map


def load_model_scores(paths: list[Path]) -> pd.DataFrame:
    """Load canonical score files."""
    frames = []
    for p in paths:
        if p.is_file():
            frames.append(pd.read_csv(p))
        elif p.is_dir():
            for sub in p.glob("*_scores_15min.csv"):
                frames.append(pd.read_csv(sub))
            for sub in [p / "scores_15min.csv", p / "ALL_METERS_scores_15min.csv", p / "ALL_SCORES_15MIN.csv"]:
                if sub.exists():
                    frames.append(pd.read_csv(sub))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model", "series_id", "interval_start"])
    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True)
    return df


def evaluate_model(
    model_name: str,
    injected_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    gt_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a single model against its original baseline and Ground Truth."""
    inj = injected_df[injected_df["model"] == model_name].copy()
    orig = orig_df[orig_df["model"] == model_name].copy()

    if inj.empty or orig.empty:
        logging.warning(f"No records for model {model_name}")
        return pd.DataFrame(), {}

    orig_sub = orig[["series_id", "interval_start", "raw_score", "is_anomaly"]].rename(
        columns={"raw_score": "raw_score_orig", "is_anomaly": "is_anomaly_orig"}
    )
    merged = pd.merge(inj, orig_sub, on=["series_id", "interval_start"], how="inner")

    # Match ground truth
    gt_flags = []
    for _, row in merged.iterrows():
        sid = row["series_id"]
        start = row["interval_start"]
        end = row["interval_end"]
        if sid in gt_map:
            meter_gt = gt_map[sid]
            window_gt = meter_gt.loc[(meter_gt.index >= start) & (meter_gt.index < end)]
            gt_flags.append(bool(window_gt["is_injected"].any()) if not window_gt.empty else False)
        else:
            gt_flags.append(False)

    merged["ground_truth_injected"] = gt_flags
    merged["is_anomaly"] = merged["is_anomaly"].astype(str).str.lower().isin(["true", "1"])
    merged["is_anomaly_orig"] = merged["is_anomaly_orig"].astype(str).str.lower().isin(["true", "1"])
    merged["delta_score"] = merged["raw_score"] - merged["raw_score_orig"]
    merged["newly_triggered"] = merged["is_anomaly"] & (~merged["is_anomaly_orig"])

    tp = int((merged["is_anomaly"] & merged["ground_truth_injected"]).sum())
    fp = int((merged["is_anomaly"] & ~merged["ground_truth_injected"]).sum())
    fn = int((~merged["is_anomaly"] & merged["ground_truth_injected"]).sum())
    tn = int((~merged["is_anomaly"] & ~merged["ground_truth_injected"]).sum())
    total_gt = int(merged["ground_truth_injected"].sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    mean_inj_score = merged[merged["ground_truth_injected"]]["raw_score"].mean()
    mean_norm_score = merged[~merged["ground_truth_injected"]]["raw_score"].mean()
    mean_delta_gt = merged[merged["ground_truth_injected"]]["delta_score"].mean()
    newly_trig_cnt = int(merged["newly_triggered"].sum())

    # Delta detection (score increased by > 10% or > 2.0)
    delta_detected = (merged["delta_score"] > 1.0) & merged["ground_truth_injected"]
    delta_recall = delta_detected.sum() / total_gt if total_gt > 0 else 0.0

    metrics = {
        "model": model_name,
        "total_windows": len(merged),
        "gt_injected_windows": total_gt,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "newly_triggered_anomalies": newly_trig_cnt,
        "delta_recall_gt1": round(delta_recall, 4),
        "mean_score_injected_period": round(float(mean_inj_score), 2) if np.isfinite(mean_inj_score) else 0.0,
        "mean_score_normal_period": round(float(mean_norm_score), 2) if np.isfinite(mean_norm_score) else 0.0,
        "score_amplification": round(float(mean_inj_score / (mean_norm_score + 1e-6)), 2) if np.isfinite(mean_inj_score) and np.isfinite(mean_norm_score) else 0.0,
        "mean_delta_in_anomaly_period": round(float(mean_delta_gt), 2) if np.isfinite(mean_delta_gt) else 0.0,
    }
    return merged, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    gt_map = load_ground_truth(INJECTED_DIR)
    logging.info(f"Loaded ground truth for {len(gt_map)} series.")

    # Injected paths
    lstm_ae_inj = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "injected" / "scores_15min.csv"
    rpca_inj = PROJECT_ROOT / "runs" / "rpca" / "injected"
    mp_inj = PROJECT_ROOT / "runs" / "matrix_profile" / "injected" / "aggregate" / "ALL_SCORES_15MIN.csv"

    # Original paths
    lstm_ae_orig = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "scores_15min.csv"
    rpca_orig = PROJECT_ROOT / "runs" / "rpca" / "august"
    mp_orig = PROJECT_ROOT / "runs" / "matrix_profile" / "results" / "matrix_profile_august_fixed_july" / "aggregate" / "ALL_SCORES_15MIN.csv"

    injected_all = load_model_scores([lstm_ae_inj, rpca_inj, mp_inj])
    original_all = load_model_scores([lstm_ae_orig, rpca_orig, mp_orig])

    logging.info(f"Loaded injected scores: {len(injected_all)} rows across models: {injected_all['model'].unique()}")
    logging.info(f"Loaded original scores: {len(original_all)} rows across models: {original_all['model'].unique()}")

    summary = []
    for model_name in ["lstm_autoencoder", "rpca", "matrix_profile"]:
        merged, metrics = evaluate_model(model_name, injected_all, original_all, gt_map)
        if metrics:
            summary.append(metrics)
            merged.to_csv(out_dir / f"{model_name}_comparison_details.csv", index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "all_models_injected_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("ALL MODELS INJECTED ANOMALY DETECTION COMPARISON:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
