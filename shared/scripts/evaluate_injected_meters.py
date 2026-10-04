#!/usr/bin/env python3
"""
Evaluate Injected Meters using trained LSTM Autoencoder.
Compares reconstruction errors between original and anomaly-injected meter data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure group14reshape root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.lstm_autoencoder.src.data import (
    MeterWindowDataset,
    clean_series,
    chronological_date_split,
    discover_files,
    load_series,
    parse_series_id,
)
from methods.lstm_autoencoder.src.model import LSTMAutoencoder
from methods.lstm_autoencoder.src.pipeline import (
    frozen_ecdf,
    score_interval_blocks,
    select_device,
)
from methods.lstm_autoencoder.src.utils import load_config


def extract_ground_truth(
    injected_dir: Path,
    timestamp_col: str = "timestamp",
    power_col: str = "pRealKw",
    orig_power_col: str = "pRealKw_original",
) -> dict[str, pd.DataFrame]:
    """Extract ground-truth anomaly timestamps where pRealKw != pRealKw_original."""
    gt_map: dict[str, pd.DataFrame] = {}
    csv_files = discover_files(injected_dir, [".csv"])
    logging.info(f"Extracting Ground Truth from {len(csv_files)} files in {injected_dir}...")

    for path in csv_files:
        series_id = parse_series_id(path)
        df = pd.read_csv(path)
        if timestamp_col not in df.columns or power_col not in df.columns:
            continue
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

        has_gt = orig_power_col in df.columns
        if has_gt:
            diff = (pd.to_numeric(df[power_col], errors="coerce") -
                    pd.to_numeric(df[orig_power_col], errors="coerce")).abs()
            is_anomaly = diff > 1e-4
            kw_diff = diff.fillna(0.0)
        else:
            is_anomaly = pd.Series(False, index=df.index)
            kw_diff = pd.Series(0.0, index=df.index)

        gt_df = pd.DataFrame({
            "timestamp": df[timestamp_col],
            "pRealKw": pd.to_numeric(df[power_col], errors="coerce"),
            "pRealKw_original": pd.to_numeric(df[orig_power_col], errors="coerce") if has_gt else np.nan,
            "kw_diff": kw_diff,
            "is_injected": is_anomaly,
        }).set_index("timestamp")
        gt_map[series_id] = gt_df

    return gt_map


def build_injected_dataset(
    config: dict[str, Any],
    injected_dir: Path,
    scaler: Any,
) -> tuple[MeterWindowDataset, dict[str, Any]]:
    """Build causal evaluation dataset for August from injected_meters directory."""
    d = config["data"]
    split = config["splits"]
    block_size = d.get("score_block_size", d["stride"])
    min_baseline = d.get("window_size", 288)

    logging.info(f"Loading raw series from {injected_dir}...")
    raw = load_series(
        injected_dir,
        d["extensions"],
        d["timestamp_column"],
        d["features"],
        d.get("meter_limit"),
    )

    frames: dict[str, pd.DataFrame] = {}
    skipped = []

    for key, frame in raw.items():
        try:
            train_raw, val_raw, test_raw = chronological_date_split(
                frame,
                split["train_end"],
                split["test_start"],
                split["test_end"],
                split["timezone"],
                split["validation_ratio"],
                min_baseline_rows=min_baseline,
            )
            test_clean = clean_series(test_raw, d["frequency"], d["max_interpolation_gap"])[0]
            if test_clean.dropna().empty:
                skipped.append((key, "empty_test"))
                continue
            baseline = clean_series(
                pd.concat([train_raw, val_raw]).sort_index(),
                d["frequency"],
                d["max_interpolation_gap"],
            )[0]
            context_rows = d["window_size"] - block_size
            if len(baseline) < context_rows:
                skipped.append((key, "short_baseline"))
                continue
            scoring = pd.concat([baseline.tail(context_rows), test_clean]).sort_index()
            full_index = pd.date_range(
                scoring.index.min(), scoring.index.max(), freq=d["frequency"], tz=scoring.index.tz
            )
            frames[key] = scoring.reindex(full_index)
        except ValueError as exc:
            skipped.append((key, str(exc)))
            continue

    logging.info(f"Successfully prepared {len(frames)} series for inference (skipped {len(skipped)}).")
    dataset = MeterWindowDataset.from_frames(
        frames, scaler, d["window_size"], d["stride"], d["max_missing_ratio"]
    )
    return dataset, {"skipped": skipped, "frames": frames}


def run_inference(
    config: dict[str, Any],
    dataset: MeterWindowDataset,
    checkpoint_path: Path,
    threshold: float,
    calibration_path: Path | None = None,
    device_name: str = "auto",
) -> pd.DataFrame:
    """Run model inference on dataset and return scored DataFrame."""
    d, t = config["data"], config["training"]
    device = select_device(device_name if device_name != "auto" else t["device"])
    logging.info(f"Running inference on device: {device} with {len(dataset)} windows...")

    model = LSTMAutoencoder(input_size=len(d["features"]), **config["model"]).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t.get("num_workers", 2),
    )
    block_size = d.get("score_block_size", d["stride"])
    scores = score_interval_blocks(model, loader, device, block_size)

    max_percentiles = np.full(len(dataset), np.nan)
    if calibration_path and calibration_path.exists():
        stored = np.load(calibration_path)
        max_percentiles = frozen_ecdf(scores["max"], stored["max_scores"])

    step = pd.to_timedelta(d["frequency"])
    block_duration = step * block_size
    rows = []
    for index, meta in enumerate(dataset.metadata):
        interval_end = meta.end + step
        interval_start = interval_end - block_duration
        rows.append({
            "series_id": meta.series_id,
            "interval_start": interval_start,
            "interval_end": interval_end,
            "raw_score_injected": float(scores["max"][index]),
            "mean_score_injected": float(scores["mean"][index]),
            "max_percentile_injected": float(max_percentiles[index]) if np.isfinite(max_percentiles[index]) else np.nan,
            "is_anomaly_injected": bool(scores["max"][index] > threshold),
        })

    df = pd.DataFrame(rows)
    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True)
    return df


def evaluate_and_compare(
    injected_scores: pd.DataFrame,
    original_scores_path: Path,
    gt_map: dict[str, pd.DataFrame],
    threshold: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Merge injected scores with original baseline, compare errors, and compute metrics."""
    logging.info(f"Loading original scores from {original_scores_path}...")
    orig_df = pd.read_csv(original_scores_path)
    orig_df["interval_start"] = pd.to_datetime(orig_df["interval_start"], utc=True)
    orig_df["interval_end"] = pd.to_datetime(orig_df["interval_end"], utc=True)

    orig_subset = orig_df[[
        "series_id", "interval_start", "interval_end",
        "raw_score", "mean_score", "is_anomaly"
    ]].rename(columns={
        "raw_score": "raw_score_original",
        "mean_score": "mean_score_original",
        "is_anomaly": "is_anomaly_original",
    })

    # Merge on (series_id, interval_start)
    merged = pd.merge(
        injected_scores,
        orig_subset,
        on=["series_id", "interval_start"],
        how="inner",
        suffixes=("", "_orig"),
    )

    # Compute delta and ratio
    merged["delta_error"] = merged["raw_score_injected"] - merged["raw_score_original"]
    merged["error_ratio"] = merged["raw_score_injected"] / (merged["raw_score_original"] + 1e-6)
    merged["threshold"] = threshold

    # Match ground truth for each 15-min interval
    logging.info("Matching ground truth anomaly labels per 15-minute interval...")
    gt_flags = []
    max_kw_diffs = []

    for _, row in merged.iterrows():
        sid = row["series_id"]
        start = row["interval_start"]
        end = row["interval_end"]
        if sid in gt_map:
            meter_gt = gt_map[sid]
            window_gt = meter_gt.loc[(meter_gt.index >= start) & (meter_gt.index < end)]
            has_inj = window_gt["is_injected"].any() if not window_gt.empty else False
            max_diff = window_gt["kw_diff"].max() if not window_gt.empty else 0.0
            gt_flags.append(bool(has_inj))
            max_kw_diffs.append(float(max_diff))
        else:
            gt_flags.append(False)
            max_kw_diffs.append(0.0)

    merged["ground_truth_injected"] = gt_flags
    merged["max_injected_kw_diff"] = max_kw_diffs
    merged["newly_triggered"] = merged["is_anomaly_injected"] & (~merged["is_anomaly_original"])

    # Summary per meter
    summary_rows = []
    for sid, group in merged.groupby("series_id"):
        n_windows = len(group)
        n_gt_injected = group["ground_truth_injected"].sum()

        inj_period = group[group["ground_truth_injected"]]
        norm_period = group[~group["ground_truth_injected"]]

        # Mean errors
        mean_err_orig = group["raw_score_original"].mean()
        mean_err_inj = group["raw_score_injected"].mean()
        mean_delta = group["delta_error"].mean()

        mean_err_in_anomaly = inj_period["raw_score_injected"].mean() if not inj_period.empty else np.nan
        mean_err_in_normal = norm_period["raw_score_injected"].mean() if not norm_period.empty else np.nan
        mean_delta_in_anomaly = inj_period["delta_error"].mean() if not inj_period.empty else np.nan

        # Detection by threshold
        tp = (group["is_anomaly_injected"] & group["ground_truth_injected"]).sum()
        fp = (group["is_anomaly_injected"] & ~group["ground_truth_injected"]).sum()
        fn = (~group["is_anomaly_injected"] & group["ground_truth_injected"]).sum()
        tn = (~group["is_anomaly_injected"] & ~group["ground_truth_injected"]).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Detection by delta error > 5.0 (significant error increase)
        delta_detected = (group["delta_error"] > 5.0) & group["ground_truth_injected"]
        delta_recall = delta_detected.sum() / n_gt_injected if n_gt_injected > 0 else 0.0

        summary_rows.append({
            "series_id": sid,
            "windows": n_windows,
            "gt_injected_windows": int(n_gt_injected),
            "orig_anomalies": int(group["is_anomaly_original"].sum()),
            "injected_anomalies": int(group["is_anomaly_injected"].sum()),
            "newly_triggered": int(group["newly_triggered"].sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "delta_recall_gt5": round(delta_recall, 4),
            "mean_score_orig": round(mean_err_orig, 2),
            "mean_score_injected": round(mean_err_inj, 2),
            "mean_delta_error": round(mean_delta, 2),
            "mean_error_in_anomaly_period": round(mean_err_in_anomaly, 2) if np.isfinite(mean_err_in_anomaly) else np.nan,
            "mean_error_in_normal_period": round(mean_err_in_normal, 2) if np.isfinite(mean_err_in_normal) else np.nan,
            "mean_delta_in_anomaly_period": round(mean_delta_in_anomaly, 2) if np.isfinite(mean_delta_in_anomaly) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("gt_injected_windows", ascending=False)

    # Global metrics
    total_tp = int((merged["is_anomaly_injected"] & merged["ground_truth_injected"]).sum())
    total_fp = int((merged["is_anomaly_injected"] & ~merged["ground_truth_injected"]).sum())
    total_fn = int((~merged["is_anomaly_injected"] & merged["ground_truth_injected"]).sum())
    total_tn = int((~merged["is_anomaly_injected"] & ~merged["ground_truth_injected"]).sum())
    total_gt = int(merged["ground_truth_injected"].sum())

    glob_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    glob_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    glob_f1 = (2 * glob_prec * glob_rec / (glob_prec + glob_rec)) if (glob_prec + glob_rec) > 0 else 0.0

    inj_mean_err = merged[merged["ground_truth_injected"]]["raw_score_injected"].mean()
    norm_mean_err = merged[~merged["ground_truth_injected"]]["raw_score_injected"].mean()
    mean_delta_gt = merged[merged["ground_truth_injected"]]["delta_error"].mean()

    overall = {
        "total_series": len(summary_df),
        "total_windows": len(merged),
        "total_gt_injected_windows": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "true_negatives": total_tn,
        "global_precision": round(glob_prec, 4),
        "global_recall": round(glob_rec, 4),
        "global_f1": round(glob_f1, 4),
        "mean_error_injected_period": round(float(inj_mean_err), 2) if np.isfinite(inj_mean_err) else 0.0,
        "mean_error_normal_period": round(float(norm_mean_err), 2) if np.isfinite(norm_mean_err) else 0.0,
        "mean_delta_in_anomaly_period": round(float(mean_delta_gt), 2) if np.isfinite(mean_delta_gt) else 0.0,
        "error_amplification_ratio": round(float(inj_mean_err / (norm_mean_err + 1e-6)), 2) if np.isfinite(inj_mean_err) and np.isfinite(norm_mean_err) else 0.0,
    }

    return merged, summary_df, overall


def plot_comparisons(
    merged: pd.DataFrame,
    gt_map: dict[str, pd.DataFrame],
    output_dir: Path,
    max_plots: int = 6,
) -> None:
    """Generate comparative visualization plots for top injected meters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available, skipping plotting.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Pick top series with most injected windows
    top_series = (
        merged.groupby("series_id")["ground_truth_injected"]
        .sum()
        .sort_values(ascending=False)
        .head(max_plots)
        .index.tolist()
    )

    logging.info(f"Generating comparison plots for top {len(top_series)} meters in {plots_dir}...")

    for sid in top_series:
        sub = merged[merged["series_id"] == sid].sort_values("interval_start")
        if sub.empty:
            continue

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        times = sub["interval_start"]
        threshold = sub["threshold"].iloc[0]

        # 1. Power Comparison (if available in gt_map)
        if sid in gt_map:
            meter_gt = gt_map[sid]
            gt_sub = meter_gt.loc[(meter_gt.index >= times.min()) & (meter_gt.index <= times.max())]
            if not gt_sub.empty:
                ax1.plot(gt_sub.index, gt_sub["pRealKw_original"], label="Original pRealKw", color="#2ca02c", alpha=0.8, lw=1.2)
                ax1.plot(gt_sub.index, gt_sub["pRealKw"], label="Injected pRealKw", color="#d62728", lw=1.2, ls="--")
                ax1.set_ylabel("Power (kW)")
                ax1.legend(loc="upper right")
                ax1.set_title(f"Meter {sid}: Raw Power & Injected Modification")
                ax1.grid(True, linestyle=":", alpha=0.6)

        # Highlight injected regions
        injected_intervals = sub[sub["ground_truth_injected"]]
        for _, inj_row in injected_intervals.iterrows():
            ax1.axvspan(inj_row["interval_start"], inj_row["interval_end"], color="red", alpha=0.15)
            ax2.axvspan(inj_row["interval_start"], inj_row["interval_end"], color="red", alpha=0.15)
            ax3.axvspan(inj_row["interval_start"], inj_row["interval_end"], color="red", alpha=0.15)

        # 2. Reconstruction Errors
        ax2.plot(times, sub["raw_score_original"], label="Original Error", color="#1f77b4", lw=1.2)
        ax2.plot(times, sub["raw_score_injected"], label="Injected Error", color="#ff7f0e", lw=1.5)
        ax2.axhline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")
        ax2.set_ylabel("Reconstruction Error (MSE)")
        ax2.legend(loc="upper right")
        ax2.set_title("Reconstruction Error: Injected vs Original")
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. Delta Error
        ax3.plot(times, sub["delta_error"], label="ΔError (Injected - Original)", color="#9467bd", lw=1.5)
        ax3.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax3.set_ylabel("Δ Error")
        ax3.set_xlabel("Time (August 2026)")
        ax3.legend(loc="upper right")
        ax3.set_title("Reconstruction Error Increase (ΔError)")
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plot_path = plots_dir / f"comparison_{sid}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    logging.info(f"Comparison plots saved to {plots_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Injected Meters with LSTM AE")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "lstm_autoencoder.yaml"))
    parser.add_argument("--injected-dir", default=str(PROJECT_ROOT / "data" / "injected_meters"))
    parser.add_argument("--original-scores", default=str(PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "scores_15min.csv"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "best_model.pt"))
    parser.add_argument("--scaler", default=str(PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "scaler.joblib"))
    parser.add_argument("--metadata", default=str(PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "metadata.json"))
    parser.add_argument("--calibration", default=str(PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august" / "calibration_scores.npz"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "lstm_autoencoder" / "injected_evaluation"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])
    logging.info(f"Loaded threshold from metadata: {threshold:.4f}")

    scaler = joblib.load(args.scaler)
    logging.info(f"Loaded RobustScaler from {args.scaler}")

    # 1. Ground Truth Extraction
    gt_map = extract_ground_truth(Path(args.injected_dir))

    # 2. Build Dataset & Run Inference
    dataset, meta = build_injected_dataset(config, Path(args.injected_dir), scaler)
    injected_scores = run_inference(
        config,
        dataset,
        Path(args.checkpoint),
        threshold,
        calibration_path=Path(args.calibration),
        device_name=args.device,
    )

    # Save raw injected scores
    injected_scores.to_csv(output_dir / "injected_scores_15min.csv", index=False)
    logging.info(f"Injected scores saved to {output_dir / 'injected_scores_15min.csv'}")

    # 3. Evaluate and Compare with Original Scores
    merged, summary_df, overall = evaluate_and_compare(
        injected_scores,
        Path(args.original_scores),
        gt_map,
        threshold,
        output_dir,
    )

    merged.to_csv(output_dir / "injected_comparison_details.csv", index=False)
    summary_df.to_csv(output_dir / "injected_summary_metrics.csv", index=False)
    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info("=" * 60)
    logging.info("OVERALL EVALUATION RESULTS:")
    for k, v in overall.items():
        logging.info(f"  {k}: {v}")
    logging.info("=" * 60)

    # 4. Generate Visualizations
    if not args.no_plot:
        plot_comparisons(merged, gt_map, output_dir)

    print(f"\n[SUCCESS] 评估完成！产物保存在: {output_dir.resolve()}")
    print(f"  - 详细打分与差值表: {output_dir / 'injected_comparison_details.csv'}")
    print(f"  - 各电表指标汇总: {output_dir / 'injected_summary_metrics.csv'}")
    print(f"  - 全局统计与指标: {output_dir / 'overall_metrics.json'}")
    if not args.no_plot:
        print(f"  - 对比分析图表: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
