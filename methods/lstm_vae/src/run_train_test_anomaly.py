"""
Per-meter VAE anomaly detection with a FIXED train/test date split:
  - TRAIN: all data strictly before --train_end (e.g. everything before
    1 August 2026, so the last training day is 31 July)
  - TEST:  only data in [--test_start, --test_end) (e.g. all of August 2026)
  - anything outside [start of data, --test_end) is simply never loaded/used
    (e.g. September 2026 is discarded)

One model per meter (no cluster sharing). Only meters that actually have
data in the test window are processed -- if you built your CSVs with
build_meter_csvs.py's --require_start/--require_end pointed at the test
period, every CSV in --csv_dir already satisfies this; this script also
re-checks per meter as a safety net in case you didn't.

All detected anomaly periods, across every meter, are written to ONE
consolidated CSV (in addition to a per-meter CSV each), so you can open a
single file and see every anomaly found in the test period with the
meter_id attached.

Usage
-----
    python run_train_test_anomaly.py --csv_dir ./meter_csvs --out_dir ./out_august \
        --train_end 2026-08-01 --test_start 2026-08-01 --test_end 2026-09-01

    # quick test on a few meters first:
    python run_train_test_anomaly.py --csv_dir ./meter_csvs --out_dir ./out_august_test \
        --train_end 2026-08-01 --test_start 2026-08-01 --test_end 2026-09-01 \
        --epochs 15 --limit 5
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from vae_anomaly_detection import (
    load_meter_series, make_windows_with_index, add_calendar_features, CALENDAR_FEATURES,
    TrainConfig, train_vae, score_window_point_errors, frozen_ecdf,
    find_anomaly_periods, plot_results,
)


def _to_tz_aware(ts: pd.Timestamp, tz):
    if tz is None:
        return ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def days_since_last_real_data(df: pd.DataFrame, boundary_ts) -> float:
    """
    How many days between the most recent NON-imputed (genuinely observed,
    not interpolated/missing) data point strictly before boundary_ts, and
    boundary_ts itself.

    Uses df['is_imputed'] (set by load_meter_series -- True for any
    resampled timestamp that had no real reading, regardless of whether a
    short gap was later interpolated over it), so this reflects actual
    data availability, not just whether a window happened to be droppable.

    Returns float('inf') if there's no real data before boundary_ts at all.
    """
    before = df.loc[df.index < boundary_ts]
    real_before = before[~before["is_imputed"]]
    if real_before.empty:
        return float("inf")
    last_real_ts = real_before.index.max()
    return (pd.Timestamp(boundary_ts) - last_real_ts).total_seconds() / 86400


def run_meter_train_test(csv_path: str, out_dir: str, train_end: str, test_start: str, test_end: str,
                          features: list[str] = None, signal_col: str = "pRealKw", window: int = 288,
                          stride: int = 3, epochs: int = 40, threshold_pct: float = 99.0,
                          use_calendar: bool = True, make_plot: bool = True,
                          min_train_windows: int = 100, min_test_windows: int = 10,
                          max_baseline_gap_days: float = 30.0,
                          verbose: bool = True) -> dict:
    """Run the fixed train/test split pipeline for ONE meter. Returns a summary
    dict; the anomaly periods DataFrame (with meter_id attached) is under
    result['periods_df'] for the caller to aggregate across meters.

    max_baseline_gap_days: if the most recent REAL (non-imputed) training
    data point is more than this many days before test_start, the meter is
    skipped rather than trained -- a large gap immediately before the test
    period means the model would be judging brand-new data against a stale
    baseline it has no recent information to support (e.g. a meter that
    went offline for months and only just came back online right as the
    test period starts). This does NOT flag gaps elsewhere in the training
    history -- those are already handled correctly by windows containing
    NaN being dropped from training entirely (see make_windows_with_index)
    -- only staleness of the most recent training data relative to the
    test boundary, which windows-with-NaN-dropping does not catch."""
    features = features or ["pRealKw", "pReactiveKw", "powerFactor", "vRMSMax", "iRMSMax"]
    if stride != 3:
        raise ValueError("The canonical output contract requires stride=3 at 5-minute sampling")
    os.makedirs(out_dir, exist_ok=True)
    meter_id = os.path.splitext(os.path.basename(csv_path))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_meter_series(csv_path, features)
    if use_calendar:
        df = add_calendar_features(df)
    all_features = features + (CALENDAR_FEATURES if use_calendar else [])

    tz = df.index.tz
    train_end_ts = _to_tz_aware(pd.Timestamp(train_end), tz)
    test_start_ts = _to_tz_aware(pd.Timestamp(test_start), tz)
    test_end_ts = _to_tz_aware(pd.Timestamp(test_end), tz)

    # safety net: skip meters that don't actually have test-period data,
    # even if they made it into csv_dir (e.g. if built without --require_*)
    test_presence = df.loc[(df.index >= test_start_ts) & (df.index < test_end_ts)]
    if test_presence.empty:
        raise ValueError(f"{meter_id}: no data in test window [{test_start}, {test_end}) -- skipping "
                          f"(not present in test period)")

    # stale-baseline check: skip if the model's most recent real training
    # data is too far in the past relative to the test period it's about
    # to judge -- prevents cases like a meter offline for 85 days with data
    # resuming right as the test window starts (silently produces a model
    # with zero recent information, flagging the whole test period)
    baseline_gap_days = days_since_last_real_data(df, test_start_ts)
    if baseline_gap_days > max_baseline_gap_days:
        raise ValueError(
            f"{meter_id}: most recent real (non-interpolated) data before the test period is "
            f"{baseline_gap_days:.0f} days old -- exceeds --max_baseline_gap_days={max_baseline_gap_days:.0f}. "
            f"Skipping: the model would have no recent baseline to judge the test period against."
        )

    windows, start_idx = make_windows_with_index(df, all_features, window, stride)
    sample_step = pd.Timedelta(minutes=5)
    block_duration = sample_step * stride
    window_ends = df.index[start_idx + window - 1] + sample_step
    interval_starts = window_ends - block_duration

    # A test window may use pre-August history as causal context, but its scored
    # final block must be wholly inside August. No test value enters calibration.
    train_mask = window_ends <= train_end_ts
    test_mask = ((interval_starts >= test_start_ts) & (window_ends <= test_end_ts))

    if train_mask.sum() < min_train_windows:
        raise ValueError(f"{meter_id}: only {train_mask.sum()} training windows before {train_end} -- too little history")
    if test_mask.sum() < min_test_windows:
        raise ValueError(f"{meter_id}: only {test_mask.sum()} test windows in [{test_start}, {test_end}) -- too sparse")

    n_features = len(all_features)
    train_windows_raw = windows[train_mask]
    test_windows_raw = windows[test_mask]

    # scaler fit ONLY on training (pre-August) data -- no leakage from the test period
    scaler = StandardScaler().fit(train_windows_raw.reshape(-1, n_features))
    train_scaled = scaler.transform(train_windows_raw.reshape(-1, n_features)).reshape(train_windows_raw.shape).astype(np.float32)
    test_scaled = scaler.transform(test_windows_raw.reshape(-1, n_features)).reshape(test_windows_raw.shape).astype(np.float32)

    cfg = TrainConfig(epochs=epochs, train_frac=1.0)
    model = train_vae(train_scaled, cfg, device=device, verbose=verbose)

    train_point_errors = score_window_point_errors(model, train_scaled, device=device)
    test_point_errors = score_window_point_errors(model, test_scaled, device=device)
    train_blocks = train_point_errors[:, -stride:]
    test_blocks = test_point_errors[:, -stride:]
    train_max = train_blocks.max(axis=1)
    train_mean = train_blocks.mean(axis=1)
    test_max = test_blocks.max(axis=1)
    test_mean = test_blocks.mean(axis=1)
    test_std = test_blocks.std(axis=1)
    threshold = np.percentile(train_max, threshold_pct)
    max_percentile = frozen_ecdf(test_max, train_max)
    mean_percentile = frozen_ecdf(test_mean, train_mean)
    calibration_version = f"train_pre_{train_end}_v1"

    calibration_path = os.path.join(out_dir, f"{meter_id}_calibration_scores.npz")
    np.savez_compressed(calibration_path, max_scores=train_max, mean_scores=train_mean)
    calibration_meta = pd.DataFrame([{
        "model": "lstm_vae", "meter_id": meter_id, "fit_end_exclusive": train_end,
        "source": "train_only", "score_definition": "last_15min_point_reconstruction_error_max",
        "sample_count": len(train_max), "calibration_version": calibration_version,
    }])
    calibration_meta.to_json(
        os.path.join(out_dir, f"{meter_id}_calibration.json"), orient="records", indent=2
    )

    score_rows = pd.DataFrame({
        "model": "lstm_vae",
        "series_id": meter_id,
        "interval_start": interval_starts[test_mask],
        "interval_end": window_ends[test_mask],
        "raw_score": test_max,
        "max_score": test_max,
        "mean_score": test_mean,
        "score_std": test_std,
        "max_percentile": max_percentile,
        "mean_percentile": mean_percentile,
        "threshold": threshold,
        "is_anomaly": test_max > threshold,
        "valid_point_count": stride,
        "expected_point_count": stride,
        "coverage_ratio": 1.0,
        "source_resolution_minutes": 5,
        "aggregation_method": "last_15min_of_causal_24h_window",
        "data_status": "native_5m",
        "calibration_version": calibration_version,
        "available": True,
    })
    scores_path = os.path.join(out_dir, f"{meter_id}_scores_15min.csv")
    score_rows.to_csv(scores_path, index=False)

    test_df = df.loc[(df.index >= test_start_ts) & (df.index < test_end_ts)]
    score_timeline = pd.Series(
        test_max, index=pd.DatetimeIndex(interval_starts[test_mask]), name="anomaly_score"
    )

    periods = find_anomaly_periods(
        score_timeline, threshold, min_duration_steps=1, severity_multiplier=1.0
    )
    periods.insert(0, "meter_id", meter_id)
    periods["threshold"] = threshold
    periods["n_train_windows"] = int(train_mask.sum())
    periods["n_test_windows"] = int(test_mask.sum())
    periods["baseline_gap_days"] = round(baseline_gap_days, 1)

    periods_path = os.path.join(out_dir, f"{meter_id}_anomaly_periods.csv")
    periods.to_csv(periods_path, index=False)

    plot_path = None
    if make_plot:
        plot_path = os.path.join(out_dir, f"{meter_id}_anomaly_diagram.png")
        plot_results(test_df, score_timeline, threshold, periods.drop(columns=["meter_id", "threshold",
                     "n_train_windows", "n_test_windows"], errors="ignore"), signal_col, plot_path,
                     f"{meter_id} — test: {test_start} to {test_end}")

    return {"meter_id": meter_id, "n_periods": len(periods), "threshold": threshold,
            "periods_path": periods_path, "scores_path": scores_path,
            "plot_path": plot_path, "periods_df": periods, "scores_df": score_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True, help="Directory of combined per-meter CSVs")
    parser.add_argument("--out_dir", default="./out_train_test")
    parser.add_argument("--train_end", required=True, help="Train on all data BEFORE this date, e.g. 2026-08-01")
    parser.add_argument("--test_start", required=True, help="Test window start (inclusive), e.g. 2026-08-01")
    parser.add_argument("--test_end", required=True, help="Test window end (exclusive), e.g. 2026-09-01")
    parser.add_argument("--features", nargs="+", default=[
        "pRealKw", "pReactiveKw", "powerFactor", "vRMSMax", "iRMSMax"])
    parser.add_argument("--signal_col", default="pRealKw")
    parser.add_argument("--window", type=int, default=288, help="24 hours at 5-minute sampling")
    parser.add_argument("--stride", type=int, default=3, help="15 minutes at 5-minute sampling")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threshold_pct", type=float, default=99.0)
    parser.add_argument("--max_baseline_gap_days", type=float, default=30.0,
                         help="Skip a meter if its most recent real (non-interpolated) training data is "
                              "more than this many days before --test_start (default 30). Catches meters "
                              "whose data resumes shortly before the test period after a long outage, "
                              "where the model would have no recent baseline to judge against.")
    parser.add_argument("--no_calendar_features", dest="use_calendar", action="store_false", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N meters (testing)")
    parser.add_argument("--no_plots", dest="make_plots", action="store_false", default=True)
    parser.add_argument("--array_index", type=int, default=0,
                         help="This task's index within a Slurm job array (0-based, i.e. $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--array_count", type=int, default=1,
                         help="Total number of array tasks -- meters are split round-robin across tasks "
                              "so each Slurm array task trains a disjoint subset of meters in parallel")
    args = parser.parse_args()

    csv_files = sorted(f for f in glob.glob(os.path.join(args.csv_dir, "*.csv")) if not f.endswith("_manifest.csv"))
    if args.limit:
        csv_files = csv_files[:args.limit]
    if args.array_count > 1:
        csv_files = csv_files[args.array_index::args.array_count]  # disjoint, round-robin split across array tasks

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[array {args.array_index}/{args.array_count}] Running train/test split pipeline on {len(csv_files)} meters "
          f"(train < {args.train_end}, test [{args.test_start}, {args.test_end}))...")

    summary_rows, all_periods, all_scores = [], [], []
    for csv_path in tqdm(csv_files, desc="meters"):
        meter_id = os.path.splitext(os.path.basename(csv_path))[0]
        t0 = time.time()
        try:
            result = run_meter_train_test(
                csv_path=csv_path, out_dir=args.out_dir, train_end=args.train_end,
                test_start=args.test_start, test_end=args.test_end, features=args.features,
                signal_col=args.signal_col, window=args.window, stride=args.stride,
                epochs=args.epochs, threshold_pct=args.threshold_pct, use_calendar=args.use_calendar,
                max_baseline_gap_days=args.max_baseline_gap_days,
                make_plot=args.make_plots, verbose=False,
            )
            summary_rows.append({"meter_id": meter_id, "status": "ok", "n_anomaly_periods": result["n_periods"],
                                  "threshold": result["threshold"], "seconds": round(time.time() - t0, 1), "error": ""})
            if len(result["periods_df"]):
                all_periods.append(result["periods_df"])
            all_scores.append(result["scores_df"])
        except Exception as e:
            summary_rows.append({"meter_id": meter_id, "status": "skipped/failed", "n_anomaly_periods": None,
                                  "threshold": None, "seconds": round(time.time() - t0, 1), "error": str(e)})

    # --- consolidated export: every anomaly, every meter, one CSV ---
    if all_periods:
        combined = pd.concat(all_periods, ignore_index=True)
        combined = combined.sort_values(["meter_id", "start"]).reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=["meter_id", "start", "end", "duration_minutes", "max_score",
                                          "mean_score", "threshold", "n_train_windows", "n_test_windows"])
    combined_path = os.path.join(args.out_dir, f"ALL_METERS_anomalies_part{args.array_index}.csv"
                                  if args.array_count > 1 else "ALL_METERS_anomalies.csv")
    combined.to_csv(combined_path, index=False)

    score_columns = [
        "model", "series_id", "interval_start", "interval_end", "raw_score",
        "max_score", "mean_score", "score_std", "max_percentile", "mean_percentile",
        "threshold", "is_anomaly", "valid_point_count", "expected_point_count",
        "coverage_ratio", "source_resolution_minutes", "aggregation_method", "data_status",
        "calibration_version", "available",
    ]
    combined_scores = (
        pd.concat(all_scores, ignore_index=True)
        if all_scores else pd.DataFrame(columns=score_columns)
    )
    scores_name = (f"ALL_METERS_scores_15min_part{args.array_index}.csv"
                   if args.array_count > 1 else "ALL_METERS_scores_15min.csv")
    combined_scores.to_csv(os.path.join(args.out_dir, scores_name), index=False)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, f"_run_summary_part{args.array_index}.csv"
                                 if args.array_count > 1 else "_run_summary.csv")
    summary.to_csv(summary_path, index=False)

    n_ok = (summary.status == "ok").sum()
    n_skipped = (summary.status == "skipped/failed").sum()
    print(f"\nDone. {n_ok} meters processed, {n_skipped} skipped/failed (see {summary_path}).")
    print(f"Total anomaly periods found across all meters: {len(combined)}")
    print(f"Consolidated anomaly export -> {combined_path}")
    if len(combined):
        print("\nMeters with the most anomaly periods:")
        print(combined["meter_id"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
