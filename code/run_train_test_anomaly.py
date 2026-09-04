"""
Per-meter VAE anomaly detection with a FIXED train/test date split:
  - TRAIN: all data strictly before --train_end (e.g. everything before
    1 July 2026)
  - TEST:  only data in [--test_start, --test_end) (e.g. all of July 2026)
  - anything outside [start of data, --test_end) is simply never loaded/used
    (e.g. August 2026 is discarded)

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
    python run_train_test_anomaly.py --csv_dir ./meter_csvs --out_dir ./out_july \
        --train_end 2026-07-01 --test_start 2026-07-01 --test_end 2026-08-01

    # quick test on a few meters first:
    python run_train_test_anomaly.py --csv_dir ./meter_csvs --out_dir ./out_july_test \
        --train_end 2026-07-01 --test_start 2026-07-01 --test_end 2026-08-01 \
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
    TrainConfig, train_vae, score_windows, scores_to_timeline, find_anomaly_periods, plot_results,
)


def _to_tz_aware(ts: pd.Timestamp, tz):
    if tz is None:
        return ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def run_meter_train_test(csv_path: str, out_dir: str, train_end: str, test_start: str, test_end: str,
                          features: list[str] = None, signal_col: str = "pRealKw", window: int = 48,
                          stride: int = 1, epochs: int = 40, threshold_pct: float = 99.5,
                          use_calendar: bool = True, make_plot: bool = True,
                          min_train_windows: int = 100, min_test_windows: int = 10,
                          verbose: bool = True) -> dict:
    """Run the fixed train/test split pipeline for ONE meter. Returns a summary
    dict; the anomaly periods DataFrame (with meter_id attached) is under
    result['periods_df'] for the caller to aggregate across meters."""
    features = features or ["pRealKw", "pReactiveKw", "powerFactor", "vRMSMax", "iRMSMax"]
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
    if df.loc[test_start_ts:test_end_ts].empty:
        raise ValueError(f"{meter_id}: no data in test window [{test_start}, {test_end}) -- skipping "
                          f"(not present in test period)")

    windows, start_idx = make_windows_with_index(df, all_features, window, stride)
    window_times = df.index.values[start_idx]

    def _to_naive_utc(ts):
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts

    train_mask = window_times < np.datetime64(_to_naive_utc(train_end_ts))
    test_mask = ((window_times >= np.datetime64(_to_naive_utc(test_start_ts))) &
                 (window_times < np.datetime64(_to_naive_utc(test_end_ts))))

    if train_mask.sum() < min_train_windows:
        raise ValueError(f"{meter_id}: only {train_mask.sum()} training windows before {train_end} -- too little history")
    if test_mask.sum() < min_test_windows:
        raise ValueError(f"{meter_id}: only {test_mask.sum()} test windows in [{test_start}, {test_end}) -- too sparse")

    n_features = len(all_features)
    train_windows_raw = windows[train_mask]
    test_windows_raw = windows[test_mask]

    # scaler fit ONLY on training (pre-July) data -- no leakage from the test period
    scaler = StandardScaler().fit(train_windows_raw.reshape(-1, n_features))
    train_scaled = scaler.transform(train_windows_raw.reshape(-1, n_features)).reshape(train_windows_raw.shape).astype(np.float32)
    test_scaled = scaler.transform(test_windows_raw.reshape(-1, n_features)).reshape(test_windows_raw.shape).astype(np.float32)

    cfg = TrainConfig(epochs=epochs, train_frac=1.0)
    model = train_vae(train_scaled, cfg, device=device, verbose=verbose)

    train_scores = score_windows(model, train_scaled, device=device)
    threshold = np.percentile(train_scores, threshold_pct)

    test_scores = score_windows(model, test_scaled, device=device)
    test_df = df.loc[test_start_ts:test_end_ts]
    test_offset = df.index.get_loc(test_df.index[0])
    local_idx = start_idx[test_mask] - test_offset
    score_timeline = scores_to_timeline(test_df, test_scores, local_idx, window)

    periods = find_anomaly_periods(score_timeline, threshold, min_duration_steps=3, severity_multiplier=1.5)
    periods.insert(0, "meter_id", meter_id)
    periods["threshold"] = threshold
    periods["n_train_windows"] = int(train_mask.sum())
    periods["n_test_windows"] = int(test_mask.sum())

    periods_path = os.path.join(out_dir, f"{meter_id}_anomaly_periods.csv")
    periods.to_csv(periods_path, index=False)

    plot_path = None
    if make_plot:
        plot_path = os.path.join(out_dir, f"{meter_id}_anomaly_diagram.png")
        plot_results(test_df, score_timeline, threshold, periods.drop(columns=["meter_id", "threshold",
                     "n_train_windows", "n_test_windows"], errors="ignore"), signal_col, plot_path,
                     f"{meter_id} — test: {test_start} to {test_end}")

    return {"meter_id": meter_id, "n_periods": len(periods), "threshold": threshold,
            "periods_path": periods_path, "plot_path": plot_path, "periods_df": periods}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True, help="Directory of combined per-meter CSVs")
    parser.add_argument("--out_dir", default="./out_train_test")
    parser.add_argument("--train_end", required=True, help="Train on all data BEFORE this date, e.g. 2026-07-01")
    parser.add_argument("--test_start", required=True, help="Test window start (inclusive), e.g. 2026-07-01")
    parser.add_argument("--test_end", required=True, help="Test window end (exclusive), e.g. 2026-08-01")
    parser.add_argument("--features", nargs="+", default=[
        "pRealKw", "pReactiveKw", "powerFactor", "vRMSMax", "iRMSMax"])
    parser.add_argument("--signal_col", default="pRealKw")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threshold_pct", type=float, default=99.5)
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

    summary_rows, all_periods = [], []
    for csv_path in tqdm(csv_files, desc="meters"):
        meter_id = os.path.splitext(os.path.basename(csv_path))[0]
        t0 = time.time()
        try:
            result = run_meter_train_test(
                csv_path=csv_path, out_dir=args.out_dir, train_end=args.train_end,
                test_start=args.test_start, test_end=args.test_end, features=args.features,
                signal_col=args.signal_col, window=args.window, stride=args.stride,
                epochs=args.epochs, threshold_pct=args.threshold_pct, use_calendar=args.use_calendar,
                make_plot=args.make_plots, verbose=False,
            )
            summary_rows.append({"meter_id": meter_id, "status": "ok", "n_anomaly_periods": result["n_periods"],
                                  "threshold": result["threshold"], "seconds": round(time.time() - t0, 1), "error": ""})
            if len(result["periods_df"]):
                all_periods.append(result["periods_df"])
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
