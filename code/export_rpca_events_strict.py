#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap-minutes", type=int, default=15)
    args = ap.parse_args()

    indir = Path(args.input_dir)
    out = Path(args.out)
    rows = []

    summary_files = sorted(
        f for f in indir.glob("*_summary.csv")
        if f.name != "rpca_all_summary.csv"
    )

    for sf in summary_files:
        meter = sf.name[:-len("_summary.csv")]
        summary = pd.read_csv(sf)
        if summary.empty or "status" not in summary.columns:
            continue
        if str(summary.iloc[0]["status"]) != "ok":
            continue

        daily_file = indir / f"{meter}_daily_scores.csv"
        point_file = indir / f"{meter}_point_scores.csv"
        if not daily_file.exists() or not point_file.exists():
            continue

        daily = pd.read_csv(daily_file)
        points = pd.read_csv(point_file)

        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        daily["is_daily_anomaly"] = parse_bool(daily["is_daily_anomaly"])
        points["timestamp"] = pd.to_datetime(points["timestamp"])
        points["date"] = points["timestamp"].dt.date
        points["is_anomaly"] = parse_bool(points["is_anomaly"])

        point_threshold = float(summary.iloc[0]["point_threshold"])
        daily_threshold = float(summary.iloc[0]["daily_threshold"])
        n_train_days = int(float(summary.iloc[0]["n_train_days"]))
        n_test_days = int(float(summary.iloc[0]["n_test_days"]))

        anomalous_days = daily[daily["is_daily_anomaly"]].copy()

        for _, dr in anomalous_days.iterrows():
            date = dr["date"]
            daily_score = float(dr["daily_anomaly_score"])

            a = points[
                (points["date"] == date) & (points["is_anomaly"])
            ][["timestamp", "anomaly_score"]].copy()

            # Keep the anomalous day even if it has no point-level threshold
            # exceedance, because the day-level detector itself flagged it.
            if a.empty:
                rows.append({
                    "meter_id": meter,
                    "date": str(date),
                    "start": "",
                    "end": "",
                    "duration_minutes": 0,
                    "n_anomalous_points": 0,
                    "max_score": np.nan,
                    "mean_score": np.nan,
                    "threshold": point_threshold,
                    "max_score_multiple": np.nan,
                    "daily_score": daily_score,
                    "daily_threshold": daily_threshold,
                    "daily_score_multiple": daily_score / daily_threshold if daily_threshold else np.nan,
                    "n_train_days": n_train_days,
                    "n_test_days": n_test_days,
                })
                continue

            a = a.sort_values("timestamp").reset_index(drop=True)
            gap = a["timestamp"].diff().dt.total_seconds().div(60)
            a["event_id"] = (gap.isna() | (gap > args.max_gap_minutes)).cumsum()

            for _, g in a.groupby("event_id"):
                start = g["timestamp"].iloc[0]
                last_point = g["timestamp"].iloc[-1]
                end = last_point + pd.Timedelta(minutes=5)
                duration = int((end - start).total_seconds() / 60)
                max_score = float(g["anomaly_score"].max())
                mean_score = float(g["anomaly_score"].mean())

                rows.append({
                    "meter_id": meter,
                    "date": str(date),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "duration_minutes": duration,
                    "n_anomalous_points": int(len(g)),
                    "max_score": max_score,
                    "mean_score": mean_score,
                    "threshold": point_threshold,
                    "max_score_multiple": max_score / point_threshold if point_threshold else np.nan,
                    "daily_score": daily_score,
                    "daily_threshold": daily_threshold,
                    "daily_score_multiple": daily_score / daily_threshold if daily_threshold else np.nan,
                    "n_train_days": n_train_days,
                    "n_test_days": n_test_days,
                })

    result = pd.DataFrame(rows)
    if len(result):
        result["start_sort"] = pd.to_datetime(result["start"], errors="coerce")
        result = result.sort_values(
            ["meter_id", "date", "start_sort"],
            na_position="last"
        ).drop(columns=["start_sort"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Summary files scanned: {len(summary_files)}")
    print(f"Strict event rows written: {len(result)}")
    print(f"Unique meter-channels: {result['meter_id'].nunique() if len(result) else 0}")
    print(f"Unique anomalous meter-days: {result[['meter_id','date']].drop_duplicates().shape[0] if len(result) else 0}")
    print(f"Output: {out}")
    if len(result):
        print("\nFirst 20 rows:")
        print(result.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
