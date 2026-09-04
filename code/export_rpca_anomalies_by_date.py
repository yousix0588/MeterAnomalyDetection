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

        daily["is_daily_anomaly"] = parse_bool(daily["is_daily_anomaly"])
        points["is_anomaly"] = parse_bool(points["is_anomaly"])

        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        points["timestamp"] = pd.to_datetime(points["timestamp"])
        points["date"] = points["timestamp"].dt.date

        daily_threshold = float(summary.iloc[0]["daily_threshold"])
        point_threshold = float(summary.iloc[0]["point_threshold"])

        anomalous_days = daily[daily["is_daily_anomaly"]].copy()

        for _, dr in anomalous_days.iterrows():
            date = dr["date"]
            day_points = points[points["date"] == date]
            anom_points = day_points[day_points["is_anomaly"]]

            if len(anom_points):
                first_time = anom_points["timestamp"].min()
                last_time = anom_points["timestamp"].max()
                max_point = float(anom_points["anomaly_score"].max())
                mean_point = float(anom_points["anomaly_score"].mean())
                n_points = int(len(anom_points))
            else:
                first_time = pd.NaT
                last_time = pd.NaT
                max_point = np.nan
                mean_point = np.nan
                n_points = 0

            daily_score = float(dr["daily_anomaly_score"])
            rows.append({
                "meter_id": meter,
                "date": str(date),
                "daily_score": daily_score,
                "daily_threshold": daily_threshold,
                "daily_score_multiple": daily_score / daily_threshold if daily_threshold else np.nan,
                "n_anomalous_points": n_points,
                "first_anomaly_time": first_time.isoformat() if pd.notna(first_time) else "",
                "last_anomaly_time": last_time.isoformat() if pd.notna(last_time) else "",
                "max_point_score": max_point,
                "mean_point_score": mean_point,
                "point_threshold": point_threshold,
            })

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["meter_id", "date"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Summary files scanned: {len(summary_files)}")
    print(f"Anomalous meter-days written: {len(result)}")
    print(f"Unique meter-channels with anomalous days: {result['meter_id'].nunique() if len(result) else 0}")
    print(f"Output: {out}")
    if len(result):
        print("\nFirst 20 rows:")
        print(result.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
