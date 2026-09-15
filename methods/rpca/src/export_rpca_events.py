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
    ap.add_argument("--max-gap-minutes", type=int, default=15,
                    help="Merge anomalous points into one event when the gap is <= this many minutes.")
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

        point_file = indir / f"{meter}_point_scores.csv"
        daily_file = indir / f"{meter}_daily_scores.csv"
        if not point_file.exists() or not daily_file.exists():
            continue

        points = pd.read_csv(point_file)
        daily = pd.read_csv(daily_file)

        points["timestamp"] = pd.to_datetime(points["timestamp"])
        points["is_anomaly"] = parse_bool(points["is_anomaly"])
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        daily["is_daily_anomaly"] = parse_bool(daily["is_daily_anomaly"])

        point_threshold = float(summary.iloc[0]["point_threshold"])
        daily_threshold = float(summary.iloc[0]["daily_threshold"])
        n_train_days = int(float(summary.iloc[0]["n_train_days"]))
        n_test_days = int(float(summary.iloc[0]["n_test_days"]))

        daily_map = daily.set_index("date")[["daily_anomaly_score", "is_daily_anomaly"]].to_dict("index")

        a = points.loc[points["is_anomaly"], ["timestamp", "anomaly_score"]].copy()
        if a.empty:
            continue
        a = a.sort_values("timestamp").reset_index(drop=True)

        # Start a new event when the time since the previous anomalous point exceeds max_gap_minutes.
        gap = a["timestamp"].diff().dt.total_seconds().div(60)
        a["event_id"] = (gap.isna() | (gap > args.max_gap_minutes)).cumsum()

        for _, g in a.groupby("event_id"):
            start = g["timestamp"].iloc[0]
            last_point = g["timestamp"].iloc[-1]
            # Each score represents a 5-minute sample, so extend end by 5 minutes.
            end = last_point + pd.Timedelta(minutes=5)
            duration = int((end - start).total_seconds() / 60)

            # Attach daily context from the calendar day where the event starts.
            d = start.date()
            day_info = daily_map.get(d, {})
            daily_score = day_info.get("daily_anomaly_score", np.nan)
            daily_flag = day_info.get("is_daily_anomaly", False)

            max_score = float(g["anomaly_score"].max())
            mean_score = float(g["anomaly_score"].mean())

            rows.append({
                "meter_id": meter,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_minutes": duration,
                "n_anomalous_points": int(len(g)),
                "max_score": max_score,
                "mean_score": mean_score,
                "threshold": point_threshold,
                "max_score_multiple": max_score / point_threshold if point_threshold else np.nan,
                "daily_score": float(daily_score) if pd.notna(daily_score) else np.nan,
                "daily_threshold": daily_threshold,
                "daily_anomaly": bool(daily_flag),
                "n_train_days": n_train_days,
                "n_test_days": n_test_days,
            })

    result = pd.DataFrame(rows)
    if len(result):
        result["start_dt"] = pd.to_datetime(result["start"])
        result = result.sort_values(["meter_id", "start_dt"]).drop(columns=["start_dt"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Summary files scanned: {len(summary_files)}")
    print(f"Event rows written: {len(result)}")
    print(f"Unique meter-channels with events: {result['meter_id'].nunique() if len(result) else 0}")
    print(f"Merge gap: <= {args.max_gap_minutes} minutes")
    print(f"Output: {out}")
    if len(result):
        print("\nFirst 20 events:")
        print(result.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
