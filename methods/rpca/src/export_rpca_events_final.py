#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-duration-minutes", type=int, default=15)
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
        if not point_file.exists():
            continue

        points = pd.read_csv(point_file)
        points["timestamp"] = pd.to_datetime(points["timestamp"])
        points["is_anomaly"] = parse_bool(points["is_anomaly"])

        threshold = float(summary.iloc[0]["point_threshold"])

        a = points.loc[
            points["is_anomaly"],
            ["timestamp", "anomaly_score"]
        ].copy()

        if a.empty:
            continue

        a = a.sort_values("timestamp").reset_index(drop=True)

        gap_minutes = a["timestamp"].diff().dt.total_seconds().div(60)
        a["event_id"] = (
            gap_minutes.isna() | (gap_minutes > 5.01)
        ).cumsum()

        for _, g in a.groupby("event_id"):
            start = g["timestamp"].iloc[0]
            last_point = g["timestamp"].iloc[-1]
            end = last_point + pd.Timedelta(minutes=5)
            duration = int(round((end - start).total_seconds() / 60))

            if duration < args.min_duration_minutes:
                continue

            rows.append({
                "meter_id": meter,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_minutes": duration,
                "max_score": float(g["anomaly_score"].max()),
                "mean_score": float(g["anomaly_score"].mean()),
                "threshold": threshold,
                "n_train_windows": "",
                "n_test_windows": "",
            })

    result = pd.DataFrame(rows, columns=[
        "meter_id",
        "start",
        "end",
        "duration_minutes",
        "max_score",
        "mean_score",
        "threshold",
        "n_train_windows",
        "n_test_windows",
    ])

    if len(result):
        result["_start"] = pd.to_datetime(result["start"])
        result = (
            result.sort_values(["meter_id", "_start"])
                  .drop(columns=["_start"])
                  .reset_index(drop=True)
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Summary files scanned: {len(summary_files)}")
    print(f"Final sustained anomaly events: {len(result)}")
    print(f"Unique meter-channels with events: {result['meter_id'].nunique() if len(result) else 0}")
    print(f"Minimum event duration: {args.min_duration_minutes} minutes")
    print("Event continuity rule: consecutive 5-minute anomaly points")
    print(f"Output: {out}")

    if len(result):
        print("\nFirst 20 events:")
        print(result.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
