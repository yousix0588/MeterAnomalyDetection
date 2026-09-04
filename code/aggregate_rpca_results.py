#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input-dir", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

indir = Path(args.input_dir)
files = sorted(f for f in indir.glob("*_summary.csv") if f.name != "rpca_all_summary.csv")
if not files:
    raise SystemExit(f"No *_summary.csv files found in {indir}")

frames = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df):
            frames.append(df)
    except Exception as e:
        print(f"Skipping {f.name}: {e}")

if not frames:
    raise SystemExit("No readable summary rows found.")

all_df = pd.concat(frames, ignore_index=True)

# Keep successful active channels for ranking.
ok = all_df[all_df["status"].eq("ok")].copy()
if len(ok):
    ok = ok.sort_values(
        ["n_anomalous_days", "max_daily_score"],
        ascending=[False, False],
    )

all_df.to_csv(args.out, index=False)

top_path = Path(args.out).with_name("rpca_top50.csv")
ok.head(50).to_csv(top_path, index=False)

print(f"Summary files read: {len(files)}")
print(f"Rows written: {len(all_df)}")
print(f"Successful channels: {len(ok)}")
print(f"All results: {args.out}")
print(f"Top 50: {top_path}")

if len(ok):
    print("\nTop 10:")
    cols = [
        c for c in [
            "meter_channel",
            "n_anomalous_days",
            "n_anomalous_points",
            "max_daily_score",
            "max_daily_score_date",
        ] if c in ok.columns
    ]
    print(ok[cols].head(10).to_string(index=False))
