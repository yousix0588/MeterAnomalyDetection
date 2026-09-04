#!/usr/bin/env python3
"""
Create a clean RPCA channel list from meter_csvs/_manifest.csv.

For the first large-data run, keep channels with n_days_missing == 0.
This avoids mixing the anomaly model with unresolved missing-day handling.
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.manifest)

required = {
    "meter_channel",
    "n_days_present",
    "n_days_missing",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

clean = df[
    (df["n_days_missing"] == 0)
    & (df["n_days_present"] >= 250)
].copy()

clean[["meter_channel"]].to_csv(args.out, index=False)

print(f"Input manifest rows: {len(df)}")
print(f"Clean channels retained: {len(clean)}")
print(f"Written to: {args.out}")
