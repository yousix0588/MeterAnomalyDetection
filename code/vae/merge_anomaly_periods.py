"""
Append a new period's anomaly detections (e.g. August) onto an existing
anomaly CSV (e.g. July's ALL_METERS_anomalies.csv), producing one combined
file -- rather than re-scoring the whole thing from scratch.

Usage
-----
    python merge_anomaly_periods.py \
        --existing ALL_METERS_anomalies.csv \
        --new out_august/ALL_METERS_anomalies.csv \
        --out ALL_METERS_anomalies.csv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True, help="The current anomaly csv (e.g. July results)")
    parser.add_argument("--new", required=True, help="The new period's anomaly csv (e.g. August results)")
    parser.add_argument("--out", required=True, help="Output path (can be the same as --existing to overwrite)")
    args = parser.parse_args()

    existing = pd.read_csv(args.existing)
    new = pd.read_csv(args.new)

    if list(existing.columns) != list(new.columns):
        raise SystemExit(f"Column mismatch -- existing has {list(existing.columns)}, "
                          f"new has {list(new.columns)}. Re-run both through the same pipeline version.")

    combined = pd.concat([existing, new], ignore_index=True)
    # de-dupe defensively in case of any overlapping re-run (same meter+start+end)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["meter_id", "start", "end"])
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} duplicate row(s) (same meter_id+start+end already present)")

    combined = combined.sort_values(["meter_id", "start"]).reset_index(drop=True)
    combined.to_csv(args.out, index=False)
    print(f"Merged: {len(existing)} existing + {len(new)} new -> {len(combined)} total rows -> {args.out}")


if __name__ == "__main__":
    main()
