#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJECTED_DIR = PROJECT_ROOT / "data" / "injected_meters"

rows = []
for p in sorted(INJECTED_DIR.glob("*.csv")):
    if p.name.startswith("_"):
        continue
    series_id = p.stem
    df = pd.read_csv(p, usecols=["timestamp"])
    parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    bad_rows = int(parsed.isna().sum())
    timestamps = parsed.dropna().sort_values()
    dates = timestamps.dt.date.unique()
    days_in_span = (timestamps.max().date() - timestamps.min().date()).days + 1 if len(timestamps) else 0
    rows.append({
        "meter_channel": series_id,
        "n_rows": len(timestamps),
        "n_days_present": len(dates),
        "n_days_missing": days_in_span - len(dates),
        "n_bad_timestamp_rows_dropped": bad_rows,
        "start": str(timestamps.min()) if len(timestamps) else "",
        "end": str(timestamps.max()) if len(timestamps) else "",
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(INJECTED_DIR / "_manifest.csv", index=False)
print(f"Generated manifest with {len(manifest)} rows at {INJECTED_DIR / '_manifest.csv'}")
