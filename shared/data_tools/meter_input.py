"""Shared strict five-minute input preparation, extracted from the local MP runner.

No training, scoring, interpolation or signed-power recomputation takes place.
The archive reader remains the existing build_meter_csvs implementation.
"""
from datetime import timedelta

import numpy as np
import pandas as pd


COLUMNS = ["timestamp", "powerFactor", "pRealKw", "pRealPositiveKw", "pRealNegativeKw",
           "pReactiveKw", "pReactivePositiveKw", "pReactiveNegativeKw", "vRMSMin",
           "vRMSMax", "iRMSMin", "iRMSMax"]


def index_target(reader, start, end):
    from pathlib import Path
    from build_meter_csvs import find_date_in_path
    result = {}
    for name in reader.namelist():
        day = find_date_in_path(name)
        if day is not None and start <= day.date() < end and name.lower().endswith(".csv"):
            key = Path(name).stem
            result.setdefault(key, []).append((day.date(), name))
    for entries in result.values():
        entries.sort()
        if len({day for day, _ in entries}) != len(entries):
            raise ValueError("Duplicate daily CSV for a channel/date in target source")
    return result


def normalize(frame, timezone):
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Expected the unified 12-column raw schema, got {list(frame.columns)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise").dt.tz_convert(timezone)
    for column in COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if np.isinf(result[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite input values in {column}")
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate input timestamp; refusing to silently average or discard")
    timestamps = result["timestamp"]
    if (timestamps.isna().any() or not timestamps.dt.minute.mod(5).eq(0).all()
            or not timestamps.dt.second.eq(0).all() or not timestamps.dt.microsecond.eq(0).all()
            or not timestamps.dt.nanosecond.eq(0).all()):
        raise ValueError("Input timestamps must be valid and aligned to the 5-minute grid")
    return result.sort_values("timestamp").reset_index(drop=True)


def prepare_inputs(reader, indexed, keys, history, destination, config):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    first = pd.Timestamp(config.baseline_start, tz=config.timezone)
    last = pd.Timestamp(config.baseline_end + timedelta(days=1), tz=config.timezone)
    for position, key in enumerate(keys, 1):
        frames = []
        baseline_path = history / f"{key}.csv"
        if baseline_path.exists():
            for chunk in pd.read_csv(baseline_path, chunksize=50000):
                local_dates = chunk["timestamp"].astype(str).str.slice(0, 10)
                selected = chunk.loc[local_dates.between(str(config.baseline_start), str(config.baseline_end))]
                if not selected.empty:
                    selected = normalize(selected, config.timezone)
                    frames.append(selected.loc[(selected.timestamp >= first) & (selected.timestamp < last)])
        for day, name in indexed.get(key, []):
            with reader.open(name) as stream:
                frame = normalize(pd.read_csv(stream), config.timezone)
            if not frame.timestamp.dt.date.eq(day).all():
                raise ValueError(f"Timestamp/date-folder mismatch: {name}")
            frames.append(frame)
        combined = normalize(pd.concat(frames, ignore_index=True), config.timezone) if frames else pd.DataFrame(columns=COLUMNS)
        combined.to_csv(destination / f"{key}.csv", index=False)
        baseline_mask = (combined.timestamp >= first) & (combined.timestamp < last) if len(combined) else pd.Series(dtype=bool)
        baseline_days = combined.loc[baseline_mask, "timestamp"].dt.date.nunique() if len(combined) else 0
        target_days = combined.loc[~baseline_mask, "timestamp"].dt.date.nunique() if len(combined) else 0
        rows.append(dict(meter_channel=key, n_rows=len(combined), baseline_days_present=baseline_days,
                         target_days_present=target_days,
                         target_days_missing=(config.detection_end-config.detection_start).days+1-target_days))
        if position % 20 == 0 or position == len(keys):
            print(f"Prepared {position}/{len(keys)} channels", flush=True)
    pd.DataFrame(rows).to_csv(destination / "_manifest.csv", index=False)
