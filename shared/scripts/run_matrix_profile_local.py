"""Run the existing fixed-baseline detector locally, with resumable canonical output.

All relative CLI paths are resolved against group14reshape, not the shell cwd.
--end is exclusive; the detector's internal detection_end is inclusive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Avoid each Windows worker creating a full pool of BLAS threads.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
MP_SRC = ROOT / "methods/matrix_profile/src"
sys.path.insert(0, str(MP_SRC))
sys.path.insert(0, str(ROOT / "shared/data_tools"))
sys.path.insert(0, str(ROOT / "shared/scripts"))

import numpy as np
import pandas as pd
from build_meter_csvs import ArchiveReader, find_date_in_path
import build_ensemble_features as ensemble
from anomalies.multiscale.aggregate import aggregate_shards, SHARD_FILES
from anomalies.multiscale.cloud_batch import (
    discover_merged_channels, _detect_merged_worker, populate_missing_day_statuses,
)
from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.output import INTERVAL_SCORE_COLUMNS, _upsample_interval_frame, write_shard_outputs

COLUMNS = ["timestamp", "powerFactor", "pRealKw", "pRealPositiveKw", "pRealNegativeKw",
           "pReactiveKw", "pReactivePositiveKw", "pReactiveNegativeKw", "vRMSMin",
           "vRMSMax", "iRMSMin", "iRMSMax"]


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def index_target(reader, start, end):
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
            or not timestamps.dt.second.eq(0).all()):
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
                # Old merged timestamps are local ISO timestamps; normalize the
                # selected rows again before making the authoritative date cut.
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


def complete_grid(native, key, day_status, config):
    """Keep skipped/missing intervals explicit, with NaN scores, not zero scores."""
    start = pd.Timestamp(config.detection_start, tz=config.timezone)
    end = pd.Timestamp(config.detection_end + timedelta(days=1), tz=config.timezone)
    grid = pd.date_range(start, end, freq="30min", inclusive="left", name="interval_start")
    native = native.copy()
    native["interval_start"] = pd.to_datetime(native["interval_start"], utc=True).dt.tz_convert(config.timezone)
    if native.interval_start.duplicated().any():
        raise ValueError(f"Duplicate native scores: {key}")
    result = native.set_index("interval_start").reindex(grid).reset_index()
    missing = result["model"].isna()
    result["interval_end"] = result.interval_start + pd.Timedelta(minutes=30)
    constants = dict(model="matrix_profile", series_id=key, threshold=config.candidate_percentile/100,
                     expected_point_count=3, source_resolution_minutes=30,
                     aggregation_method="native_30m_multiscale_percentiles", data_status="native_30m",
                     calibration_version=f"fixed_{config.baseline_start}_{config.baseline_end}_v2")
    for column, value in constants.items():
        result.loc[missing, column] = value
    for column in ("available", "is_anomaly", "daily_outlier"):
        result[column] = result[column].fillna(False).astype(bool)
    for column in ("coverage_ratio", "valid_point_count"):
        result.loc[missing, column] = 0
    statuses = day_status.set_index("date")["status"].to_dict()
    result["availability_reason"] = result.interval_start.dt.strftime("%Y-%m-%d").map(statuses)
    result.loc[result.available, "availability_reason"] = "scored"
    return result


def export_aligned(shards, keys, output, config):
    aligned = output / "aligned"
    aligned.mkdir(exist_ok=True)
    paths = {name: aligned / f"{name}.csv" for name in (
        "all_models_scores_15min", "ensemble_features_15min", "all_models_events", "run_summary")}
    native_path = output / "ALL_SCORES_30MIN.csv"
    canonical_path = output / "ALL_SCORES_15MIN.csv"
    available = anomalous = 0
    # Reuse the shared validator/feature/event builders, one channel at a time.
    # This avoids keeping million-row long AND wide tables in local RAM.
    for position, key in enumerate(keys):
        part = shards / f"part-{position:03d}"
        native = complete_grid(pd.read_csv(part / "scores_30min.csv"), key,
                               pd.read_csv(part / "day_status.csv"), config)
        canonical = ensemble.validate(_upsample_interval_frame(native), config.timezone)
        first = position == 0
        mode = "w" if first else "a"
        native.to_csv(native_path, index=False, header=first, mode=mode)
        canonical.to_csv(canonical_path, index=False, header=first, mode=mode)
        canonical.to_csv(paths["all_models_scores_15min"], index=False, header=first, mode=mode)
        start = pd.Timestamp(config.detection_start, tz=config.timezone)
        end = pd.Timestamp(config.detection_end + timedelta(days=1), tz=config.timezone)
        ensemble.build_wide(canonical, start, end).to_csv(paths["ensemble_features_15min"], index=False, header=first, mode=mode)
        ensemble.build_events(canonical).to_csv(paths["all_models_events"], index=False, header=first, mode=mode)
        ensemble.build_run_summary(canonical).to_csv(paths["run_summary"], index=False, header=first, mode=mode)
        available += int(canonical.available.sum())
        anomalous += int((canonical.available & canonical.is_anomaly).sum())
        if (position + 1) % 40 == 0 or position + 1 == len(keys):
            print(f"Aligned {position+1}/{len(keys)} channels", flush=True)
    intervals = len(pd.date_range(start, end, freq="15min", inclusive="left")) * len(keys)
    return dict(canonical_intervals=intervals, available_intervals=available,
                flagged_intervals=anomalous, unavailable_intervals=intervals-available)


def signature_for(source, reader, indexed, keys, history, repository, config):
    digest = hashlib.sha256()
    digest.update(json.dumps(config.__dict__, default=str, sort_keys=True).encode())
    digest.update(str(source).encode())
    paths = [repository, Path(__file__), ROOT / "shared/scripts/build_ensemble_features.py",
             ROOT / "shared/data_tools/build_meter_csvs.py"] + sorted(MP_SRC.rglob("*.py"))
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    for key in keys:
        digest.update(key.encode())
        path = history / f"{key}.csv"
        if path.exists():
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        for _, name in indexed.get(key, []):
            digest.update(name.encode())
            if reader.kind == "dir":
                stat = (source / name).stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    if not source.is_dir():
        stat = source.stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="AugSep_meters")
    parser.add_argument("--history-dir", default="data/processed/meter_csvs")
    parser.add_argument("--repository", default="methods/matrix_profile/src/meter_repository.json")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 1), help="Exclusive end")
    parser.add_argument("--baseline-start", type=date.fromisoformat, default=date(2026, 7, 1))
    parser.add_argument("--baseline-end", type=date.fromisoformat, default=date(2026, 7, 31), help="Inclusive baseline end")
    parser.add_argument("--output-dir", default="runs/matrix_profile/results/september_fixed_july_local")
    parser.add_argument("--input-dir", help="Prepared 5-minute CSV directory; default data/processed/matrix_profile/OUTPUT_NAME")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--channel-keys", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    config = DetectionConfig(baseline_start=args.baseline_start, baseline_end=args.baseline_end,
                             detection_start=args.start, detection_end=args.end-timedelta(days=1))
    source, history, repository, output = map(resolve, (args.data, args.history_dir, args.repository, args.output_dir))
    inputs_dir = resolve(args.input_dir) if args.input_dir else ROOT / "data/processed/matrix_profile" / output.name
    if inputs_dir.resolve() == history.resolve():
        raise ValueError("Prepared input directory must not overwrite original history")
    started = time.monotonic()
    with ArchiveReader(str(source)) as reader:
        indexed = index_target(reader, args.start, args.end)
        if not indexed:
            raise ValueError("No target CSVs in the requested period")
        manifest = pd.read_csv(history / "_manifest.csv")
        keys = sorted(set(manifest.meter_channel.astype(str)) | set(indexed))
        if args.channel_keys:
            unknown = set(args.channel_keys) - set(keys)
            if unknown:
                raise ValueError(f"Unknown requested channels: {sorted(unknown)}")
            keys = sorted(set(args.channel_keys))
        if args.limit:
            keys = keys[:args.limit]
        signature = signature_for(source, reader, indexed, keys, history, repository, config)
        signature = hashlib.sha256((signature + str(inputs_dir)).encode()).hexdigest()
        metadata_path = output / "run_metadata.json"
        if output.exists() and any(output.iterdir()):
            if not args.resume:
                raise ValueError("Output is not empty; use --resume or a new --output-dir")
            if not metadata_path.exists() or json.loads(metadata_path.read_text())["signature"] != signature:
                raise ValueError("Resume inputs/parameters/code changed; use a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        metadata = dict(signature=signature, status="running", source=str(source), history_dir=str(history),
                        baseline_start=str(args.baseline_start), baseline_end=str(args.baseline_end),
                        detection_start=str(args.start), detection_end_exclusive=str(args.end),
                        timezone=config.timezone, history_mode="fixed", channels=len(keys),
                        models_run=["matrix_profile"], native_resolution_minutes=30,
                        canonical_resolution_minutes=15, ecdf_ties="disjoint_midrank",
                        input_fingerprint="source file size/mtime; code and metadata SHA256",
                        python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["prepared_input_dir"] = str(inputs_dir)
        input_marker = inputs_dir / "input_signature.json"
        if inputs_dir.exists() and any(inputs_dir.iterdir()):
            if not args.resume or not input_marker.exists() or json.loads(input_marker.read_text()).get("signature") != signature:
                raise ValueError("Prepared input directory already exists with different inputs; choose a new --input-dir")
        if not (args.resume and (inputs_dir / "_manifest.csv").exists()):
            prepare_inputs(reader, indexed, keys, history, inputs_dir, config)
            input_marker.write_text(json.dumps(dict(signature=signature)), encoding="utf-8")
    inputs, failures = discover_merged_channels(inputs_dir, inputs_dir / "_manifest.csv", repository)
    positions = {key: position for position, key in enumerate(keys)}
    shards = output / "shards"

    def part_for(key):
        return shards / f"part-{positions[key]:03d}"

    def is_complete(key):
        part = part_for(key)
        marker = part / "complete.json"
        return (marker.exists() and json.loads(marker.read_text()).get("signature") == signature
                and all((part / name).exists() for name in SHARD_FILES))

    completed = sum(is_complete(key) for key in keys)
    def save(result):
        nonlocal completed
        key = f"{result.context.device_id}_{result.context.channel_idx}"
        populate_missing_day_statuses([result], config)
        part = part_for(key)
        write_shard_outputs([result], part)
        (part / "complete.json").write_text(json.dumps(dict(signature=signature, series_id=key)))
        completed += 1
        print(f"Detected {completed}/{len(keys)} | {key} | {result.status.value} | events={len(result.events)}", flush=True)

    for failure in failures:
        if not is_complete(f"{failure.context.device_id}_{failure.context.channel_idx}"):
            save(failure)
    pending = [item for item in inputs if not is_complete(f"{item.context.device_id}_{item.context.channel_idx}")]
    print(f"Running {len(pending)} pending channels; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_detect_merged_worker, item, config) for item in pending]
        for future in as_completed(futures):
            save(future.result())
    aggregate_shards(shards, output, len(keys))
    counts = export_aligned(shards, keys, output, config)
    statuses = pd.read_csv(output / "ALL_CHANNEL_STATUS.csv")
    day_status = pd.read_csv(output / "ALL_DAY_STATUS.csv")
    metadata.update(counts, status="complete", elapsed_seconds=round(time.monotonic()-started, 1),
                    channel_status_counts=statuses.status.value_counts().to_dict(),
                    day_status_counts=day_status.status.value_counts().to_dict(),
                    reportable_events=len(pd.read_csv(output / "ALL_EVENTS.csv")),
                    candidate_events=len(pd.read_csv(output / "ALL_CANDIDATES.csv")))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
