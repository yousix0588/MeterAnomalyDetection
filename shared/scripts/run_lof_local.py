"""Run the original batch LOF through shared input and non-frozen sidecar output.

Relative paths resolve from group14reshape, not the shell working directory.
Never feed these outputs into the frozen four-model ECDF ensemble.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/data_tools"))
sys.path.insert(0, str(ROOT / "methods/lof/src"))

import pandas as pd
import sklearn
from meter_input import normalize
from lof_batch import (
    CALIBRATION_VERSION, CONTAMINATION, FEATURES, N_NEIGHBORS,
    aggregate_15min, events_from_scores, plot_native_events, score_channel,
)


def resolve(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("LOF paths must stay inside group14reshape")
    return path


def relative(path):
    return resolve(path).relative_to(ROOT.resolve()).as_posix()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def load_config(path):
    value = json.loads(resolve(path).read_text(encoding="utf-8"))
    if value.get("version") != 1 or value.get("model") != "lof":
        raise ValueError("Unsupported LOF configuration")
    if (value.get("mode") != "batch_outlier_detection" or value.get("ensemble_eligible") is not False
            or value.get("features") != FEATURES or value.get("n_neighbors") != N_NEIGHBORS
            or value.get("contamination") != CONTAMINATION):
        raise ValueError("This adapter preserves the approved original LOF mode, features and parameters")
    return value


def load_period(profile_path):
    profile = json.loads(resolve(profile_path).read_text(encoding="utf-8"))
    time = profile["time"]
    if time["input_interval_minutes"] != 5 or time["ensemble_interval_minutes"] != 15:
        raise ValueError("LOF input/output require the existing 5/15-minute contract")
    def boundary(value):
        stamp = pd.Timestamp(value)
        return stamp.tz_localize(time["timezone"]) if stamp.tz is None else stamp.tz_convert(time["timezone"])
    start, end = map(boundary, (time["test_start_inclusive"], time["test_end_exclusive"]))
    if start >= end or any(stamp != stamp.floor("15min") for stamp in (start, end)):
        raise ValueError("Detection boundaries must be ordered and aligned to fifteen minutes")
    return profile, start, end


def discover_inputs(directory, series_ids=None, max_channels=None):
    paths = sorted(path for path in resolve(directory).glob("*.csv") if not path.name.startswith("_"))
    if series_ids:
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("Duplicate --series-id")
        wanted = set(series_ids)
        missing = wanted.difference(path.stem for path in paths)
        if missing:
            raise FileNotFoundError(f"Requested channels missing: {sorted(missing)}")
        paths = [path for path in paths if path.stem in wanted]
    if max_channels is not None:
        if max_channels < 1:
            raise ValueError("--max-channels must be positive")
        paths = paths[:max_channels]
    if not paths:
        raise FileNotFoundError("No per-channel shared input CSV found")
    for path in paths:
        resolve(path)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.stem):
            raise ValueError(f"Unsafe channel filename: {path.name}")
    return paths


def append_csv(frame, destination):
    frame.to_csv(destination, mode="a", header=not destination.exists(), index=False)


def run(config_path="configs/lof.json", *, dataset_profile=None, input_dir=None,
        output_dir=None, series_ids=None, max_channels=None, plots=None):
    config_path = resolve(config_path)
    config = load_config(config_path)
    profile_path = resolve(dataset_profile or config["dataset_profile"])
    profile, start, end = load_period(profile_path)
    directory = resolve(input_dir or config.get("input_directory") or profile["data"]["shared_input_directory"])
    paths = discover_inputs(directory, series_ids, max_channels)
    output = resolve(output_dir or config["output_directory"])
    if not output.is_relative_to((ROOT/"runs/lof").resolve()) or output == (ROOT/"runs/lof").resolve():
        raise ValueError("Use a separate run directory under runs/lof; existing model/data directories are read-only")
    if directory == output or directory in output.parents or output in directory.parents:
        raise ValueError("LOF output must be separate from input")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("LOF output directory is nonempty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    (output/"channels").mkdir(exist_ok=False)
    metadata = dict(
        version=1, model="lof", mode="batch_outlier_detection", status="running",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_id=profile["dataset_id"], timezone=profile["time"]["timezone"],
        detection_start_inclusive=start.isoformat(), detection_end_exclusive=end.isoformat(),
        fit_scope="available feature-complete rows inside detection period, independently per channel",
        historical_baseline_used=False, context_used=False, window_minutes=None,
        features=FEATURES, n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION,
        calibration_version=CALIBRATION_VERSION, percentile_calibration="not_performed",
        ensemble_eligible=False, input_directory=relative(directory),
        output_directory=relative(output), input_sha256={}, channel_fit_details={},
        implementation_source="methods/lof/src/lof_batch.py", backup_required=False,
        code_sha256={relative(path):digest(path) for path in (
            ROOT/"methods/lof/src/lof_batch.py", ROOT/"shared/scripts/run_lof_local.py",
            ROOT/"shared/data_tools/meter_input.py")},
        sklearn_version=sklearn.__version__, pandas_version=pd.__version__,
        selected_channels=[path.stem for path in paths],
        partial_channel_selection=series_ids is not None or max_channels is not None,
        event_semantics=dict(native="consecutive flagged 5-minute points, no gap bridging",
            native_min15="native events of at least 15 consecutive minutes; shorter flags retained elsewhere",
            aligned="consecutive occupied 15-minute bins; bin duration is not continuous fault duration"))
    meta_path = output/"run_metadata.json"
    def save_metadata():
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_metadata()
    (output/"lof_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output/"dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    summaries = []
    try:
        for position, path in enumerate(paths, 1):
            before = digest(path)
            frame = normalize(pd.read_csv(path), profile["time"]["timezone"])
            if digest(path) != before:
                raise ValueError(f"Input changed while reading: {path.name}")
            selected = frame.loc[frame.timestamp.ge(start) & frame.timestamp.lt(end)].copy()
            points, details = score_channel(selected, path.stem)
            points["source_file"] = relative(path)
            aligned = aggregate_15min(points, start, end, series_id=path.stem)
            native_events = events_from_scores(points, 5)
            aligned_events = events_from_scores(aligned, 15)
            long_events = native_events.loc[native_events.duration_minutes.ge(15)].copy()
            anomalies = points.loc[points.available & points.is_anomaly.fillna(False)].copy()
            channel = output/"channels"/path.stem
            channel.mkdir()
            points.to_csv(channel/"point_scores_5min.csv", index=False)
            aligned.to_csv(channel/"scores_15min.csv", index=False)
            native_events.to_csv(channel/"events_5min.csv", index=False)
            append_csv(anomalies, output/"ALL_ANOMALIES_5MIN.csv")
            append_csv(aligned, output/"ALL_SCORES_15MIN.csv")
            append_csv(native_events, output/"ALL_EVENTS_5MIN.csv")
            append_csv(long_events, output/"ALL_EVENTS_NATIVE_MIN15MIN.csv")
            append_csv(aligned_events, output/"ALL_EVENTS_15MIN.csv")
            summary = dict(model="lof", series_id=path.stem, status=details["status"],
                source_rows=len(frame), target_rows=len(points), valid_feature_rows=details["valid_rows"],
                scored_points=int(points.available.sum()), anomalous_points=len(anomalies),
                intervals=len(aligned), available_intervals=int(aligned.available.sum()),
                anomalous_intervals=int(aligned.is_anomaly.fillna(False).sum()),
                native_events=len(native_events), native_events_min15=len(long_events),
                aligned_events=len(aligned_events), threshold=details["threshold"],
                calibration_version=CALIBRATION_VERSION, ensemble_eligible=False)
            summaries.append(summary)
            metadata["input_sha256"][relative(path)] = before
            metadata["channel_fit_details"][path.stem] = details
            print(f"LOF {position}/{len(paths)} {path.stem}: {summary['scored_points']} scored, {len(anomalies)} flagged points", flush=True)
        pd.DataFrame(summaries).to_csv(output/"run_summary.csv", index=False)
        make_plots = config["plots"] if plots is None else plots
        if make_plots:
            events = pd.read_csv(output/"ALL_EVENTS_NATIVE_MIN15MIN.csv")
            plot_native_events(events, output/"plots", start, end)
        metadata["status"] = "complete"
        metadata["channels"] = len(summaries)
        metadata["scored_channels"] = sum(row["status"] == "scored" for row in summaries)
        metadata["output_sha256"] = {relative(path):digest(path) for path in sorted(output.rglob("*.csv"))}
        save_metadata()
        return output
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_metadata()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lof.json")
    parser.add_argument("--dataset-profile")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--series-id", action="append")
    parser.add_argument("--max-channels", type=int)
    parser.add_argument("--plots", action="store_true", default=None)
    args = parser.parse_args(argv)
    result = run(args.config, dataset_profile=args.dataset_profile, input_dir=args.input_dir,
                 output_dir=args.output_dir, series_ids=args.series_id, max_channels=args.max_channels, plots=args.plots)
    print(f"Completed: {relative(result)} (not eligible for frozen-ECDF ensemble)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
