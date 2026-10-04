#!/usr/bin/env python3
"""Validate canonical model scores and build the 15-minute ensemble feature table."""

from __future__ import annotations

import argparse
from hashlib import sha1
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "model", "series_id", "interval_start", "interval_end", "raw_score",
    "max_score", "mean_score", "score_std", "max_percentile", "mean_percentile",
    "threshold", "is_anomaly", "coverage_ratio", "source_resolution_minutes",
    "aggregation_method", "data_status", "calibration_version", "available",
}
ALLOWED_MODELS = {"lstm_autoencoder", "lstm_vae", "rpca", "matrix_profile"}


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            preferred = [
                path / "scores_15min.csv",
                path / "ALL_METERS_scores_15min.csv",
                path / "ALL_SCORES_15MIN.csv",
            ]
            candidates = [candidate for candidate in preferred if candidate.exists()]
            if not candidates:
                aggregate_parts = sorted(path.glob("ALL_METERS_scores_15min_part*.csv"))
                candidates = aggregate_parts or sorted(path.glob("*_scores_15min.csv"))
        else:
            candidates = [path]
        if not candidates:
            raise FileNotFoundError(f"No canonical score CSV found under {path}")
        for candidate in candidates:
            frame = pd.read_csv(candidate)
            missing = REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                raise ValueError(f"{candidate} is missing canonical columns: {sorted(missing)}")
            frame["source_file"] = str(candidate)
            frames.append(frame)
    if not frames:
        raise ValueError("At least one --input is required")
    return pd.concat(frames, ignore_index=True)


def validate(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    result = frame.copy()
    unknown = set(result["model"].dropna().unique()).difference(ALLOWED_MODELS)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    if result["calibration_version"].astype(str).str.contains("not_frozen").any():
        raise ValueError("A score file uses a non-frozen calibration distribution")

    for column in ("interval_start", "interval_end"):
        parsed = pd.to_datetime(result[column], utc=True, errors="raise")
        result[column] = parsed.dt.tz_convert(timezone)
    duration = result["interval_end"] - result["interval_start"]
    if not duration.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Every canonical score row must cover exactly 15 minutes")
    if not result["interval_start"].dt.minute.isin([0, 15, 30, 45]).all():
        raise ValueError("interval_start is not aligned to the 15-minute grid")
    for column in ("max_percentile", "mean_percentile"):
        values = pd.to_numeric(result[column], errors="coerce")
        if ((values < -1e-9) | (values > 1 + 1e-9)).any():
            raise ValueError(f"{column} must use the frozen 0..1 ECDF scale")
        result[column] = values.clip(0.0, 1.0)
    result["available"] = parse_bool(result["available"])
    result["is_anomaly"] = parse_bool(result["is_anomaly"])
    matrix_profile = result["model"].eq("matrix_profile")
    bad_mp_status = matrix_profile & ~result["data_status"].eq("upsampled_from_30m")
    bad_mp_resolution = matrix_profile & ~pd.to_numeric(
        result["source_resolution_minutes"], errors="coerce"
    ).eq(30)
    if bad_mp_status.any() or bad_mp_resolution.any():
        raise ValueError(
            "Matrix Profile canonical rows must disclose data_status=upsampled_from_30m "
            "and source_resolution_minutes=30"
        )
    duplicates = result.duplicated(["model", "series_id", "interval_start"], keep=False)
    if duplicates.any():
        examples = result.loc[duplicates, ["model", "series_id", "interval_start"]].head()
        raise ValueError(f"Duplicate model/series/interval rows:\n{examples}")
    return result


def build_wide(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = frame.loc[(frame["interval_start"] >= start) & (frame["interval_start"] < end)].copy()
    series_ids = sorted(frame["series_id"].dropna().unique())
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    full_index = pd.MultiIndex.from_product(
        [series_ids, grid], names=["series_id", "interval_start"]
    )
    features = frame.pivot(
        index=["series_id", "interval_start"], columns="model",
        values=[
            "max_percentile", "mean_percentile", "max_score", "mean_score", "available",
            "source_resolution_minutes", "data_status", "aggregation_method",
        ],
    )
    features.columns = [f"{model}_{metric}" for metric, model in features.columns]
    features = features.reindex(full_index).reset_index()
    for model in ALLOWED_MODELS:
        for metric in (
            "max_percentile", "mean_percentile", "max_score", "mean_score",
            "source_resolution_minutes", "data_status", "aggregation_method",
        ):
            column = f"{model}_{metric}"
            if column not in features:
                features[column] = pd.NA
        column = f"{model}_available"
        if column not in features:
            features[column] = False
        else:
            features[column] = features[column].fillna(False).astype(bool)
    return features.sort_values(["series_id", "interval_start"])


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model", "event_id", "series_id", "start", "end", "duration_minutes",
        "max_raw_score", "mean_raw_score", "max_score_percentile", "threshold",
        "interval_count", "calibration_version",
    ]
    anomalous = frame.loc[frame["available"] & frame["is_anomaly"]].copy()
    if anomalous.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for (model, series_id), group in anomalous.groupby(["model", "series_id"]):
        group = group.sort_values("interval_start").copy()
        gap = group["interval_start"].diff()
        group["_event"] = (gap.isna() | gap.gt(pd.Timedelta(minutes=15))).cumsum()
        for _, event in group.groupby("_event"):
            start = event["interval_start"].iloc[0]
            end = event["interval_end"].iloc[-1]
            identity = f"{model}|{series_id}|{start.isoformat()}|{end.isoformat()}"
            records.append({
                "model": model,
                "event_id": sha1(identity.encode("utf-8")).hexdigest()[:16],
                "series_id": series_id,
                "start": start,
                "end": end,
                "duration_minutes": int((end - start).total_seconds() / 60),
                "max_raw_score": float(event["max_score"].max()),
                "mean_raw_score": float(event["mean_score"].mean()),
                "max_score_percentile": float(event["max_percentile"].max()),
                "threshold": float(event["threshold"].iloc[0]),
                "interval_count": len(event),
                "calibration_version": str(event["calibration_version"].iloc[0]),
            })
    return pd.DataFrame(records, columns=columns).sort_values(["series_id", "start", "model"])


def build_run_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["model", "series_id", "intervals", "available_intervals", "anomalies"])
    return (
        frame.groupby(["model", "series_id"], as_index=False)
        .agg(
            intervals=("interval_start", "size"),
            available_intervals=("available", "sum"),
            anomalies=("is_anomaly", "sum"),
            first_interval=("interval_start", "min"),
            last_interval=("interval_end", "max"),
            calibration_version=("calibration_version", "first"),
            data_status=("data_status", lambda values: "|".join(sorted(set(map(str, values))))),
        )
        .sort_values(["model", "series_id"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Canonical CSV or directory; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timezone", default="Australia/Sydney")
    parser.add_argument("--start", default="2026-08-01")
    parser.add_argument("--end", default="2026-09-01")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz=args.timezone)
    end = pd.Timestamp(args.end, tz=args.timezone)
    long = validate(read_inputs(args.input), args.timezone)
    long = long.loc[(long["interval_start"] >= start) & (long["interval_start"] < end)]
    long.sort_values(["series_id", "interval_start", "model"]).to_csv(
        output / "all_models_scores_15min.csv", index=False
    )
    build_wide(long, start, end).to_csv(output / "ensemble_features_15min.csv", index=False)
    build_events(long).to_csv(output / "all_models_events.csv", index=False)
    build_run_summary(long).to_csv(output / "run_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
