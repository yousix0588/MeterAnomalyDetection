import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from anomalies.multiscale.batch import ChannelInput
from anomalies.multiscale.models import AnomalyEvent, ChannelDetectionResult


EVENT_COLUMNS = [
    "event_id",
    "device_id",
    "channel_idx",
    "organization",
    "category_id",
    "category_name",
    "start_time",
    "end_time",
    "duration_minutes",
    "confidence",
    "event_type",
    "dominant_scale",
    "dominant_metric",
    "short_percentile",
    "medium_percentile",
    "daily_score",
    "magnitude_percentile",
    "change_point_confirmed",
    "corroborating_metrics",
    "baseline_median_kw",
    "observed_mean_kw",
    "observed_max_kw",
    "absolute_change_kw",
    "rated_power_ratio",
    "is_common_mode",
    "common_mode_channel_count",
    "evidence",
]

CHANNEL_STATUS_COLUMNS = [
    "device_id",
    "channel_idx",
    "organization",
    "category_id",
    "category_name",
    "status",
    "event_count",
    "candidate_count",
    "reason",
]

DAY_STATUS_COLUMNS = [
    "device_id",
    "channel_idx",
    "organization",
    "category_id",
    "category_name",
    "date",
    "status",
    "valid_history_days",
    "reason",
]

INTERVAL_SCORE_COLUMNS = [
    "model", "series_id", "interval_start", "interval_end", "raw_score",
    "max_score", "mean_score", "score_std", "max_percentile", "mean_percentile",
    "threshold", "is_anomaly", "valid_point_count", "expected_point_count",
    "coverage_ratio", "source_resolution_minutes", "aggregation_method",
    "data_status", "calibration_version", "available", "daily_raw_score",
    "daily_outlier", "magnitude_mean_percentile",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _event_hour_bounds(event: AnomalyEvent) -> tuple[float, float]:
    day_start = event.start_time.normalize()
    start_hour = (event.start_time - day_start).total_seconds() / 3600.0
    end_hour = (event.end_time - day_start).total_seconds() / 3600.0
    return start_hour, end_hour


def _interval_frame(results: list[ChannelDetectionResult]) -> pd.DataFrame:
    rows = [row for result in results for row in result.interval_scores]
    return pd.DataFrame(rows, columns=INTERVAL_SCORE_COLUMNS)


def _upsample_interval_frame(native: pd.DataFrame) -> pd.DataFrame:
    """Expose 30-minute context on a 15-minute grid without claiming new resolution."""
    if native.empty:
        return native.copy()
    first = native.copy()
    second = native.copy()
    first["interval_end"] = pd.to_datetime(first["interval_start"]) + pd.Timedelta(minutes=15)
    second["interval_start"] = pd.to_datetime(second["interval_start"]) + pd.Timedelta(minutes=15)
    second["interval_end"] = pd.to_datetime(second["interval_start"]) + pd.Timedelta(minutes=15)
    result = pd.concat([first, second], ignore_index=True)
    result["aggregation_method"] = "repeat_from_native_30m"
    result["data_status"] = "upsampled_from_30m"
    result["source_resolution_minutes"] = 30
    return result.sort_values(["series_id", "interval_start"]).reset_index(drop=True)


def write_detection_outputs(
    results: list[ChannelDetectionResult],
    output_dir: str | Path,
    include_candidates: bool = False,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    events = [event for result in results for event in result.events]
    if include_candidates:
        events.extend(event for result in results for event in result.candidate_events)

    for day in sorted({event.start_time.date() for event in events}):
        day_events = [event for event in events if event.start_time.date() == day]
        day_dir = root / day.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        records = [event.to_dict() for event in day_events]
        (day_dir / "events.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        csv_records = [
            {key: _csv_value(value) for key, value in record.items()} for record in records
        ]
        pd.DataFrame(csv_records, columns=EVENT_COLUMNS).to_csv(
            day_dir / "events.csv",
            index=False,
        )

    status_path = root / "channel_status.csv"
    status_records = [
        {
            "device_id": result.context.device_id,
            "channel_idx": result.context.channel_idx,
            "organization": result.context.organization,
            "category_id": result.context.category_id,
            "category_name": result.context.category_name,
            "status": result.status.value,
            "event_count": len(result.events),
            "candidate_count": len(result.candidate_events),
            "reason": result.reason,
        }
        for result in results
    ]
    pd.DataFrame(status_records).to_csv(status_path, index=False)

    summary_path = root / "daily_summary.csv"
    summary_records = [
        {
            "date": event.start_time.date().isoformat(),
            "category_name": event.category_name,
            "confidence": event.confidence.value,
            "event_type": event.event_type.value,
        }
        for event in events
    ]
    if summary_records:
        summary = (
            pd.DataFrame(summary_records)
            .groupby(["date", "category_name", "confidence", "event_type"], as_index=False)
            .size()
            .rename(columns={"size": "event_count"})
        )
    else:
        summary = pd.DataFrame(
            columns=["date", "category_name", "confidence", "event_type", "event_count"]
        )
    summary.to_csv(summary_path, index=False)
    native_scores_path = root / "scores_30min.csv"
    canonical_scores_path = root / "scores_15min.csv"
    native_scores = _interval_frame(results)
    native_scores.to_csv(native_scores_path, index=False)
    _upsample_interval_frame(native_scores).to_csv(canonical_scores_path, index=False)
    return {
        "status_csv": status_path,
        "summary_csv": summary_path,
        "scores_30min_csv": native_scores_path,
        "scores_15min_csv": canonical_scores_path,
    }


def _event_frame(events: list[AnomalyEvent]) -> pd.DataFrame:
    records = [event.to_dict() for event in events]
    csv_records = [
        {key: _csv_value(value) for key, value in record.items()} for record in records
    ]
    return pd.DataFrame(csv_records, columns=EVENT_COLUMNS)


def write_shard_outputs(
    results: list[ChannelDetectionResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    events_path = root / "events.csv"
    candidates_path = root / "candidates.csv"
    channel_status_path = root / "channel_status.csv"
    day_status_path = root / "day_status.csv"
    scores_30min_path = root / "scores_30min.csv"
    scores_15min_path = root / "scores_15min.csv"

    _event_frame([event for result in results for event in result.events]).to_csv(
        events_path, index=False
    )
    _event_frame([event for result in results for event in result.candidate_events]).to_csv(
        candidates_path, index=False
    )
    channel_records = [
        {
            "device_id": result.context.device_id,
            "channel_idx": result.context.channel_idx,
            "organization": result.context.organization,
            "category_id": result.context.category_id,
            "category_name": result.context.category_name,
            "status": result.status.value,
            "event_count": len(result.events),
            "candidate_count": len(result.candidate_events),
            "reason": result.reason,
        }
        for result in results
    ]
    pd.DataFrame(channel_records, columns=CHANNEL_STATUS_COLUMNS).to_csv(
        channel_status_path, index=False
    )
    day_records = [
        {
            "device_id": result.context.device_id,
            "channel_idx": result.context.channel_idx,
            "organization": result.context.organization,
            "category_id": result.context.category_id,
            "category_name": result.context.category_name,
            "date": day_status.date.isoformat(),
            "status": day_status.status.value,
            "valid_history_days": day_status.valid_history_days,
            "reason": day_status.reason,
        }
        for result in results
        for day_status in result.day_statuses
    ]
    pd.DataFrame(day_records, columns=DAY_STATUS_COLUMNS).to_csv(day_status_path, index=False)
    native_scores = _interval_frame(results)
    native_scores.to_csv(scores_30min_path, index=False)
    _upsample_interval_frame(native_scores).to_csv(scores_15min_path, index=False)
    return {
        "events_csv": events_path,
        "candidates_csv": candidates_path,
        "channel_status_csv": channel_status_path,
        "day_status_csv": day_status_path,
        "scores_30min_csv": scores_30min_path,
        "scores_15min_csv": scores_15min_path,
    }


def write_evidence_plots(
    results: list[ChannelDetectionResult],
    inputs: list[ChannelInput],
    output_dir: str | Path,
) -> list[Path]:
    root = Path(output_dir)
    frames = {
        (item.context.device_id, item.context.channel_idx): item.frame for item in inputs
    }
    paths: list[Path] = []
    for result in results:
        frame = frames.get((result.context.device_id, result.context.channel_idx))
        if frame is None:
            continue
        data = frame.copy()
        data.index = pd.to_datetime(data.index)
        for event in result.events:
            target = data.loc[data.index.date == event.start_time.date(), "pRealKw"]
            if target.empty:
                continue
            nearest = (
                event.evidence.get("nearest_neighbors", {})
                .get(event.dominant_scale, {})
                .get("reference_day")
            )
            reference = pd.Series(dtype=float)
            if nearest:
                reference_day = pd.Timestamp(nearest).date()
                reference = data.loc[data.index.date == reference_day, "pRealKw"]

            target_x = (target.index - target.index.normalize()).total_seconds() / 3600.0
            figure, axis = plt.subplots(figsize=(11, 4.5))
            if not reference.empty:
                reference_x = (
                    reference.index - reference.index.normalize()
                ).total_seconds() / 3600.0
                axis.plot(reference_x, reference.to_numpy(), label=f"nearest {nearest}", alpha=0.75)
            axis.plot(target_x, target.to_numpy(), label=f"target {event.start_time.date()}", linewidth=1.4)
            start_hour, end_hour = _event_hour_bounds(event)
            axis.axvspan(start_hour, end_hour, color="tab:red", alpha=0.2, label="event")
            axis.set(
                title=(
                    f"{event.device_id}_{event.channel_idx} | {event.event_type.value} | "
                    f"{event.confidence.value}"
                ),
                xlabel="Local hour",
                ylabel="pRealKw (kW)",
                xlim=(0, 24),
            )
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
            figure.tight_layout()
            evidence_dir = root / event.start_time.date().isoformat() / "evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            path = evidence_dir / f"{event.event_id}.png"
            figure.savefig(path, dpi=140)
            plt.close(figure)
            paths.append(path)
    return paths
