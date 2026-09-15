from datetime import timedelta

import numpy as np
import pandas as pd

from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.data_quality import detect_flatline_events, overlaps_quality_event
from anomalies.multiscale.events import build_day_events
from anomalies.multiscale.models import (
    ChannelContext,
    ChannelDetectionResult,
    ChannelStatus,
    Confidence,
    DayDetectionStatus,
)
from anomalies.multiscale.preprocessing import (
    baseline_admission_status,
    prepare_channel,
    prepare_rolling_channel,
)
from anomalies.multiscale.rolling import RollingProfileCache, score_rolling_day
from anomalies.multiscale.scoring import score_day


def _interval_score_records(scores, context, config) -> list[dict]:
    """Preserve Matrix Profile's native 30-minute effective resolution."""
    short = np.asarray(scores.short_percentiles, dtype=float) / 100.0
    medium = np.asarray(scores.medium_percentiles, dtype=float) / 100.0
    magnitude = scores.magnitude_percentiles.resample("30min").agg(["max", "mean"])
    magnitude_max = magnitude["max"].to_numpy(dtype=float) / 100.0
    magnitude_mean = magnitude["mean"].to_numpy(dtype=float) / 100.0
    count = min(len(short), len(medium), len(magnitude_max))
    calibration_version = (
        f"fixed_{config.baseline_start}_{config.baseline_end}_v1"
        if config.history_mode == "fixed"
        else "rolling_history_not_frozen"
    )
    rows = []
    day_start = pd.Timestamp(scores.day, tz=config.timezone)
    for index in range(count):
        components = np.asarray([short[index], medium[index], magnitude_max[index]], dtype=float)
        finite = components[np.isfinite(components)]
        maximum = float(np.max(finite)) if finite.size else np.nan
        mean = float(np.mean(finite)) if finite.size else np.nan
        rows.append({
            "model": "matrix_profile",
            "series_id": f"{context.device_id}_{context.channel_idx}",
            "interval_start": day_start + pd.Timedelta(minutes=30 * index),
            "interval_end": day_start + pd.Timedelta(minutes=30 * (index + 1)),
            "raw_score": maximum,
            "max_score": maximum,
            "mean_score": mean,
            "score_std": float(np.std(finite)) if finite.size else np.nan,
            "max_percentile": maximum,
            "mean_percentile": mean,
            "threshold": config.candidate_percentile / 100.0,
            "is_anomaly": bool(maximum > config.candidate_percentile / 100.0),
            "valid_point_count": int(finite.size),
            "expected_point_count": 3,
            "coverage_ratio": float(finite.size / 3.0),
            "source_resolution_minutes": 30,
            "aggregation_method": "native_30m_multiscale_percentiles",
            "data_status": "native_30m",
            "calibration_version": calibration_version,
            "available": bool(finite.size),
            "daily_raw_score": scores.daily_score,
            "daily_outlier": scores.daily_outlier,
            "magnitude_mean_percentile": (
                float(magnitude_mean[index]) if np.isfinite(magnitude_mean[index]) else np.nan
            ),
        })
    return rows


class MultiscaleAnomalyDetector:
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def detect_channel(
        self,
        frame: pd.DataFrame,
        context: ChannelContext,
    ) -> ChannelDetectionResult:
        if self.config.history_mode == "rolling":
            return self._detect_rolling_channel(frame, context)

        preparation = prepare_channel(frame, self.config, context.category_name)
        if preparation.status is not ChannelStatus.READY or preparation.data is None:
            return ChannelDetectionResult(
                context=context,
                status=preparation.status,
                reason=preparation.reason,
            )

        prepared = preparation.data
        baseline_median = float(
            pd.concat([day["pRealKw"] for day in prepared.history_5m.values()]).median()
        )
        events = []
        interval_scores = []
        try:
            for target_day in sorted(prepared.target_5m):
                quality_events = detect_flatline_events(
                    target_day,
                    prepared.target_5m[target_day],
                    prepared.history_5m,
                    context,
                )
                scores = score_day(prepared, target_day, context, self.config)
                interval_scores.extend(_interval_score_records(scores, context, self.config))
                behavior_events = build_day_events(
                    scores,
                    prepared.target_5m[target_day],
                    baseline_median,
                    context,
                    self.config,
                )
                events.extend(quality_events)
                events.extend(
                    event
                    for event in behavior_events
                    if not overlaps_quality_event(event, quality_events)
                )
        except ValueError as error:
            return ChannelDetectionResult(
                context=context,
                status=ChannelStatus.CALIBRATION_FAILED,
                reason=str(error),
            )
        except Exception as error:
            return ChannelDetectionResult(
                context=context,
                status=ChannelStatus.PROCESSING_ERROR,
                reason=f"{type(error).__name__}: {error}",
            )

        reportable = [event for event in events if event.confidence is not Confidence.CANDIDATE]
        candidates = [event for event in events if event.confidence is Confidence.CANDIDATE]
        return ChannelDetectionResult(
            context=context,
            status=ChannelStatus.DETECTED if reportable else ChannelStatus.NO_EVENT,
            events=reportable,
            candidate_events=candidates,
            interval_scores=interval_scores,
            reason=(
                f"Skipped invalid target days: {prepared.invalid_target_dates}"
                if prepared.invalid_target_dates
                else ""
            ),
        )

    def _detect_rolling_channel(
        self,
        frame: pd.DataFrame,
        context: ChannelContext,
    ) -> ChannelDetectionResult:
        status, reason, prepared = prepare_rolling_channel(
            frame, self.config, context.category_name
        )
        if status is not ChannelStatus.READY or prepared is None:
            return ChannelDetectionResult(context=context, status=status, reason=reason)

        arrays = {
            day: day_frame["pRealKw"].to_numpy()
            for day, day_frame in prepared.days_30m.items()
        }
        cache = RollingProfileCache(arrays)
        events = []
        interval_scores = []
        day_statuses: list[DayDetectionStatus] = []
        target_day = self.config.detection_start
        while target_day <= self.config.detection_end:
            history_dates = [
                target_day - timedelta(days=offset)
                for offset in range(self.config.rolling_history_days, 0, -1)
            ]
            valid_history = [day for day in history_dates if day in prepared.days_5m]

            if target_day in prepared.dst_dates:
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        ChannelStatus.DST_TRANSITION,
                        len(valid_history),
                        prepared.invalid_dates[target_day],
                    )
                )
                target_day += timedelta(days=1)
                continue
            if target_day not in prepared.days_5m:
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        ChannelStatus.MISSING_TARGET,
                        len(valid_history),
                        prepared.invalid_dates.get(target_day, "Target day missing"),
                    )
                )
                target_day += timedelta(days=1)
                continue
            if len(valid_history) < self.config.min_valid_history_days:
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        ChannelStatus.INSUFFICIENT_HISTORY,
                        len(valid_history),
                        (
                            f"Only {len(valid_history)} valid rolling history days; "
                            f"{self.config.min_valid_history_days} required"
                        ),
                    )
                )
                target_day += timedelta(days=1)
                continue

            history_5m = {day: prepared.days_5m[day] for day in valid_history}
            admission_status, admission_reason = baseline_admission_status(
                history_5m, self.config, context.category_name
            )
            if admission_status is not ChannelStatus.READY:
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        admission_status,
                        len(valid_history),
                        admission_reason,
                    )
                )
                target_day += timedelta(days=1)
                continue

            try:
                quality_events = detect_flatline_events(
                    target_day,
                    prepared.days_5m[target_day],
                    history_5m,
                    context,
                )
                scores = score_rolling_day(
                    prepared,
                    target_day,
                    valid_history,
                    context,
                    self.config,
                    cache,
                )
                interval_scores.extend(_interval_score_records(scores, context, self.config))
                baseline_median = float(
                    pd.concat([day["pRealKw"] for day in history_5m.values()]).median()
                )
                behavior_events = build_day_events(
                    scores,
                    prepared.days_5m[target_day],
                    baseline_median,
                    context,
                    self.config,
                )
                day_events = quality_events + [
                    event
                    for event in behavior_events
                    if not overlaps_quality_event(event, quality_events)
                ]
                events.extend(day_events)
                reportable_day = any(
                    event.confidence is not Confidence.CANDIDATE for event in day_events
                )
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        ChannelStatus.DETECTED if reportable_day else ChannelStatus.NO_EVENT,
                        len(valid_history),
                    )
                )
            except ValueError as error:
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        ChannelStatus.CALIBRATION_FAILED,
                        len(valid_history),
                        str(error),
                    )
                )
            except Exception as error:
                day_statuses.append(
                    DayDetectionStatus(
                        target_day,
                        ChannelStatus.PROCESSING_ERROR,
                        len(valid_history),
                        f"{type(error).__name__}: {error}",
                    )
                )
            target_day += timedelta(days=1)

        reportable = [event for event in events if event.confidence is not Confidence.CANDIDATE]
        candidates = [event for event in events if event.confidence is Confidence.CANDIDATE]
        processed = [
            item
            for item in day_statuses
            if item.status in {ChannelStatus.DETECTED, ChannelStatus.NO_EVENT}
        ]
        if reportable:
            channel_status = ChannelStatus.DETECTED
        elif processed:
            channel_status = ChannelStatus.NO_EVENT
        elif day_statuses:
            channel_status = day_statuses[0].status
        else:
            channel_status = ChannelStatus.MISSING_TARGET
        return ChannelDetectionResult(
            context=context,
            status=channel_status,
            events=reportable,
            candidate_events=candidates,
            day_statuses=day_statuses,
            interval_scores=interval_scores,
            reason="",
        )
