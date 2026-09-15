from datetime import date
from hashlib import sha1

import numpy as np
import pandas as pd

from anomalies.multiscale.models import (
    AnomalyEvent,
    ChannelContext,
    Confidence,
    EventType,
)


NEAR_ZERO_POWER_KW = 0.05
MIN_SAME_WEEKDAY_HISTORY_DAYS = 3


def _flatline_runs(
    series: pd.Series,
    tolerance_kw: float,
    minimum_points: int,
) -> list[tuple[int, int]]:
    values = series.astype(float).to_numpy()
    runs: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(values) + 1):
        changed = (
            position == len(values)
            or not np.isfinite(values[position - 1])
            or not np.isfinite(values[position])
            or abs(values[position] - values[position - 1]) > tolerance_kw
        )
        if not changed:
            continue
        segment = values[start:position]
        if (
            position - start >= minimum_points
            and np.isfinite(segment).all()
            and float(np.ptp(segment)) <= tolerance_kw
        ):
            runs.append((start, position))
        start = position
    return runs


def _select_state_reference_days(
    target_day: date,
    history: dict[date, pd.DataFrame],
) -> dict[date, pd.DataFrame]:
    same_weekday = {
        day: frame for day, frame in history.items() if day.weekday() == target_day.weekday()
    }
    if len(same_weekday) >= MIN_SAME_WEEKDAY_HISTORY_DAYS:
        return same_weekday
    return history


def _quality_event_id(
    context: ChannelContext,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    raw = f"quality|{context.device_id}|{context.channel_idx}|{start.isoformat()}|{end.isoformat()}"
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def detect_flatline_events(
    target_day: date,
    target: pd.DataFrame,
    history: dict[date, pd.DataFrame],
    context: ChannelContext,
    tolerance_kw: float = 0.001,
    minimum_points: int = 24,
) -> list[AnomalyEvent]:
    if "pRealKw" not in target or not history:
        return []
    state_history = _select_state_reference_days(target_day, history)
    baseline = np.vstack(
        [
            frame["pRealKw"].astype(float).to_numpy()
            for _, frame in sorted(state_history.items())
        ]
    )
    events: list[AnomalyEvent] = []
    for start_position, stop_position in _flatline_runs(
        target["pRealKw"], tolerance_kw, minimum_points
    ):
        historical_ranges = np.nanmax(
            baseline[:, start_position:stop_position], axis=1
        ) - np.nanmin(baseline[:, start_position:stop_position], axis=1)
        typical_historical_range = float(np.nanmedian(historical_ranges))
        if typical_historical_range <= max(tolerance_kw * 10.0, 0.05):
            continue

        start = target.index[start_position]
        end = target.index[stop_position - 1] + pd.Timedelta(minutes=5)
        observed = target["pRealKw"].iloc[start_position:stop_position]
        historical_segment = baseline[:, start_position:stop_position]
        baseline_median = float(np.nanmedian(historical_segment))
        observed_mean = float(observed.mean())
        historical_levels = np.nanmean(np.abs(historical_segment), axis=1)
        typical_historical_level = float(np.nanmedian(historical_levels))
        if (
            abs(observed_mean) <= NEAR_ZERO_POWER_KW
            and typical_historical_level <= NEAR_ZERO_POWER_KW
        ):
            continue
        change = abs(observed_mean - baseline_median)
        ratio = change / context.rated_power_kw if context.rated_power_kw else None
        events.append(
            AnomalyEvent(
                event_id=_quality_event_id(context, start, end),
                device_id=context.device_id,
                channel_idx=context.channel_idx,
                organization=context.organization,
                category_id=context.category_id,
                category_name=context.category_name,
                start_time=start,
                end_time=end,
                duration_minutes=int((end - start).total_seconds() // 60),
                confidence=Confidence.PHYSICAL,
                event_type=EventType.DATA_QUALITY,
                dominant_scale="data_quality",
                dominant_metric="pRealKw",
                corroborating_metrics=["flatline"],
                baseline_median_kw=baseline_median,
                observed_mean_kw=observed_mean,
                observed_max_kw=float(observed.max()),
                absolute_change_kw=change,
                rated_power_ratio=ratio,
                evidence={
                    "quality_issue": "flatline",
                    "identical_point_count": stop_position - start_position,
                    "tolerance_kw": tolerance_kw,
                    "typical_historical_range_kw": typical_historical_range,
                    "typical_historical_level_kw": typical_historical_level,
                    "state_reference_day_count": len(state_history),
                    "state_reference_weekday": target_day.strftime("%A"),
                    "target_day": target_day.isoformat(),
                },
            )
        )
    return events


def overlaps_quality_event(
    event: AnomalyEvent,
    quality_events: list[AnomalyEvent],
) -> bool:
    return any(
        quality.start_time < event.end_time and quality.end_time > event.start_time
        for quality in quality_events
    )
