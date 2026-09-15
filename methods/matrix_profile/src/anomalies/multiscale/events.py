from hashlib import sha1

import numpy as np
import pandas as pd

from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.models import (
    AnomalyEvent,
    ChannelContext,
    Confidence,
    DayScores,
    EventType,
)


def _candidate_mask(scores: DayScores, config: DetectionConfig) -> np.ndarray:
    short = np.asarray(scores.short_percentiles, dtype=float)
    medium = np.asarray(scores.medium_percentiles, dtype=float)
    return (
        (short > config.candidate_percentile)
        | (medium > config.candidate_percentile)
        | ((short > config.joint_percentile) & (medium > config.joint_percentile))
    )


def _candidate_groups(mask: np.ndarray, config: DetectionConfig) -> list[tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []
    allowed_missing_bins = config.merge_gap_minutes // 30
    groups: list[tuple[int, int]] = []
    start = previous = int(positions[0])
    for value in positions[1:]:
        position = int(value)
        if position - previous - 1 <= allowed_missing_bins:
            previous = position
            continue
        groups.append((start, previous))
        start = previous = position
    groups.append((start, previous))
    return groups


def _refine_boundaries(
    coarse_start: pd.Timestamp,
    coarse_end: pd.Timestamp,
    magnitude: pd.Series,
    config: DetectionConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    local = magnitude.loc[(magnitude.index >= coarse_start) & (magnitude.index < coarse_end)]
    high = local[local >= config.candidate_percentile]
    if high.empty:
        return coarse_start, coarse_end

    start_position = magnitude.index.get_loc(high.index[0])
    end_position = magnitude.index.get_loc(high.index[-1])
    while start_position > 0 and magnitude.iloc[start_position - 1] >= config.boundary_low_percentile:
        start_position -= 1
    while end_position + 1 < len(magnitude) and magnitude.iloc[end_position + 1] >= config.boundary_low_percentile:
        end_position += 1
    return magnitude.index[start_position], magnitude.index[end_position] + pd.Timedelta(minutes=5)


def _event_id(context: ChannelContext, start: pd.Timestamp, end: pd.Timestamp) -> str:
    raw = f"{context.device_id}|{context.channel_idx}|{start.isoformat()}|{end.isoformat()}"
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_day_events(
    scores: DayScores,
    target_5m: pd.DataFrame,
    baseline_median_kw: float,
    context: ChannelContext,
    config: DetectionConfig,
) -> list[AnomalyEvent]:
    day_start = pd.Timestamp(scores.day, tz=config.timezone)
    events: list[AnomalyEvent] = []

    for first_bin, last_bin in _candidate_groups(_candidate_mask(scores, config), config):
        coarse_start = day_start + pd.Timedelta(minutes=30 * first_bin)
        coarse_end = day_start + pd.Timedelta(minutes=30 * (last_bin + 1))
        start, end = _refine_boundaries(
            coarse_start,
            coarse_end,
            scores.magnitude_percentiles,
            config,
        )
        duration = int((end - start).total_seconds() // 60)
        if duration < config.minimum_event_minutes:
            continue

        short_max = float(np.nanmax(scores.short_percentiles[first_bin : last_bin + 1]))
        medium_max = float(np.nanmax(scores.medium_percentiles[first_bin : last_bin + 1]))
        short_signal = short_max > config.joint_percentile
        medium_signal = medium_max > config.joint_percentile

        local_magnitude = scores.magnitude_percentiles.loc[
            (scores.magnitude_percentiles.index >= start)
            & (scores.magnitude_percentiles.index < end)
        ]
        magnitude_max = float(local_magnitude.max()) if not local_magnitude.empty else 0.0
        magnitude_signal = magnitude_max >= config.candidate_percentile

        corroborating: list[str] = []
        if magnitude_signal:
            corroborating.append("pRealKw")
        for metric, percentiles in scores.metric_percentiles.items():
            local_metric = percentiles.loc[(percentiles.index >= start) & (percentiles.index < end)]
            if not local_metric.empty and float(local_metric.max()) >= config.candidate_percentile:
                corroborating.append(metric)
        change_confirmed = any(
            start - pd.Timedelta(minutes=30) <= point <= end + pd.Timedelta(minutes=30)
            for point in scores.change_points
        )
        if change_confirmed:
            corroborating.append("change_point")
        if scores.daily_outlier:
            corroborating.append("daily_profile")

        extra_confirmation = magnitude_signal or change_confirmed or scores.daily_outlier or any(
            metric != "pRealKw" for metric in corroborating
        )
        if short_signal and medium_signal and extra_confirmation:
            confidence = Confidence.HIGH
        elif (short_signal and medium_signal) or extra_confirmation:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.CANDIDATE

        if scores.daily_outlier:
            event_type = EventType.DAILY_PATTERN_CHANGE
        elif change_confirmed:
            event_type = EventType.STATE_CHANGE
        elif medium_signal and not short_signal:
            event_type = EventType.SUSTAINED_DEVIATION
        elif magnitude_signal and not (short_signal or medium_signal):
            event_type = EventType.MAGNITUDE_ANOMALY
        else:
            event_type = EventType.SHAPE_CHANGE

        dominant_scale = "short" if short_max >= medium_max else "medium"
        observed = target_5m.loc[(target_5m.index >= start) & (target_5m.index < end), "pRealKw"].dropna()
        observed_mean = float(observed.mean()) if not observed.empty else None
        observed_max = float(observed.max()) if not observed.empty else None
        absolute_change = (
            abs(observed_mean - baseline_median_kw) if observed_mean is not None else None
        )
        rated_ratio = None
        if context.rated_power_kw and absolute_change is not None:
            rated_ratio = absolute_change / context.rated_power_kw

        events.append(
            AnomalyEvent(
                event_id=_event_id(context, start, end),
                device_id=context.device_id,
                channel_idx=context.channel_idx,
                organization=context.organization,
                category_id=context.category_id,
                category_name=context.category_name,
                start_time=start,
                end_time=end,
                duration_minutes=duration,
                confidence=confidence,
                event_type=event_type,
                dominant_scale=dominant_scale,
                dominant_metric="pRealKw",
                short_percentile=short_max,
                medium_percentile=medium_max,
                daily_score=scores.daily_score,
                magnitude_percentile=magnitude_max,
                change_point_confirmed=change_confirmed,
                corroborating_metrics=corroborating,
                baseline_median_kw=baseline_median_kw,
                observed_mean_kw=observed_mean,
                observed_max_kw=observed_max,
                absolute_change_kw=absolute_change,
                rated_power_ratio=rated_ratio,
                evidence={
                    "coarse_start": coarse_start.isoformat(),
                    "coarse_end": coarse_end.isoformat(),
                    "nearest_neighbors": scores.nearest_neighbors,
                },
            )
        )
    return events
