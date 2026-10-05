from datetime import date
from typing import TypeVar

import numpy as np
import pandas as pd

from anomalies.multiscale.change_points import detect_persistent_change_points
from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.models import ChannelContext, DayScores, PreparedChannel
from anomalies.multiscale.profile import (
    ProfileResult,
    calibrate_leave_one_day_out,
    empirical_percentiles,
    matrix_profile_ab_join,
)


T = TypeVar("T")


def select_reference_days(
    history: dict[date, T],
    target_day: date,
    category_name: str,
    min_group_days: int,
) -> dict[date, T]:
    if category_name.casefold() == "solar generation":
        return dict(history)
    target_is_weekend = target_day.weekday() >= 5
    matching = {
        day: values
        for day, values in history.items()
        if (day.weekday() >= 5) == target_is_weekend
    }
    return matching if len(matching) >= min_group_days else dict(history)


def infer_solar_active_mask(history: dict[date, np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("Solar activity inference requires historical days")
    matrix = np.vstack([np.abs(values) for _, values in sorted(history.items())])
    typical = np.nanmedian(matrix, axis=0)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.zeros(matrix.shape[1], dtype=bool)
    threshold = max(0.05, float(np.nanquantile(finite, 0.95)) * 0.05)
    return typical > threshold


def same_slot_percentiles(
    history: dict[date, pd.Series],
    target: pd.Series,
) -> pd.Series:
    if not history:
        raise ValueError("Magnitude scoring requires historical days")
    matrix = np.vstack([series.astype(float).to_numpy() for _, series in sorted(history.items())])
    target_values = target.astype(float).to_numpy()
    if matrix.shape[1] != len(target_values):
        raise ValueError("Historical and target days must have the same number of slots")

    medians = np.nanmedian(matrix, axis=0)
    baseline_deviations = np.abs(matrix - medians)
    target_deviations = np.abs(target_values - medians)
    result = np.full(len(target_values), np.nan, dtype=float)
    for slot, score in enumerate(target_deviations):
        reference = baseline_deviations[:, slot]
        reference = reference[np.isfinite(reference)]
        if not np.isfinite(score) or reference.size == 0:
            continue
        tied = np.isclose(reference, score, rtol=1e-9, atol=1e-12)
        lower = np.count_nonzero((reference < score) & ~tied)
        equal = np.count_nonzero(tied)
        result[slot] = (lower + 0.5 * equal) / reference.size * 100.0
    return pd.Series(result, index=target.index, name="magnitude_percentile")


def project_window_scores(window_scores: np.ndarray, window: int, length: int = 48) -> np.ndarray:
    projected = np.full(length, np.nan, dtype=float)
    for start, score in enumerate(np.asarray(window_scores, dtype=float)):
        if not np.isfinite(score):
            continue
        stop = min(start + window, length)
        current = projected[start:stop]
        projected[start:stop] = np.where(
            np.isnan(current),
            score,
            np.maximum(current, score),
        )
    return projected


def _daily_outlier(score: float, calibration: np.ndarray) -> bool:
    finite = calibration[np.isfinite(calibration)]
    if finite.size == 0 or not np.isfinite(score):
        return False
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return score > float(np.max(finite)) and score > median + 4.0 * mad


def _neighbor_evidence(profile: ProfileResult) -> dict[str, object]:
    if not np.isfinite(profile.scores).any():
        return {}
    position = int(np.nanargmax(profile.scores))
    day = profile.neighbor_days[position]
    return {
        "query_window_start": position,
        "reference_day": day.isoformat() if day is not None else None,
        "reference_window_start": int(profile.neighbor_starts[position]),
        "distance": float(profile.scores[position]),
    }


def score_day(
    prepared: PreparedChannel,
    target_day: date,
    context: ChannelContext,
    config: DetectionConfig,
    calibration_cache: dict | None = None,
) -> DayScores:
    # This cache belongs to one channel's fixed baseline. Target values never
    # enter it; weekdays/weekends have distinct reference-date keys.
    def frozen_calibration(arrays, window, scale):
        key = (scale, window, tuple(sorted(arrays)))
        if calibration_cache is None:
            return calibrate_leave_one_day_out(arrays, window)
        if key not in calibration_cache:
            calibration_cache[key] = calibrate_leave_one_day_out(arrays, window)
        return calibration_cache[key]

    reference_30m = select_reference_days(
        prepared.history_30m,
        target_day,
        context.category_name,
        config.min_reference_group_days,
    )
    reference_5m = {day: prepared.history_5m[day] for day in reference_30m}
    target_30m = prepared.target_30m[target_day]["pRealKw"].to_numpy()
    is_solar = context.category_name.casefold() == "solar generation"
    solar_active = (
        infer_solar_active_mask(
            {day: frame["pRealKw"].to_numpy() for day, frame in reference_30m.items()}
        )
        if is_solar
        else np.ones(len(target_30m), dtype=bool)
    )

    profiles: dict[str, ProfileResult] = {}
    point_percentiles: dict[str, np.ndarray] = {}
    for name, window in (
        ("short", config.short_window_points),
        ("medium", config.medium_window_points),
    ):
        arrays = {day: frame["pRealKw"].to_numpy() for day, frame in reference_30m.items()}
        calibration = frozen_calibration(arrays, window, name)
        profile = matrix_profile_ab_join(target_30m, arrays, window)
        profiles[name] = profile
        window_percentiles = empirical_percentiles(profile.scores, calibration)
        point_percentiles[name] = project_window_scores(window_percentiles, window)
        point_percentiles[name][~solar_active] = 0.0

    daily_arrays = {day: frame["pRealKw"].to_numpy() for day, frame in reference_30m.items()}
    daily_target = target_30m
    daily_window = config.daily_window_points
    if is_solar:
        if solar_active.sum() < config.medium_window_points:
            raise ValueError("Solar channel has too few active daylight slots")
        daily_arrays = {day: values[solar_active] for day, values in daily_arrays.items()}
        daily_target = target_30m[solar_active]
        daily_window = int(solar_active.sum())
    daily_calibration = frozen_calibration(daily_arrays, daily_window, "daily")
    daily_profile = matrix_profile_ab_join(
        daily_target,
        daily_arrays,
        daily_window,
    )
    daily_score = float(daily_profile.scores[0])

    target_5m = prepared.target_5m[target_day]
    magnitude = same_slot_percentiles(
        {day: frame["pRealKw"] for day, frame in reference_5m.items()},
        target_5m["pRealKw"],
    )
    metric_percentiles: dict[str, pd.Series] = {}
    for metric in ("iRMSMax",):
        if metric in target_5m and all(metric in frame for frame in reference_5m.values()):
            metric_percentiles[metric] = same_slot_percentiles(
                {day: frame[metric] for day, frame in reference_5m.items()},
                target_5m[metric],
            )

    return DayScores(
        day=target_day,
        short_percentiles=point_percentiles["short"],
        medium_percentiles=point_percentiles["medium"],
        daily_score=daily_score,
        daily_outlier=_daily_outlier(daily_score, daily_calibration),
        magnitude_percentiles=magnitude,
        metric_percentiles=metric_percentiles,
        change_points=detect_persistent_change_points(target_5m["pRealKw"]),
        nearest_neighbors={
            "short": _neighbor_evidence(profiles["short"]),
            "medium": _neighbor_evidence(profiles["medium"]),
            "daily": _neighbor_evidence(daily_profile),
        },
    )
