from dataclasses import dataclass, field
from datetime import date

import numpy as np

from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.models import ChannelContext, DayScores, RollingPreparedChannel
from anomalies.multiscale.profile import (
    ProfileResult,
    calibrate_leave_one_day_out,
    empirical_percentiles,
    matrix_profile_ab_join,
)
from anomalies.multiscale.scoring import (
    _daily_outlier,
    _neighbor_evidence,
    infer_solar_active_mask,
    project_window_scores,
    same_slot_percentiles,
    select_reference_days,
)
from anomalies.multiscale.change_points import detect_persistent_change_points


@dataclass
class RollingProfileCache:
    arrays: dict[date, np.ndarray]
    pair_profiles: dict[tuple[int, date, date], ProfileResult] = field(default_factory=dict)

    def pair(self, query_day: date, reference_day: date, window: int) -> ProfileResult:
        key = (window, query_day, reference_day)
        cached = self.pair_profiles.get(key)
        if cached is None:
            cached = matrix_profile_ab_join(
                self.arrays[query_day],
                {reference_day: self.arrays[reference_day]},
                window,
            )
            self.pair_profiles[key] = cached
        return cached

    def join(
        self,
        query_day: date,
        reference_days: list[date],
        window: int,
    ) -> ProfileResult:
        if not reference_days:
            raise ValueError("Rolling profile needs at least one reference day")
        first = self.pair(query_day, reference_days[0], window)
        best_scores = np.full(len(first.scores), np.inf, dtype=float)
        best_starts = np.full(len(first.scores), -1, dtype=int)
        best_days: list[date | None] = [None] * len(best_scores)
        for reference_day in reference_days:
            profile = self.pair(query_day, reference_day, window)
            improved = np.isfinite(profile.scores) & (profile.scores < best_scores)
            best_scores[improved] = profile.scores[improved]
            best_starts[improved] = profile.neighbor_starts[improved]
            for position in np.flatnonzero(improved):
                best_days[int(position)] = reference_day
        best_scores[~np.isfinite(best_scores)] = np.nan
        return ProfileResult(best_scores, best_days, best_starts)

    def calibrate(self, reference_days: list[date], window: int) -> np.ndarray:
        if len(reference_days) < 2:
            raise ValueError("Rolling calibration needs at least two reference days")
        scores = []
        for query_day in reference_days:
            other_days = [day for day in reference_days if day != query_day]
            scores.append(self.join(query_day, other_days, window).scores)
        return np.concatenate(scores)


def score_rolling_day(
    prepared: RollingPreparedChannel,
    target_day: date,
    history_days: list[date],
    context: ChannelContext,
    config: DetectionConfig,
    cache: RollingProfileCache,
) -> DayScores:
    history_30m = {day: prepared.days_30m[day] for day in history_days}
    reference_30m = select_reference_days(
        history_30m,
        target_day,
        context.category_name,
        config.min_reference_group_days,
    )
    reference_days = sorted(reference_30m)
    reference_5m = {day: prepared.days_5m[day] for day in reference_days}
    target_30m = prepared.days_30m[target_day]["pRealKw"].to_numpy()
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
        calibration = cache.calibrate(reference_days, window)
        profile = cache.join(target_day, reference_days, window)
        profiles[name] = profile
        window_percentiles = empirical_percentiles(profile.scores, calibration)
        point_percentiles[name] = project_window_scores(window_percentiles, window)
        point_percentiles[name][~solar_active] = 0.0

    daily_window = config.daily_window_points
    if is_solar:
        if solar_active.sum() < config.medium_window_points:
            raise ValueError("Solar channel has too few active daylight slots")
        daily_arrays = {
            day: prepared.days_30m[day]["pRealKw"].to_numpy()[solar_active]
            for day in reference_days
        }
        daily_target = target_30m[solar_active]
        daily_window = int(solar_active.sum())
        daily_calibration = calibrate_leave_one_day_out(daily_arrays, daily_window)
        daily_profile = matrix_profile_ab_join(
            daily_target,
            daily_arrays,
            daily_window,
        )
    else:
        daily_calibration = cache.calibrate(reference_days, daily_window)
        daily_profile = cache.join(target_day, reference_days, daily_window)
    daily_score = float(daily_profile.scores[0])

    target_5m = prepared.days_5m[target_day]
    magnitude = same_slot_percentiles(
        {day: frame["pRealKw"] for day, frame in reference_5m.items()},
        target_5m["pRealKw"],
    )
    metric_percentiles = {}
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
