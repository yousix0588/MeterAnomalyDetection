from datetime import date, timedelta
from math import ceil

import numpy as np
import pandas as pd

from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.models import (
    ChannelStatus,
    PreparationResult,
    PreparedChannel,
    RollingPreparedChannel,
)


POWER_COLUMN = "pRealKw"
FIVE_MINUTE_POINTS_PER_DAY = 288
THIRTY_MINUTE_POINTS_PER_DAY = 48


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _normalise_index(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    data = frame.copy()
    if "timestamp" in data.columns:
        raw_index = data.pop("timestamp")
    else:
        raw_index = data.index

    if isinstance(raw_index, pd.DatetimeIndex):
        parsed = raw_index
    elif isinstance(getattr(raw_index, "dtype", None), pd.DatetimeTZDtype):
        parsed = pd.DatetimeIndex(raw_index)
    elif pd.api.types.is_datetime64_dtype(getattr(raw_index, "dtype", None)):
        parsed = pd.DatetimeIndex(raw_index)
    else:
        non_null = pd.Series(raw_index).dropna()
        sample = str(non_null.iloc[0]) if not non_null.empty else ""
        has_offset = sample.endswith("Z") or bool(
            pd.Series([sample]).str.contains(r"[+-]\d{2}:?\d{2}$", regex=True).iloc[0]
        )
        parsed = pd.DatetimeIndex(
            pd.to_datetime(raw_index, errors="coerce", utc=True if has_offset else False)
        )

    if parsed.tz is None:
        parsed = parsed.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        parsed = parsed.tz_convert(timezone)
    data.index = parsed
    data = data.loc[~data.index.isna()]

    numeric = data.select_dtypes(include=[np.number]).columns
    data = data.loc[:, numeric].sort_index()
    if data.index.has_duplicates:
        data = data.groupby(level=0).mean()
    return data


def _day_frame(data: pd.DataFrame, day: date, timezone: str) -> pd.DataFrame:
    start = pd.Timestamp(day, tz=timezone)
    expected = pd.date_range(start=start, periods=FIVE_MINUTE_POINTS_PER_DAY, freq="5min")
    return data.reindex(expected)


def _calendar_day_frame(
    data: pd.DataFrame,
    day: date,
    timezone: str,
) -> tuple[pd.DataFrame, bool]:
    start = pd.Timestamp(day, tz=timezone)
    end = start + pd.DateOffset(days=1)
    expected = pd.date_range(start=start, end=end, freq="5min", inclusive="left")
    frame = data.loc[(data.index >= start) & (data.index < end)].reindex(expected)
    return frame, len(expected) != FIVE_MINUTE_POINTS_PER_DAY


def _valid_day(day_frame: pd.DataFrame, config: DetectionConfig) -> bool:
    minimum = ceil(FIVE_MINUTE_POINTS_PER_DAY * config.valid_day_ratio)
    return POWER_COLUMN in day_frame and day_frame[POWER_COLUMN].notna().sum() >= minimum


def _long_gap_bins(day_frame: pd.DataFrame, config: DetectionConfig) -> set[pd.Timestamp]:
    missing = day_frame[POWER_COLUMN].isna().to_numpy()
    max_short_points = config.max_interpolation_minutes // 5
    blocked: set[pd.Timestamp] = set()
    run_start: int | None = None

    for position, is_missing in enumerate(np.append(missing, False)):
        if is_missing and run_start is None:
            run_start = position
        elif not is_missing and run_start is not None:
            if position - run_start > max_short_points:
                blocked.update(day_frame.index[run_start:position].floor("30min"))
            run_start = None
    return blocked


def _resample_day(day_frame: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    blocked = _long_gap_bins(day_frame, config)
    max_short_points = config.max_interpolation_minutes // 5
    filled = day_frame.interpolate(
        method="time",
        limit=max_short_points,
        limit_area="inside",
    )
    start = day_frame.index[0]
    expected = pd.date_range(start=start, periods=THIRTY_MINUTE_POINTS_PER_DAY, freq="30min")
    result = filled.resample("30min").mean().reindex(expected)
    if blocked:
        result.loc[result.index.intersection(blocked)] = np.nan
    return result


def _median_absolute_deviation(values: pd.Series) -> float:
    median = values.median()
    return float((values - median).abs().median())


def baseline_admission_status(
    history_5m: dict[date, pd.DataFrame],
    config: DetectionConfig,
    category_name: str,
) -> tuple[ChannelStatus, str]:
    baseline_power = pd.concat([day[POWER_COLUMN] for day in history_5m.values()]).dropna()
    if float(baseline_power.abs().quantile(0.95)) < config.power_cutoff_kw:
        return (
            ChannelStatus.LOW_ACTIVITY,
            "Baseline absolute p95 power is below the admission cutoff",
        )
    variability_power = baseline_power
    if category_name.casefold() == "solar generation":
        active_cutoff = max(
            0.05,
            float(baseline_power.abs().quantile(0.95)) * 0.05,
        )
        variability_power = baseline_power[baseline_power.abs() > active_cutoff]
    if (
        variability_power.empty
        or variability_power.nunique() <= 1
        or _median_absolute_deviation(variability_power) < config.constant_mad_kw
    ):
        return (
            ChannelStatus.CONSTANT_SIGNAL,
            "Baseline signal is constant or has near-zero MAD",
        )
    return ChannelStatus.READY, "Channel admitted"


def prepare_channel(
    frame: pd.DataFrame,
    config: DetectionConfig,
    category_name: str,
) -> PreparationResult:
    if not category_name:
        return PreparationResult(ChannelStatus.METADATA_MISSING, "Category metadata missing")
    if POWER_COLUMN not in frame.columns:
        return PreparationResult(ChannelStatus.PROCESSING_ERROR, "pRealKw column missing")

    data = _normalise_index(frame, config.timezone)
    history_dates = _date_range(config.baseline_start, config.baseline_end)
    target_dates = _date_range(config.detection_start, config.detection_end)

    history_all = {day: _day_frame(data, day, config.timezone) for day in history_dates}
    target_all = {day: _day_frame(data, day, config.timezone) for day in target_dates}
    history_5m = {day: value for day, value in history_all.items() if _valid_day(value, config)}
    target_5m = {day: value for day, value in target_all.items() if _valid_day(value, config)}
    invalid_history = [day for day in history_dates if day not in history_5m]
    invalid_target = [day for day in target_dates if day not in target_5m]

    if len(history_5m) < config.min_valid_history_days:
        return PreparationResult(
            ChannelStatus.INSUFFICIENT_HISTORY,
            f"Only {len(history_5m)} valid baseline days; {config.min_valid_history_days} required",
        )
    if not target_5m:
        return PreparationResult(ChannelStatus.MISSING_TARGET, "No valid target days")

    admission_status, admission_reason = baseline_admission_status(
        history_5m, config, category_name
    )
    if admission_status is not ChannelStatus.READY:
        return PreparationResult(admission_status, admission_reason)

    prepared = PreparedChannel(
        history_5m=history_5m,
        target_5m=target_5m,
        history_30m={day: _resample_day(value, config) for day, value in history_5m.items()},
        target_30m={day: _resample_day(value, config) for day, value in target_5m.items()},
        invalid_history_dates=invalid_history,
        invalid_target_dates=invalid_target,
    )
    return PreparationResult(ChannelStatus.READY, "Channel admitted", prepared)


def prepare_rolling_channel(
    frame: pd.DataFrame,
    config: DetectionConfig,
    category_name: str,
) -> tuple[ChannelStatus, str, RollingPreparedChannel | None]:
    if not category_name:
        return ChannelStatus.METADATA_MISSING, "Category metadata missing", None
    if POWER_COLUMN not in frame.columns:
        return ChannelStatus.PROCESSING_ERROR, "pRealKw column missing", None

    data = _normalise_index(frame, config.timezone)
    first_day = config.detection_start - timedelta(days=config.rolling_history_days)
    all_dates = _date_range(first_day, config.detection_end)
    days_5m: dict[date, pd.DataFrame] = {}
    days_30m: dict[date, pd.DataFrame] = {}
    invalid_dates: dict[date, str] = {}
    dst_dates: set[date] = set()

    for day in all_dates:
        day_frame, is_dst = _calendar_day_frame(data, day, config.timezone)
        if is_dst:
            dst_dates.add(day)
            invalid_dates[day] = "Local day does not contain 288 five-minute slots"
            continue
        if not _valid_day(day_frame, config):
            invalid_dates[day] = "Day contains fewer than 95% valid pRealKw points"
            continue
        days_5m[day] = day_frame
        days_30m[day] = _resample_day(day_frame, config)

    if not days_5m:
        return ChannelStatus.MISSING_TARGET, "No valid rolling data days", None
    return (
        ChannelStatus.READY,
        "Channel prepared for rolling detection",
        RollingPreparedChannel(
            days_5m=days_5m,
            days_30m=days_30m,
            invalid_dates=invalid_dates,
            dst_dates=dst_dates,
        ),
    )
