"""Opt-in amplitude event path. The production detector does not enable it.

Historical envelopes and jump guards are frozen before the target period.
Five-minute observations provide genuine point evidence independently of the
30-minute Matrix Profile score grid. Event overlap is not a fault probability.
"""
from dataclasses import dataclass
from hashlib import sha1

import numpy as np
import pandas as pd

from anomalies.multiscale.models import AnomalyEvent, Confidence, EventType
from anomalies.multiscale.scoring import select_reference_days


@dataclass(frozen=True)
class MagnitudeEventConfig:
    minimum_effect_kw: float = 0.5
    relative_effect: float = 0.05
    slot_radius_points: int = 2  # +/- 10 minutes of historical schedule jitter
    envelope_quantile: float = 0.01
    jump_quantile: float = 0.995
    minimum_reference_days: int = 6
    minimum_duration_minutes: int = 15

    def __post_init__(self):
        if self.minimum_effect_kw <= 0 or self.relative_effect <= 0:
            raise ValueError("Effect guards must be positive")
        if not 0 <= self.slot_radius_points <= 6:
            raise ValueError("Slot radius must be between zero and six points")
        if not 0 <= self.envelope_quantile < 0.5 or not 0 < self.jump_quantile <= 1:
            raise ValueError("Invalid historical quantiles")
        if self.minimum_reference_days < 2 or self.minimum_duration_minutes < 5:
            raise ValueError("Invalid support or event duration")


@dataclass(frozen=True)
class FrozenMagnitudeReference:
    dates: tuple
    lower_kw: np.ndarray
    upper_kw: np.ndarray
    median_kw: np.ndarray
    support_days: np.ndarray
    effect_kw: float
    jump_kw: float
    current_lower: np.ndarray | None
    current_upper: np.ndarray | None
    current_effect: float | None


def _envelope(matrix, radius, quantile):
    padded = np.pad(matrix, ((0, 0), (radius, radius)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=1)
    support = np.isfinite(windows).any(axis=2).sum(axis=0)
    values = windows.transpose(1, 0, 2).reshape(matrix.shape[1], -1)
    return (np.nanquantile(values, quantile, axis=1),
            np.nanquantile(values, 1 - quantile, axis=1), support)


def calibrate_magnitude_reference(prepared, target_day, context, config, policy=None):
    policy = policy or MagnitudeEventConfig()
    history = select_reference_days(prepared.history_5m, target_day,
                                    context.category_name, config.min_reference_group_days)
    if not history or max(history) >= config.detection_start:
        raise ValueError("Amplitude calibration requires history strictly before detection")
    if len(history) < policy.minimum_reference_days:
        raise ValueError("Insufficient reference support for amplitude events")
    if any(len(frame) != 288 for frame in history.values()):
        raise ValueError("Amplitude reference requires 288 five-minute slots")
    arrays = [frame.pRealKw.to_numpy(dtype=float) for _, frame in sorted(history.items())]
    matrix = np.vstack(arrays)
    finite = matrix[np.isfinite(matrix)]
    if not finite.size:
        raise ValueError("No finite historical power")
    lower, upper, support = _envelope(matrix, policy.slot_radius_points, policy.envelope_quantile)
    median = np.nanmedian(matrix, axis=0)
    effect = max(policy.minimum_effect_kw, policy.relative_effect * float(np.quantile(np.abs(finite), .95)))
    jumps = np.abs(np.diff(matrix, axis=1))
    jumps = jumps[np.isfinite(jumps)]
    jump = max(effect, float(np.quantile(jumps, policy.jump_quantile)) if jumps.size else effect)
    current_lower = current_upper = current_effect = None
    if all("iRMSMax" in frame for frame in history.values()):
        current = np.vstack([frame.iRMSMax.to_numpy(dtype=float) for _, frame in sorted(history.items())])
        valid_current = current[np.isfinite(current)]
        if valid_current.size:
            current_lower, current_upper, current_support = _envelope(
                current, policy.slot_radius_points, policy.envelope_quantile)
            current_lower[current_support < policy.minimum_reference_days] = np.nan
            current_upper[current_support < policy.minimum_reference_days] = np.nan
            current_effect = max(.5, policy.relative_effect * float(np.quantile(np.abs(valid_current), .95)))
    # No caller can accidentally adapt the reference to target observations.
    for array in (lower, upper, median, support, current_lower, current_upper):
        if array is not None:
            array.setflags(write=False)
    return FrozenMagnitudeReference(tuple(sorted(history)), lower, upper, median,
                                    support, effect, jump, current_lower, current_upper, current_effect)


def _contiguous_groups(mask, index):
    groups = []
    start = None
    for position, flagged in enumerate(mask):
        adjacent = position == 0 or index[position] - index[position - 1] == pd.Timedelta(minutes=5)
        if start is not None and (not flagged or not adjacent):
            groups.append((start, position))
            start = None
        if flagged and start is None:
            start = position
    if start is not None:
        groups.append((start, len(mask)))
    return groups


def event_point_mask(events, index):
    mask = np.zeros(len(index), dtype=bool)
    for event in events:
        mask |= (index < event.end_time) & (index + pd.Timedelta(minutes=5) > event.start_time)
    return mask


def build_magnitude_events(scores, target, reference, context, config,
                           legacy_events=(), quality_events=(), excluded_mask=None,
                           mode="transition_guarded", policy=None):
    if mode not in {"envelope_confirmed", "transition_guarded"}:
        raise ValueError("Unknown opt-in amplitude event mode")
    policy = policy or MagnitudeEventConfig()
    index = target.index
    if len(index) != 288 or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Amplitude target requires a sorted unique 288-slot day")
    if max(reference.dates) >= scores.day:
        raise ValueError("Amplitude reference must precede target day")
    power = target.pRealKw.to_numpy(dtype=float)
    excess = np.maximum(power - reference.upper_kw, reference.lower_kw - power)
    percentile = scores.magnitude_percentiles.reindex(index).to_numpy(dtype=float)
    mask = (np.isfinite(power) & np.isfinite(excess) & np.isfinite(percentile)
            & (excess > reference.effect_kw) & (percentile >= config.candidate_percentile)
            & (reference.support_days >= policy.minimum_reference_days))
    if excluded_mask is not None:
        excluded = np.asarray(excluded_mask, dtype=bool)
        if excluded.shape != mask.shape:
            raise ValueError("Quality mask shape mismatch")
        mask &= ~excluded
    mask &= ~event_point_mask(quality_events, index)
    current = np.zeros(len(index), dtype=bool)
    if reference.current_lower is not None and "iRMSMax" in target:
        values = target.iRMSMax.to_numpy(dtype=float)
        current_excess = np.maximum(values - reference.current_upper, reference.current_lower - values)
        current_pct = scores.metric_percentiles.get("iRMSMax")
        if current_pct is not None:
            current = (np.isfinite(values) & np.isfinite(current_excess)
                       & (current_excess > reference.current_effect)
                       & (current_pct.reindex(index).to_numpy(dtype=float) >= config.candidate_percentile))
    jumps = np.r_[np.nan, np.abs(np.diff(power))]
    adjacent = np.r_[False, np.diff(index.asi8) == pd.Timedelta(minutes=5).value]
    jump_mask = adjacent & np.isfinite(jumps) & (jumps > reference.jump_kw)
    already = event_point_mask(legacy_events, index)
    events = []
    for first, stop in _contiguous_groups(mask, index):
        start = index[first]
        end = index[stop - 1] + pd.Timedelta(minutes=5)
        duration = int((end - start).total_seconds() // 60)
        if duration < policy.minimum_duration_minutes or already[first:stop].all():
            continue
        # Only evidence near the START may admit a transition. A later jump
        # must not validate hours of earlier background deviation.
        jump_near = bool(jump_mask[max(0, first - 1):first + 1].any())
        change_near = any(start - pd.Timedelta(minutes=10) <= point <= start + pd.Timedelta(minutes=5)
                          for point in scores.change_points)
        bins = (np.arange(48) * 30 >= first * 5 - 30) & (np.arange(48) * 30 < stop * 5)
        short = np.asarray(scores.short_percentiles, dtype=float)[bins]
        medium = np.asarray(scores.medium_percentiles, dtype=float)[bins]
        shape = bool(np.any(short > config.joint_percentile) or np.any(medium > config.joint_percentile))
        current_fraction = float(current[first:stop].mean())
        peak_excess = float(np.max(excess[first:stop]))
        if mode == "envelope_confirmed":
            admit = current_fraction >= .5 or jump_near or change_near
        else:
            admit = (peak_excess > 2 * reference.effect_kw
                     and (jump_near or change_near or (shape and current_fraction >= .5)))
        if not admit:
            continue
        raw_id = f"{mode}|{context.device_id}|{context.channel_idx}|{start.isoformat()}|{end.isoformat()}"
        observed = power[first:stop]
        baseline = float(np.nanmedian(reference.median_kw[first:stop]))
        change = abs(float(observed.mean()) - baseline)
        corroborating = ["pRealKw"]
        if current_fraction >= .5:
            corroborating.append("iRMSMax")
        if jump_near:
            corroborating.append("local_jump")
        if change_near:
            corroborating.append("change_point")
        events.append(AnomalyEvent(
            event_id=sha1(raw_id.encode()).hexdigest()[:16], device_id=context.device_id,
            channel_idx=context.channel_idx, organization=context.organization,
            category_id=context.category_id, category_name=context.category_name,
            start_time=start, end_time=end, duration_minutes=duration,
            confidence=Confidence.MEDIUM, event_type=EventType.MAGNITUDE_ANOMALY,
            dominant_scale="point_5m", dominant_metric="pRealKw",
            short_percentile=float(np.nanmax(short)) if np.isfinite(short).any() else None,
            medium_percentile=float(np.nanmax(medium)) if np.isfinite(medium).any() else None,
            daily_score=scores.daily_score, magnitude_percentile=float(np.max(percentile[first:stop])),
            change_point_confirmed=change_near, corroborating_metrics=corroborating,
            baseline_median_kw=baseline, observed_mean_kw=float(observed.mean()),
            observed_max_kw=float(observed.max()), absolute_change_kw=change,
            rated_power_ratio=change / context.rated_power_kw if context.rated_power_kw else None,
            evidence=dict(event_policy=mode, supplemental=True, source_resolution_minutes=5,
                          reference_end=str(max(reference.dates)), reference_days=len(reference.dates),
                          effect_guard_kw=reference.effect_kw, jump_guard_kw=reference.jump_kw,
                          peak_outside_envelope_kw=peak_excess, current_confirmation_fraction=current_fraction,
                          new_5m_points=int((~already[first:stop]).sum()),
                          historical_slot_radius_minutes=policy.slot_radius_points * 5)))
    return events
