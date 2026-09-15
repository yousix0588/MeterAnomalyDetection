from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass
class ProfileResult:
    scores: np.ndarray
    neighbor_days: list[date | None]
    neighbor_starts: np.ndarray


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Matrix Profile input must be one-dimensional")
    if window < 2 or window > len(array):
        raise ValueError("window must be between 2 and the sequence length")
    return np.lib.stride_tricks.sliding_window_view(array, window)


def _distance_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    window = query.shape[1]
    query_finite = np.isfinite(query).all(axis=1)
    reference_finite = np.isfinite(reference).all(axis=1)
    query_std = np.nanstd(query, axis=1)
    reference_std = np.nanstd(reference, axis=1)
    epsilon = np.finfo(float).eps * 10
    query_constant = query_finite & (query_std <= epsilon)
    reference_constant = reference_finite & (reference_std <= epsilon)
    query_variable = query_finite & ~query_constant
    reference_variable = reference_finite & ~reference_constant

    distances = np.full((len(query), len(reference)), np.inf, dtype=float)
    if query_variable.any() and reference_variable.any():
        q = query[query_variable]
        r = reference[reference_variable]
        q = (q - q.mean(axis=1, keepdims=True)) / q.std(axis=1, keepdims=True)
        r = (r - r.mean(axis=1, keepdims=True)) / r.std(axis=1, keepdims=True)
        squared = np.maximum(2.0 * window - 2.0 * (q @ r.T), 0.0)
        distances[np.ix_(query_variable, reference_variable)] = np.sqrt(squared)

    distances[np.ix_(query_constant, reference_constant)] = 0.0
    constant_distance = np.sqrt(float(window))
    distances[np.ix_(query_constant, reference_variable)] = constant_distance
    distances[np.ix_(query_variable, reference_constant)] = constant_distance
    return distances


def matrix_profile_ab_join(
    query: np.ndarray,
    reference_days: dict[date, np.ndarray],
    window: int,
) -> ProfileResult:
    if not reference_days:
        raise ValueError("At least one reference day is required")

    query_windows = _windows(np.asarray(query, dtype=float), window)
    best_scores = np.full(len(query_windows), np.inf, dtype=float)
    best_starts = np.full(len(query_windows), -1, dtype=int)
    best_days: list[date | None] = [None] * len(query_windows)

    for reference_day, reference in sorted(reference_days.items()):
        reference_windows = _windows(np.asarray(reference, dtype=float), window)
        distances = _distance_matrix(query_windows, reference_windows)
        local_starts = distances.argmin(axis=1)
        local_scores = distances[np.arange(len(query_windows)), local_starts]
        improved = local_scores < best_scores
        best_scores[improved] = local_scores[improved]
        best_starts[improved] = local_starts[improved]
        for index in np.flatnonzero(improved):
            best_days[int(index)] = reference_day

    best_scores[~np.isfinite(best_scores)] = np.nan
    return ProfileResult(best_scores, best_days, best_starts)


def calibrate_leave_one_day_out(
    history_days: dict[date, np.ndarray],
    window: int,
) -> np.ndarray:
    if len(history_days) < 2:
        raise ValueError("Leave-one-day-out calibration needs at least two days")

    calibration: list[np.ndarray] = []
    for query_day, query in sorted(history_days.items()):
        references = {day: values for day, values in history_days.items() if day != query_day}
        calibration.append(matrix_profile_ab_join(query, references, window).scores)
    return np.concatenate(calibration)


def empirical_percentiles(scores: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    finite_baseline = np.sort(np.asarray(baseline, dtype=float))
    finite_baseline = finite_baseline[np.isfinite(finite_baseline)]
    if finite_baseline.size == 0:
        raise ValueError("Baseline score distribution is empty")

    values = np.asarray(scores, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    lower = np.searchsorted(finite_baseline, values[finite], side="left")
    upper = np.searchsorted(finite_baseline, values[finite], side="right")
    result[finite] = (lower + 0.5 * (upper - lower)) / finite_baseline.size * 100.0
    return result
