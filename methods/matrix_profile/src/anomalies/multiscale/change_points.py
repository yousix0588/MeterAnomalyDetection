import numpy as np
import pandas as pd
import ruptures as rpt


def detect_persistent_change_points(
    series: pd.Series,
    minimum_segment_points: int = 6,
) -> list[pd.Timestamp]:
    clean = series.astype(float).interpolate(method="time", limit_direction="both")
    if len(clean) < minimum_segment_points * 2 or clean.isna().any():
        return []
    values = clean.to_numpy()
    variance = max(float(np.var(values)), 1e-6)
    penalty = 3.0 * np.log(len(values)) * variance
    breakpoints = rpt.Pelt(
        model="l2",
        min_size=minimum_segment_points,
        jump=1,
    ).fit(values).predict(pen=penalty)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    minimum_shift = max(3.0 * 1.4826 * mad, 0.1)
    confirmed: list[pd.Timestamp] = []
    for breakpoint in breakpoints[:-1]:
        before = values[breakpoint - minimum_segment_points : breakpoint]
        after = values[breakpoint : breakpoint + minimum_segment_points]
        if len(after) < minimum_segment_points:
            continue
        if abs(float(np.median(after)) - float(np.median(before))) >= minimum_shift:
            confirmed.append(clean.index[breakpoint])
    return confirmed
