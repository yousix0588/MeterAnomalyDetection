"""I/O adapter for the imported per-channel, transductive LOF detector.

The seven features, scaler, neighbours, contamination and fit_predict scoring
are unchanged. There is no historical training or frozen ECDF in this mode.
"""
from __future__ import annotations

from hashlib import sha1

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

FEATURES = ["powerFactor", "pRealKw", "pReactiveKw", "vRMSMin", "vRMSMax", "iRMSMin", "iRMSMax"]
N_NEIGHBORS = 20
CONTAMINATION = 0.005
CALIBRATION_VERSION = "lof_batch_not_frozen_v1"
EVENT_COLUMNS = [
    "model", "event_id", "series_id", "start", "end", "duration_minutes",
    "max_raw_score", "mean_raw_score", "max_score_percentile", "threshold",
    "interval_count", "calibration_version", "point_count",
    "event_time_resolution_minutes", "event_semantics", "ensemble_eligible",
]


def score_channel(frame: pd.DataFrame, series_id: str):
    """Keep original valid-row scores/labels; retain invalid rows as unavailable."""
    result = frame.copy().reset_index(drop=True)
    X = result[FEATURES].copy()
    valid_rows = X.notna().all(axis=1)
    X_valid = X.loc[valid_rows]
    result["model"] = "lof"
    result["series_id"] = series_id
    result["channel"] = series_id
    result["valid_features"] = valid_rows
    result["available"] = False
    result["lof_score"] = np.nan
    result["lof_label"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["is_anomaly"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["threshold"] = np.nan
    result["data_status"] = "missing_features"
    result["calibration_version"] = CALIBRATION_VERSION
    result["ensemble_eligible"] = False
    details = dict(valid_rows=int(valid_rows.sum()), input_rows=len(result),
                   n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION,
                   threshold=None, scaler_mean=None, scaler_scale=None)
    if len(X_valid) < N_NEIGHBORS + 1:
        result.loc[valid_rows, "data_status"] = "insufficient_channel_data"
        return result, {**details, "status": "insufficient_channel_data"}

    # This is deliberately the imported model's original computation, not
    # novelty=True, a rolling model, a refitted July model or new features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    lof = LocalOutlierFactor(n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION)
    labels = lof.fit_predict(X_scaled)
    scores = -lof.negative_outlier_factor_
    threshold = float(-lof.offset_)
    result.loc[valid_rows, "lof_score"] = scores
    result.loc[valid_rows, "lof_label"] = labels
    result.loc[valid_rows, "is_anomaly"] = labels == -1
    result.loc[valid_rows, "available"] = True
    result.loc[valid_rows, "data_status"] = "observed_batch_score"
    result["threshold"] = threshold
    return result, {**details, "status": "scored", "threshold": threshold,
                    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}


def aggregate_15min(points: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, series_id=None):
    """Full target grid; max/mean/population std over available five-minute scores."""
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="interval_start")
    valid = points.loc[points.available].copy()
    valid["interval_start"] = valid.timestamp.dt.floor("15min")
    if len(valid):
        grouped = valid.groupby("interval_start").agg(
            max_score=("lof_score", "max"), mean_score=("lof_score", "mean"),
            score_std=("lof_score", lambda values: float(np.std(values, ddof=0))),
            valid_point_count=("lof_score", "size"), point_anomaly_count=("is_anomaly", "sum"),
            threshold=("threshold", "first"))
        result = grouped.reindex(grid).reset_index()
    else:
        result = pd.DataFrame(dict(interval_start=grid))
        for column in ("max_score", "mean_score", "score_std", "valid_point_count", "point_anomaly_count", "threshold"):
            result[column] = np.nan
    result["model"] = "lof"
    if series_id is None and points.empty:
        raise ValueError("An empty channel needs an explicit series_id")
    result["series_id"] = str(series_id if series_id is not None else points.series_id.iloc[0])
    result["interval_end"] = result.interval_start + pd.Timedelta(minutes=15)
    result["valid_point_count"] = result.valid_point_count.fillna(0).astype(int)
    result["point_anomaly_count"] = result.point_anomaly_count.astype("Int64")
    result["available"] = result.valid_point_count.gt(0)
    result["coverage_ratio"] = result.valid_point_count / 3
    result["classification_complete"] = result.valid_point_count.eq(3)
    result["is_anomaly"] = result.point_anomaly_count.gt(0).astype("boolean")
    result.loc[~result.available, "is_anomaly"] = pd.NA
    result["raw_score"] = result.max_score
    result["max_percentile"] = np.nan
    result["mean_percentile"] = np.nan
    result["source_resolution_minutes"] = 5
    result["aggregation_method"] = "max_mean_std_from_5m_native_labels"
    result["data_status"] = np.select(
        [result.valid_point_count.eq(3), result.available],
        ["observed_batch_score", "partial_coverage"], default="unavailable_model_score")
    if points.data_status.eq("insufficient_channel_data").any():
        result.loc[~result.available, "data_status"] = "insufficient_channel_data"
    result["calibration_version"] = CALIBRATION_VERSION
    result["calibration_status"] = "batch_model_not_frozen"
    result["ensemble_eligible"] = False
    # Keep a native threshold visible on missing bins when the channel was fit.
    thresholds = points.threshold.dropna()
    if len(thresholds):
        result["threshold"] = float(thresholds.iloc[0])
    return result


def events_from_scores(frame: pd.DataFrame, minutes: int):
    """Native consecutive points or occupied 15-minute bins, without gap bridging."""
    native = minutes == 5
    if minutes not in (5, 15):
        raise ValueError("Events support only five-minute points or fifteen-minute bins")
    selected = frame.loc[frame.available & frame.is_anomaly.fillna(False)].copy()
    if selected.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    time_column = "timestamp" if native else "interval_start"
    selected = selected.sort_values(time_column)
    steps = selected[time_column].diff()
    selected["_event"] = (steps.isna() | steps.ne(pd.Timedelta(minutes=minutes))).cumsum()
    semantics = "consecutive_native_5m_flags" if native else "occupied_15m_bins_not_continuous_fault_duration"
    records = []
    for _, group in selected.groupby("_event"):
        start = group[time_column].iloc[0]
        end = group[time_column].iloc[-1] + pd.Timedelta(minutes=minutes)
        key = str(group.series_id.iloc[0])
        identity = f"lof|{key}|{minutes}|{start.isoformat()}|{end.isoformat()}"
        records.append(dict(
            model="lof", event_id=sha1(identity.encode()).hexdigest()[:16], series_id=key,
            start=start, end=end, duration_minutes=int((end-start).total_seconds()/60),
            max_raw_score=float(group["lof_score" if native else "max_score"].max()),
            mean_raw_score=float(group["lof_score" if native else "mean_score"].mean()),
            max_score_percentile=np.nan, threshold=float(group.threshold.iloc[0]),
            interval_count=len(group), calibration_version=CALIBRATION_VERSION,
            point_count=len(group) if native else int(group.point_anomaly_count.sum()),
            event_time_resolution_minutes=minutes, event_semantics=semantics, ensemble_eligible=False))
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def plot_native_events(events: pd.DataFrame, destination, start, end):
    """Optional replacement for the imported plotting tail's missing event file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=False)
    times = pd.to_datetime(events.start, utc=True).dt.tz_convert(start.tz)
    hourly = times.dt.hour.value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly.index, hourly.values)
    ax.set(xlabel="Event start hour", ylabel="Number of native events", title="LOF native events lasting at least 15 minutes")
    ax.set_xticks(range(24)); fig.tight_layout()
    fig.savefig(destination/"01_anomalies_by_hour.png", dpi=150); plt.close(fig)

    days = pd.date_range(start, end, freq="D", inclusive="left")
    daily = times.dt.normalize().value_counts().reindex(days, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, daily.values, marker="o")
    ax.set(xlabel="Local date", ylabel="Number of native events", title="LOF native events by day")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(destination/"02_anomalies_by_day.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(events.duration_minutes.to_numpy(dtype=float), bins=30, edgecolor="black")
    ax.set(xlabel="Native consecutive duration (minutes)", ylabel="Number of events", title="LOF native event duration")
    fig.tight_layout(); fig.savefig(destination/"03_anomaly_duration.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(events.duration_minutes, events.max_raw_score, alpha=.5)
    ax.set(xlabel="Native consecutive duration (minutes)", ylabel="Maximum LOF score", title="LOF score and native duration")
    fig.tight_layout(); fig.savefig(destination/"04_lof_vs_duration.png", dpi=150); plt.close(fig)

    values = pd.to_numeric(events.max_raw_score).groupby(times.dt.hour).mean().reindex(range(24))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(values.index, values.values)
    ax.set(xlabel="Event start hour", ylabel="Mean maximum LOF score", title="LOF native score by start hour")
    ax.set_xticks(range(24)); fig.tight_layout()
    fig.savefig(destination/"07_lof_severity_by_hour.png", dpi=150); plt.close(fig)
