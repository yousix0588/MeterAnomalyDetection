from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd


class ChannelStatus(str, Enum):
    READY = "ready"
    DETECTED = "detected"
    NO_EVENT = "no_event"
    INSUFFICIENT_HISTORY = "insufficient_history"
    MISSING_TARGET = "missing_target"
    LOW_ACTIVITY = "low_activity"
    CONSTANT_SIGNAL = "constant_signal"
    METADATA_MISSING = "metadata_missing"
    CALIBRATION_FAILED = "calibration_failed"
    PROCESSING_ERROR = "processing_error"
    DST_TRANSITION = "dst_transition"


class Confidence(str, Enum):
    CANDIDATE = "candidate"
    MEDIUM = "medium"
    HIGH = "high"
    PHYSICAL = "physical"


class EventType(str, Enum):
    SHAPE_CHANGE = "shape_change"
    SUSTAINED_DEVIATION = "sustained_deviation"
    DAILY_PATTERN_CHANGE = "daily_pattern_change"
    MAGNITUDE_ANOMALY = "magnitude_anomaly"
    STATE_CHANGE = "state_change"
    SITE_OR_CATEGORY_EVENT = "site_or_category_event"
    DATA_QUALITY = "data_quality"


@dataclass(frozen=True)
class ChannelContext:
    device_id: str
    channel_idx: int
    organization: str
    category_id: int
    category_name: str
    rated_power_kw: float | None = None


@dataclass
class PreparedChannel:
    history_5m: dict[date, pd.DataFrame]
    target_5m: dict[date, pd.DataFrame]
    history_30m: dict[date, pd.DataFrame]
    target_30m: dict[date, pd.DataFrame]
    invalid_history_dates: list[date] = field(default_factory=list)
    invalid_target_dates: list[date] = field(default_factory=list)


@dataclass
class RollingPreparedChannel:
    days_5m: dict[date, pd.DataFrame]
    days_30m: dict[date, pd.DataFrame]
    invalid_dates: dict[date, str] = field(default_factory=dict)
    dst_dates: set[date] = field(default_factory=set)


@dataclass
class DayDetectionStatus:
    date: date
    status: ChannelStatus
    valid_history_days: int
    reason: str = ""


@dataclass
class DayScores:
    day: date
    short_percentiles: Any
    medium_percentiles: Any
    daily_score: float
    daily_outlier: bool
    magnitude_percentiles: pd.Series
    metric_percentiles: dict[str, pd.Series]
    change_points: list[pd.Timestamp]
    nearest_neighbors: dict[str, Any]


@dataclass
class PreparationResult:
    status: ChannelStatus
    reason: str
    data: PreparedChannel | None = None


@dataclass
class AnomalyEvent:
    event_id: str
    device_id: str
    channel_idx: int
    organization: str
    category_id: int
    category_name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: int
    confidence: Confidence
    event_type: EventType
    dominant_scale: str
    dominant_metric: str
    short_percentile: float | None = None
    medium_percentile: float | None = None
    daily_score: float | None = None
    magnitude_percentile: float | None = None
    change_point_confirmed: bool = False
    corroborating_metrics: list[str] = field(default_factory=list)
    baseline_median_kw: float | None = None
    observed_mean_kw: float | None = None
    observed_max_kw: float | None = None
    absolute_change_kw: float | None = None
    rated_power_ratio: float | None = None
    is_common_mode: bool = False
    common_mode_channel_count: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        result["confidence"] = self.confidence.value
        result["event_type"] = self.event_type.value
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat()
        return result


@dataclass
class ChannelDetectionResult:
    context: ChannelContext
    status: ChannelStatus
    events: list[AnomalyEvent] = field(default_factory=list)
    candidate_events: list[AnomalyEvent] = field(default_factory=list)
    day_statuses: list[DayDetectionStatus] = field(default_factory=list)
    interval_scores: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
