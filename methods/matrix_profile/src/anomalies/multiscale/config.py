from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class DetectionConfig:
    baseline_start: date = date(2026, 7, 1)
    baseline_end: date = date(2026, 7, 31)
    detection_start: date = date(2026, 8, 1)
    detection_end: date = date(2026, 8, 31)
    history_mode: Literal["fixed", "rolling"] = "fixed"
    rolling_history_days: int = 30
    timezone: str = "Australia/Sydney"
    min_valid_history_days: int = 24
    min_reference_group_days: int = 6
    valid_day_ratio: float = 0.95
    power_cutoff_kw: float = 0.5
    constant_mad_kw: float = 0.01
    max_interpolation_minutes: int = 30
    short_window_points: int = 4
    medium_window_points: int = 12
    daily_window_points: int = 48
    candidate_percentile: float = 99.5
    joint_percentile: float = 99.0
    boundary_low_percentile: float = 97.5
    minimum_event_minutes: int = 15
    merge_gap_minutes: int = 30
    common_mode_min_channels: int = 3
    common_mode_ratio: float = 0.30
    ev_min_sessions: int = 5

    def __post_init__(self) -> None:
        if self.history_mode == "fixed" and self.baseline_end >= self.detection_start:
            raise ValueError("Baseline must end before the detection period starts")
        if self.baseline_start > self.baseline_end:
            raise ValueError("baseline_start must not be after baseline_end")
        if self.detection_start > self.detection_end:
            raise ValueError("detection_start must not be after detection_end")
        if (
            self.history_mode == "rolling"
            and self.rolling_history_days < self.min_valid_history_days
        ):
            raise ValueError("rolling_history_days must cover min_valid_history_days")
