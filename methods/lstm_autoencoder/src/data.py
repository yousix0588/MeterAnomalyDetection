from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset


FILE_RE = re.compile(r"^(?P<meter>.+)_(?P<channel>\d+)$")


@dataclass(frozen=True)
class WindowMeta:
    series_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def discover_files(root: str | Path, extensions: Sequence[str]) -> list[Path]:
    root = Path(root)
    allowed = {e.lower() for e in extensions}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed and not p.name.startswith("_")
    )


def parse_series_id(path: Path) -> str:
    match = FILE_RE.match(path.stem)
    if not match:
        return path.stem
    return f"{match.group('meter')}_{match.group('channel')}"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_series(
    root: str | Path,
    extensions: Sequence[str],
    timestamp_column: str,
    features: Sequence[str],
    meter_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[Path]] = {}
    for path in discover_files(root, extensions):
        grouped.setdefault(parse_series_id(path), []).append(path)
    if meter_limit is not None:
        grouped = dict(list(sorted(grouped.items()))[:meter_limit])

    result: dict[str, pd.DataFrame] = {}
    required = {timestamp_column, *features}
    for series_id, paths in sorted(grouped.items()):
        frames = []
        for path in paths:
            frame = read_table(path)
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
            frames.append(frame[[timestamp_column, *features]])
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
        frame = frame.dropna(subset=[timestamp_column]).drop_duplicates(timestamp_column, keep="last")
        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        result[series_id] = frame.set_index(timestamp_column).sort_index()
    return result


def clean_series(
    frame: pd.DataFrame,
    frequency: str,
    max_interpolation_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regular = frame.resample(frequency).mean()
    observed = regular.notna()
    clean = regular.interpolate(method="time", limit=max_interpolation_gap, limit_direction="both")
    return clean, observed


def chronological_split(
    frame: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:]


def chronological_date_split(
    frame: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
    timezone: str,
    validation_ratio: float,
    min_baseline_rows: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on local-time boundaries and keep August out of train/validation."""

    def boundary(value: str) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone)
        if frame.index.tz is None:
            return timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.tz_convert(frame.index.tz)

    train_end_ts = boundary(train_end)
    test_start_ts = boundary(test_start)
    test_end_ts = boundary(test_end)
    if train_end_ts != test_start_ts:
        raise ValueError("train_end and test_start must be the same boundary")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    baseline = frame.loc[frame.index < train_end_ts]
    test = frame.loc[(frame.index >= test_start_ts) & (frame.index < test_end_ts)]
    min_required = max(2, min_baseline_rows)
    if len(baseline) < min_required:
        raise ValueError(
            f"Not enough pre-test data for train/validation splitting: got {len(baseline)} rows, minimum required is {min_required}"
        )
    if test.empty:
        raise ValueError("No data in the configured test period")

    validation_rows = max(1, int(round(len(baseline) * validation_ratio)))
    validation_rows = min(validation_rows, len(baseline) - 1)
    return baseline.iloc[:-validation_rows], baseline.iloc[-validation_rows:], test


def fit_global_scaler(frames: Sequence[pd.DataFrame], max_rows: int = 1_000_000) -> RobustScaler:
    per_frame = max(1, max_rows // max(1, len(frames)))
    available = []
    for frame in frames:
        complete = frame.dropna()
        if complete.empty:
            continue
        if len(complete) > per_frame:
            positions = np.linspace(0, len(complete) - 1, per_frame, dtype=int)
            complete = complete.iloc[positions]
        available.append(complete.to_numpy(dtype=np.float32))
    if not available:
        raise ValueError("训练区间没有可用于拟合缩放器的完整数据。")
    return RobustScaler().fit(np.concatenate(available, axis=0))


def iter_windows(
    series_id: str,
    frame: pd.DataFrame,
    scaler: RobustScaler,
    window_size: int,
    stride: int,
    max_missing_ratio: float,
) -> Iterator[tuple[np.ndarray, WindowMeta]]:
    values = frame.to_numpy(dtype=np.float32)
    valid_rows = np.isfinite(values).all(axis=1)
    for start in range(0, len(frame) - window_size + 1, stride):
        stop = start + window_size
        if 1.0 - valid_rows[start:stop].mean() > max_missing_ratio:
            continue
        chunk = pd.DataFrame(values[start:stop]).interpolate(limit_direction="both").ffill().bfill()
        if chunk.isna().any().any():
            continue
        scaled = scaler.transform(chunk.to_numpy()).astype(np.float32)
        meta = WindowMeta(series_id, frame.index[start], frame.index[stop - 1])
        yield scaled, meta


class MeterWindowDataset(Dataset):
    def __init__(self, frames: dict[str, np.ndarray | torch.Tensor], scaler: RobustScaler,
                 indices: list[tuple[str, int, int]], metadata: list[WindowMeta]):
        self.frames = frames
        self.scaler = scaler
        self.indices = indices
        self.metadata = metadata

    @classmethod
    def from_frames(
        cls,
        frames: dict[str, pd.DataFrame],
        scaler: RobustScaler,
        window_size: int,
        stride: int,
        max_missing_ratio: float,
    ) -> "MeterWindowDataset":
        arrays: dict[str, np.ndarray | torch.Tensor] = {}
        indices: list[tuple[str, int, int]] = []
        metadata: list[WindowMeta] = []
        for series_id, frame in frames.items():
            values = frame.to_numpy(dtype=np.float32)
            valid_rows = np.isfinite(values).all(axis=1)
            # Complete series can be scaled once without changing window values.
            # Keep per-window interpolation for incomplete series so future rows
            # outside a window cannot alter its score.
            if valid_rows.all():
                arrays[series_id] = torch.from_numpy(
                    scaler.transform(values).astype(np.float32)
                )
            else:
                arrays[series_id] = values
            for start in range(0, len(frame) - window_size + 1, stride):
                stop = start + window_size
                if 1.0 - valid_rows[start:stop].mean() > max_missing_ratio:
                    continue
                indices.append((series_id, start, stop))
                metadata.append(WindowMeta(series_id, frame.index[start], frame.index[stop - 1]))
        if not indices:
            raise ValueError("没有生成有效窗口；请检查数据、window_size 或缺失率设置。")
        return cls(arrays, scaler, indices, metadata)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        series_id, start, stop = self.indices[index]
        series = self.frames[series_id]
        if isinstance(series, torch.Tensor):
            return series[start:stop]
        chunk = pd.DataFrame(series[start:stop]).interpolate(
            limit_direction="both"
        ).ffill().bfill()
        scaled = self.scaler.transform(chunk.to_numpy()).astype(np.float32)
        return torch.from_numpy(scaled)
