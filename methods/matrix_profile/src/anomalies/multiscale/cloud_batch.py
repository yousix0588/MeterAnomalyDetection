from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path

import pandas as pd

from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.detector import MultiscaleAnomalyDetector
from anomalies.multiscale.models import (
    ChannelContext,
    ChannelDetectionResult,
    ChannelStatus,
    DayDetectionStatus,
)


@dataclass(frozen=True)
class MergedChannelInput:
    context: ChannelContext
    path: Path


def _context_from_metadata(
    metadata: dict,
    device_id: str,
    channel_idx: int,
) -> ChannelContext | None:
    meter = metadata.get(device_id)
    if meter is None or channel_idx >= len(meter.get("channels", [])):
        return None
    channel = meter["channels"][channel_idx]
    category_name = str(channel.get("category_name", ""))
    if not category_name:
        return None
    rated_power = channel.get("metadata", {}).get("rated_power")
    return ChannelContext(
        device_id=device_id,
        channel_idx=channel_idx,
        organization=str(meter.get("organization", "")),
        category_id=int(channel.get("category_id", -1)),
        category_name=category_name,
        rated_power_kw=float(rated_power) if rated_power is not None else None,
    )


def discover_merged_channels(
    data_root: str | Path,
    manifest_path: str | Path,
    repository_path: str | Path,
    selected_channel_keys: set[str] | None = None,
    meter_ids: set[str] | None = None,
    categories: set[int] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
) -> tuple[list[MergedChannelInput], list[ChannelDetectionResult]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")
    root = Path(data_root)
    manifest = pd.read_csv(manifest_path)
    if "meter_channel" not in manifest.columns:
        raise ValueError("Manifest must contain a meter_channel column")
    metadata = json.loads(Path(repository_path).read_text(encoding="utf-8"))
    manifest_keys = sorted(set(manifest["meter_channel"].dropna().astype(str)))

    inputs: list[MergedChannelInput] = []
    failures: list[ChannelDetectionResult] = []
    eligible_position = 0
    for key in manifest_keys:
        if selected_channel_keys is not None and key not in selected_channel_keys:
            continue
        try:
            device_id, channel_text = key.rsplit("_", 1)
            channel_idx = int(channel_text)
        except ValueError:
            assigned = eligible_position % shard_count == shard_index
            eligible_position += 1
            if assigned:
                context = ChannelContext(key, -1, "", -1, "")
                failures.append(
                    ChannelDetectionResult(
                        context=context,
                        status=ChannelStatus.METADATA_MISSING,
                        reason="Manifest meter_channel is not DEVICE_CHANNEL",
                    )
                )
            continue
        if meter_ids is not None and device_id not in meter_ids:
            continue
        context = _context_from_metadata(metadata, device_id, channel_idx)
        if categories is not None and (
            context is None or context.category_id not in categories
        ):
            continue
        assigned = eligible_position % shard_count == shard_index
        eligible_position += 1
        if not assigned:
            continue
        if context is None:
            missing_context = ChannelContext(device_id, channel_idx, "", -1, "")
            failures.append(
                ChannelDetectionResult(
                    context=missing_context,
                    status=ChannelStatus.METADATA_MISSING,
                    reason="Channel metadata missing",
                )
            )
            continue
        path = root / f"{key}.csv"
        if not path.exists():
            failures.append(
                ChannelDetectionResult(
                    context=context,
                    status=ChannelStatus.PROCESSING_ERROR,
                    reason=f"Merged channel file missing: {path}",
                )
            )
            continue
        inputs.append(MergedChannelInput(context, path))
        if limit is not None and len(inputs) >= limit:
            break
    return inputs, failures


def read_merged_channel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    columns = list(pd.read_csv(path, nrows=0).columns)
    if "timestamp" not in columns or "pRealKw" not in columns:
        raise ValueError(f"{path.name} must contain timestamp and pRealKw")
    usecols = [column for column in ("timestamp", "pRealKw", "iRMSMax") if column in columns]
    return pd.read_csv(path, usecols=usecols)


def _detect_merged_worker(
    item: MergedChannelInput,
    config: DetectionConfig,
) -> ChannelDetectionResult:
    try:
        frame = read_merged_channel(item.path)
        return MultiscaleAnomalyDetector(config).detect_channel(frame, item.context)
    except Exception as error:
        return ChannelDetectionResult(
            context=item.context,
            status=ChannelStatus.PROCESSING_ERROR,
            reason=f"{type(error).__name__}: {error}",
        )


def detect_merged_batch(
    inputs: list[MergedChannelInput],
    config: DetectionConfig,
    workers: int = 1,
) -> list[ChannelDetectionResult]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return [_detect_merged_worker(item, config) for item in inputs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_detect_merged_worker, inputs, [config] * len(inputs)))


def populate_missing_day_statuses(
    results: list[ChannelDetectionResult],
    config: DetectionConfig,
) -> None:
    for result in results:
        if result.day_statuses:
            continue
        target_day = config.detection_start
        baseline_days = (config.baseline_end - config.baseline_start).days + 1
        while target_day <= config.detection_end:
            event_on_day = any(
                event.start_time.date() == target_day
                for event in (result.events + result.candidate_events)
            )
            if result.status in {ChannelStatus.DETECTED, ChannelStatus.NO_EVENT}:
                day_status = ChannelStatus.DETECTED if event_on_day else ChannelStatus.NO_EVENT
            else:
                day_status = result.status
            result.day_statuses.append(
                DayDetectionStatus(
                    date=target_day,
                    status=day_status,
                    valid_history_days=(
                        config.rolling_history_days
                        if config.history_mode == "rolling"
                        else baseline_days
                    ),
                    reason=result.reason,
                )
            )
            target_day += timedelta(days=1)
