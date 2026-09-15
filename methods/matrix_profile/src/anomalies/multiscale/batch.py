from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.models import (
    ChannelContext,
    ChannelDetectionResult,
    ChannelStatus,
    EventType,
)


@dataclass
class ChannelInput:
    context: ChannelContext
    frame: pd.DataFrame


def load_local_channels(
    data_root: str | Path,
    repository_path: str | Path = "meter_repository.json",
    meter_ids: set[str] | None = None,
    categories: set[int] | None = None,
    limit: int | None = None,
) -> tuple[list[ChannelInput], list[str]]:
    root = Path(data_root)
    metadata = json.loads(Path(repository_path).read_text(encoding="utf-8"))
    files_by_channel: dict[tuple[str, int], list[Path]] = {}
    skipped: list[str] = []

    for path in sorted(root.glob("*/*.csv")):
        stem = path.stem
        try:
            device_id, channel_text = stem.rsplit("_", 1)
            channel_idx = int(channel_text)
        except ValueError:
            skipped.append(f"metadata_missing:{path}")
            continue
        if meter_ids is not None and device_id not in meter_ids:
            continue
        files_by_channel.setdefault((device_id, channel_idx), []).append(path)

    loaded: list[ChannelInput] = []
    for (device_id, channel_idx), paths in sorted(files_by_channel.items()):
        meter = metadata.get(device_id)
        if meter is None or channel_idx >= len(meter.get("channels", [])):
            skipped.append(f"metadata_missing:{device_id}_{channel_idx}")
            continue
        channel = meter["channels"][channel_idx]
        category_id = int(channel.get("category_id", -1))
        category_name = str(channel.get("category_name", ""))
        if not category_name:
            skipped.append(f"metadata_missing:{device_id}_{channel_idx}")
            continue
        if categories is not None and category_id not in categories:
            continue

        frames: list[pd.DataFrame] = []
        for path in paths:
            frame = pd.read_csv(path, parse_dates=["timestamp"])
            frames.append(frame.set_index("timestamp"))
        combined = pd.concat(frames).sort_index()
        rated_power = channel.get("metadata", {}).get("rated_power")
        context = ChannelContext(
            device_id=device_id,
            channel_idx=channel_idx,
            organization=str(meter.get("organization", "")),
            category_id=category_id,
            category_name=category_name,
            rated_power_kw=float(rated_power) if rated_power is not None else None,
        )
        loaded.append(ChannelInput(context, combined))
        if limit is not None and len(loaded) >= limit:
            break
    return loaded, skipped


def mark_common_mode(
    results: list[ChannelDetectionResult],
    config: DetectionConfig,
) -> None:
    eligible_counts: dict[tuple[str, int], int] = {}
    all_events = []
    for result in results:
        if result.status in {ChannelStatus.DETECTED, ChannelStatus.NO_EVENT}:
            key = (result.context.organization, result.context.category_id)
            eligible_counts[key] = eligible_counts.get(key, 0) + 1
        all_events.extend(result.events)

    for event in all_events:
        key = (event.organization, event.category_id)
        denominator = eligible_counts.get(key, 0)
        if denominator == 0:
            continue
        channels = {
            (candidate.device_id, candidate.channel_idx)
            for candidate in all_events
            if candidate.organization == event.organization
            and candidate.category_id == event.category_id
            and candidate.start_time < event.end_time
            and candidate.end_time > event.start_time
        }
        count = len(channels)
        if count >= config.common_mode_min_channels and count / denominator >= config.common_mode_ratio:
            event.is_common_mode = True
            event.common_mode_channel_count = count
            if event.event_type is not EventType.DATA_QUALITY:
                event.event_type = EventType.SITE_OR_CATEGORY_EVENT


def detect_batch(
    inputs: list[ChannelInput],
    detector,
) -> list[ChannelDetectionResult]:
    results: list[ChannelDetectionResult] = []
    for item in inputs:
        try:
            results.append(detector.detect_channel(item.frame, item.context))
        except Exception as error:
            results.append(
                ChannelDetectionResult(
                    context=item.context,
                    status=ChannelStatus.PROCESSING_ERROR,
                    reason=f"{type(error).__name__}: {error}",
                )
            )
    return results
