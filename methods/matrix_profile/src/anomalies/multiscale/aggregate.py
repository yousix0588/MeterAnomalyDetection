from hashlib import sha1
import json
from pathlib import Path

import pandas as pd


SHARD_FILES = (
    "events.csv", "candidates.csv", "channel_status.csv", "day_status.csv",
    "scores_30min.csv", "scores_15min.csv",
)
ELIGIBLE_DAY_STATUSES = {"detected", "no_event"}
CONFIDENCE_RANK = {"candidate": 1, "medium": 2, "high": 3, "physical": 4}


def _read_shard_frames(
    shards_root: Path,
    expected_shards: int,
    filename: str,
) -> list[pd.DataFrame]:
    frames = []
    for index in range(expected_shards):
        path = shards_root / f"part-{index:03d}" / filename
        if not path.exists():
            raise FileNotFoundError(f"Required shard output missing: {path}")
        frames.append(pd.read_csv(path))
    return frames


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    return frames[0].iloc[0:0].copy() if frames else pd.DataFrame()


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.empty:
        return
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head(5).to_dict("records")
        raise ValueError(f"Duplicate {label}: {examples}")


def _coalesce_refined_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate coarse candidates can refine to the same physical interval.

    Merge only matching identities/types/confidence, retaining BOTH original
    records as evidence. Conflicting IDs still fail the usual integrity check.
    Raw shard files remain untouched.
    """
    if frame.empty or not frame.event_id.duplicated().any():
        return frame
    rows = []
    identity = ["device_id", "channel_idx", "start_time", "end_time", "event_type", "confidence"]
    for event_id, group in frame.groupby("event_id", sort=False):
        row = group.iloc[0].copy()
        if len(group) > 1:
            if len(group[identity].drop_duplicates()) != 1:
                raise ValueError(f"Conflicting duplicate event id: {event_id}")
            evidence = json.loads(row["evidence"])
            evidence["refined_candidate_records"] = json.loads(group.to_json(orient="records"))
            row["evidence"] = json.dumps(evidence, sort_keys=True)
            for column in ("short_percentile", "medium_percentile", "magnitude_percentile"):
                row[column] = pd.to_numeric(group[column], errors="raise").max()
            row["dominant_scale"] = "short" if row["short_percentile"] >= row["medium_percentile"] else "medium"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _mark_common_mode(events: pd.DataFrame, day_status: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        result = events.copy()
        result["incident_id"] = pd.Series(dtype=str)
        return result
    result = events.copy()
    result["event_date"] = result["start_time"].astype(str).str.slice(0, 10)
    result["_start"] = pd.to_datetime(result["start_time"], utc=True)
    result["_end"] = pd.to_datetime(result["end_time"], utc=True)
    result["is_common_mode"] = False
    result["common_mode_channel_count"] = 0
    result["incident_id"] = ""

    eligible = day_status[day_status["status"].isin(ELIGIBLE_DAY_STATUSES)].copy()
    denominators = (
        eligible.groupby(["organization", "category_id", "date"], dropna=False)
        .apply(lambda group: group[["device_id", "channel_idx"]].drop_duplicates().shape[0])
        .to_dict()
    )

    group_columns = ["organization", "category_id", "event_date"]
    for group_key, group in result.groupby(group_columns, dropna=False):
        denominator = int(denominators.get(group_key, 0))
        if denominator <= 0:
            continue
        for row_index, row in group.iterrows():
            overlapping = group[(group["_start"] < row["_end"]) & (group["_end"] > row["_start"])]
            count = overlapping[["device_id", "channel_idx"]].drop_duplicates().shape[0]
            if count >= 3 and count / denominator >= 0.30:
                result.at[row_index, "is_common_mode"] = True
                result.at[row_index, "common_mode_channel_count"] = count
                if result.at[row_index, "event_type"] != "data_quality":
                    result.at[row_index, "event_type"] = "site_or_category_event"

    for group_key, group in result.groupby(group_columns, dropna=False):
        common = group[group["is_common_mode"]].sort_values("_start")
        clusters: list[list[int]] = []
        cluster: list[int] = []
        cluster_end = None
        for row_index, row in common.iterrows():
            if cluster and row["_start"] >= cluster_end:
                clusters.append(cluster)
                cluster = []
                cluster_end = None
            cluster.append(row_index)
            cluster_end = row["_end"] if cluster_end is None else max(cluster_end, row["_end"])
        if cluster:
            clusters.append(cluster)
        for indices in clusters:
            starts = result.loc[indices, "_start"]
            ends = result.loc[indices, "_end"]
            raw = "|".join(map(str, group_key)) + f"|{starts.min()}|{ends.max()}"
            incident_id = sha1(raw.encode("utf-8")).hexdigest()[:16]
            result.loc[indices, "incident_id"] = incident_id

    no_incident = result["incident_id"] == ""
    result.loc[no_incident, "incident_id"] = result.loc[no_incident, "event_id"].map(
        lambda value: f"event-{value}"
    )
    return result.drop(columns=["_start", "_end"])


def _incident_summary(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "incident_id",
        "date",
        "organization",
        "category_id",
        "category_name",
        "start_time",
        "end_time",
        "event_types",
        "confidence",
        "channel_count",
        "event_count",
        "is_common_mode",
        "event_ids",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for incident_id, group in events.groupby("incident_id", sort=True):
        confidence = max(
            group["confidence"].astype(str), key=lambda value: CONFIDENCE_RANK.get(value, 0)
        )
        records.append(
            {
                "incident_id": incident_id,
                "date": str(group["event_date"].iloc[0]),
                "organization": group["organization"].iloc[0],
                "category_id": group["category_id"].iloc[0],
                "category_name": group["category_name"].iloc[0],
                "start_time": pd.to_datetime(group["start_time"], utc=True).min().isoformat(),
                "end_time": pd.to_datetime(group["end_time"], utc=True).max().isoformat(),
                "event_types": json.dumps(sorted(set(group["event_type"].astype(str)))),
                "confidence": confidence,
                "channel_count": group[["device_id", "channel_idx"]].drop_duplicates().shape[0],
                "event_count": len(group),
                "is_common_mode": bool(group["is_common_mode"].any()),
                "event_ids": json.dumps(sorted(group["event_id"].astype(str))),
            }
        )
    return pd.DataFrame(records, columns=columns)


def aggregate_shards(
    shards_root: str | Path,
    output_root: str | Path,
    expected_shards: int,
) -> dict[str, Path]:
    shards_root = Path(shards_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for index in range(expected_shards):
        shard = shards_root / f"part-{index:03d}"
        for filename in SHARD_FILES:
            if not (shard / filename).exists():
                raise FileNotFoundError(f"Required shard output missing: {shard / filename}")

    events = _concat(_read_shard_frames(shards_root, expected_shards, "events.csv"))
    candidates = _concat(_read_shard_frames(shards_root, expected_shards, "candidates.csv"))
    channel_status = _concat(
        _read_shard_frames(shards_root, expected_shards, "channel_status.csv")
    )
    day_status = _concat(_read_shard_frames(shards_root, expected_shards, "day_status.csv"))
    scores_30min = _concat(_read_shard_frames(shards_root, expected_shards, "scores_30min.csv"))
    scores_15min = _concat(_read_shard_frames(shards_root, expected_shards, "scores_15min.csv"))

    events = _coalesce_refined_events(events)
    candidates = _coalesce_refined_events(candidates)
    # Channel counts must describe the coalesced output, not coarse duplicates.
    for frame, column in ((events, "event_count"), (candidates, "candidate_count")):
        counts = frame.groupby(["device_id", "channel_idx"]).size().to_dict()
        channel_status[column] = [int(counts.get((row.device_id, row.channel_idx), 0))
                                  for row in channel_status.itertuples()]
    _assert_unique(events, ["event_id"], "reportable event ids")
    _assert_unique(candidates, ["event_id"], "candidate event ids")
    _assert_unique(channel_status, ["device_id", "channel_idx"], "channel statuses")
    _assert_unique(day_status, ["device_id", "channel_idx", "date"], "day statuses")
    _assert_unique(scores_30min, ["series_id", "interval_start"], "30-minute scores")
    _assert_unique(scores_15min, ["series_id", "interval_start"], "15-minute scores")

    events = _mark_common_mode(events, day_status)
    incidents = _incident_summary(events)
    if events.empty:
        daily_summary = pd.DataFrame(
            columns=["date", "category_name", "confidence", "event_type", "event_count"]
        )
    else:
        daily_summary = (
            events.groupby(
                ["event_date", "category_name", "confidence", "event_type"], as_index=False
            )
            .size()
            .rename(columns={"event_date": "date", "size": "event_count"})
        )

    paths = {
        "events": output_root / "ALL_EVENTS.csv",
        "candidates": output_root / "ALL_CANDIDATES.csv",
        "channel_status": output_root / "ALL_CHANNEL_STATUS.csv",
        "day_status": output_root / "ALL_DAY_STATUS.csv",
        "daily_summary": output_root / "daily_summary.csv",
        "incident_summary": output_root / "incident_summary.csv",
        "scores_30min": output_root / "ALL_SCORES_30MIN.csv",
        "scores_15min": output_root / "ALL_SCORES_15MIN.csv",
    }
    events.to_csv(paths["events"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    channel_status.to_csv(paths["channel_status"], index=False)
    day_status.to_csv(paths["day_status"], index=False)
    daily_summary.to_csv(paths["daily_summary"], index=False)
    incidents.to_csv(paths["incident_summary"], index=False)
    scores_30min.to_csv(paths["scores_30min"], index=False)
    scores_15min.to_csv(paths["scores_15min"], index=False)
    return paths
