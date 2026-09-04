import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomalies.multiscale.cloud_batch import read_merged_channel
from anomalies.multiscale.preprocessing import _normalise_index


CONFIDENCE_RANK = {"candidate": 1, "medium": 2, "high": 3, "physical": 4}


def _json_list_length(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    try:
        parsed = json.loads(str(value))
        return len(parsed) if isinstance(parsed, list) else 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0


def select_representative_events(events: pd.DataFrame, per_group: int = 3) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    ranked = events.copy()
    ranked["_confidence_rank"] = ranked["confidence"].map(CONFIDENCE_RANK).fillna(0)
    ranked["_corroboration_count"] = ranked["corroborating_metrics"].map(_json_list_length)
    score_columns = ["short_percentile", "medium_percentile", "magnitude_percentile"]
    for column in score_columns:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked["_max_percentile"] = ranked[score_columns].max(axis=1).fillna(0.0)
    ranked["duration_minutes"] = pd.to_numeric(
        ranked["duration_minutes"], errors="coerce"
    ).fillna(0)
    ranked = ranked.sort_values(
        [
            "_confidence_rank",
            "_corroboration_count",
            "_max_percentile",
            "duration_minutes",
        ],
        ascending=False,
    )
    ranked = ranked.drop_duplicates("incident_id", keep="first")
    selected = ranked.groupby(["category_name", "event_type"], group_keys=False).head(per_group)
    return selected.drop(
        columns=["_confidence_rank", "_corroboration_count", "_max_percentile"]
    )


def _plot_event(row: pd.Series, data_root: Path, output_root: Path) -> Path | None:
    channel_key = f"{row['device_id']}_{int(row['channel_idx'])}"
    path = data_root / f"{channel_key}.csv"
    if not path.exists():
        return None
    data = _normalise_index(read_merged_channel(path), "Australia/Sydney")
    start = pd.Timestamp(row["start_time"])
    if start.tzinfo is None:
        start = start.tz_localize("Australia/Sydney")
    else:
        start = start.tz_convert("Australia/Sydney")
    end = pd.Timestamp(row["end_time"])
    if end.tzinfo is None:
        end = end.tz_localize("Australia/Sydney")
    else:
        end = end.tz_convert("Australia/Sydney")
    target = data.loc[data.index.date == start.date(), "pRealKw"]
    if target.empty:
        return None

    evidence = row.get("evidence", "{}")
    try:
        evidence = json.loads(evidence) if isinstance(evidence, str) else evidence
    except json.JSONDecodeError:
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {}
    nearest = (
        evidence.get("nearest_neighbors", {})
        .get(str(row.get("dominant_scale", "")), {})
        .get("reference_day")
    )
    reference = pd.Series(dtype=float)
    if nearest:
        reference_day = pd.Timestamp(nearest).date()
        reference = data.loc[data.index.date == reference_day, "pRealKw"]

    target_x = (target.index - target.index.normalize()).total_seconds() / 3600.0
    figure, axis = plt.subplots(figsize=(11, 4.5))
    if not reference.empty:
        reference_x = (
            reference.index - reference.index.normalize()
        ).total_seconds() / 3600.0
        axis.plot(reference_x, reference.to_numpy(), label=f"nearest {nearest}", alpha=0.75)
    axis.plot(target_x, target.to_numpy(), label=f"target {start.date()}", linewidth=1.4)
    day_start = start.normalize()
    start_hour = (start - day_start).total_seconds() / 3600.0
    end_hour = (end - day_start).total_seconds() / 3600.0
    axis.axvspan(start_hour, end_hour, color="tab:red", alpha=0.2, label="event")
    axis.set(
        title=f"{channel_key} | {row['event_type']} | {row['confidence']}",
        xlabel="Local hour",
        ylabel="pRealKw (kW)",
        xlim=(0, 24),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    evidence_dir = output_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_path = evidence_dir / f"{row['event_id']}.png"
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot representative aggregated events")
    parser.add_argument("--events", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    events = pd.read_csv(args.events)
    selected = select_representative_events(events, args.per_group)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_root / "selected_events.csv", index=False)
    paths = [
        path
        for _, row in selected.iterrows()
        if (path := _plot_event(row, Path(args.data_dir), output_root)) is not None
    ]
    print(f"Selected {len(selected)} events; wrote {len(paths)} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
