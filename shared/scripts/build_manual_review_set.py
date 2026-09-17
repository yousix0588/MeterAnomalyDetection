#!/usr/bin/env python3
"""Extract a small, reproducible manual-review set from August ensemble results.

The output is deliberately a review sample, not a set of ground-truth labels.
Every case has an aligned 15-minute model/raw timeline and the original 5-minute
measurements around its anchor. Missing model results remain missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODELS = ("lstm_autoencoder", "lstm_vae", "matrix_profile", "rpca")
REVIEW_CATEGORIES = (
    "four_model_consensus",
    "three_model_consensus",
    "matrix_profile_only",
    "long_event",
)
RAW_METRICS = ("pRealKw", "iRMSMax", "vRMSMax", "powerFactor")
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
AUGUST_START = datetime.fromisoformat("2026-08-01 00:00:00+10:00")
SEPTEMBER_START = datetime.fromisoformat("2026-09-01 00:00:00+10:00")


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def formatted(value: datetime) -> str:
    return value.isoformat(sep=" ")


def stable_number(*values: str) -> int:
    data = "|".join(values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def score_groups(path: Path):
    """Yield one model-row dictionary per (series, interval) in sorted CSV order."""
    previous = None
    group: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = (row["series_id"], row["interval_start"])
            if previous is not None and key < previous:
                raise ValueError(f"Score file is not sorted by series/time: {key}")
            if previous is not None and key != previous:
                yield previous, group
                group = {}
            model = row["model"]
            if model not in MODELS:
                raise ValueError(f"Unknown model: {model}")
            if model in group:
                raise ValueError(f"Duplicate score row: {key}, {model}")
            group[model] = row
            previous = key
    if previous is not None:
        yield previous, group


def candidate(series_id: str, interval: str, rows: dict, category: str) -> dict:
    available = {model for model, row in rows.items() if is_true(row["available"])}
    flagged = {model for model, row in rows.items() if is_true(row["is_anomaly"])}
    percentiles = [number(row.get("max_percentile")) or 0.0 for row in rows.values()]
    return {
        "category": category,
        "series_id": series_id,
        "device_id": series_id.rsplit("_", 1)[0],
        "anchor": timestamp(interval),
        "available_models": available,
        "flagged_models": flagged,
        "priority": sum(percentiles),
        "tie_break": stable_number(series_id, interval, category),
        "source_model": "",
        "event_id": "",
        "event_start": "",
        "event_end": "",
        "event_duration_minutes": "",
    }


def sample_score_candidates(path: Path, reservoir_size: int = 4000):
    four: list[dict] = []
    three: list[dict] = []
    mp_reservoir: list[tuple[int, str, str, dict]] = []
    population = Counter()
    for (series_id, interval), rows in score_groups(path):
        available = {model for model, row in rows.items() if is_true(row["available"])}
        if len(available) != 4:
            continue
        flagged = {model for model, row in rows.items() if is_true(row["is_anomaly"])}
        if len(flagged) == 4:
            population["four_model_consensus"] += 1
            four.append(candidate(series_id, interval, rows, "four_model_consensus"))
        elif len(flagged) == 3:
            population["three_model_consensus"] += 1
            three.append(candidate(series_id, interval, rows, "three_model_consensus"))
        elif flagged == {"matrix_profile"}:
            population["matrix_profile_only"] += 1
            item = candidate(series_id, interval, rows, "matrix_profile_only")
            rank = item["tie_break"]
            entry = (-rank, series_id, interval, item)
            if len(mp_reservoir) < reservoir_size:
                heapq.heappush(mp_reservoir, entry)
            elif rank < -mp_reservoir[0][0]:
                heapq.heapreplace(mp_reservoir, entry)
    mp_only = [entry[3] for entry in mp_reservoir]
    return four, three, mp_only, population


def select_diverse(candidates: list[dict], quota: int, already: list[dict]) -> list[dict]:
    """Prefer different devices, then relax device caps without repeating a nearby case."""
    chosen: list[dict] = []
    device_counts: Counter[str] = Counter()
    for cap in (1, 2, 4, quota):
        for item in candidates:
            if len(chosen) >= quota:
                break
            if item in chosen or device_counts[item["device_id"]] >= cap:
                continue
            if any(
                other["series_id"] == item["series_id"]
                and abs(other["anchor"] - item["anchor"]) < timedelta(hours=12)
                for other in already + chosen
            ):
                continue
            chosen.append(item)
            device_counts[item["device_id"]] += 1
        if len(chosen) >= quota:
            break
    return chosen


def event_candidates(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {model: [] for model in MODELS}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            model = row["model"]
            duration = int(row["duration_minutes"])
            if model not in result or duration < 360:
                continue
            series_id = row["series_id"]
            result[model].append({
                "category": "long_event",
                "series_id": series_id,
                "device_id": series_id.rsplit("_", 1)[0],
                "anchor": timestamp(row["start"]),
                "available_models": set(),
                "flagged_models": set(),
                "priority": duration,
                "tie_break": stable_number(row["event_id"]),
                "source_model": model,
                "event_id": row["event_id"],
                "event_start": row["start"],
                "event_end": row["end"],
                "event_duration_minutes": duration,
            })
    for values in result.values():
        values.sort(key=lambda item: (-item["priority"], item["tie_break"]))
    return result


def choose_cases(scores: Path, events: Path, per_category: int,
                 long_per_model: int, context: timedelta):
    four, three, mp_only, population = sample_score_candidates(scores)
    def has_full_model_context(item: dict) -> bool:
        return (item["anchor"] - context >= AUGUST_START
                and item["anchor"] + context <= SEPTEMBER_START)

    four = [item for item in four if has_full_model_context(item)]
    three = [item for item in three if has_full_model_context(item)]
    mp_only = [item for item in mp_only if has_full_model_context(item)]
    four.sort(key=lambda item: (-item["priority"], item["tie_break"]))
    three.sort(key=lambda item: (-item["priority"], item["tie_break"]))
    mp_only.sort(key=lambda item: item["tie_break"])

    selected: list[dict] = []
    selected.extend(select_diverse(four, per_category, selected))
    three_by_pattern: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in three:
        three_by_pattern[tuple(sorted(item["flagged_models"]))].append(item)
    for pattern in sorted(three_by_pattern, key=lambda key: (len(three_by_pattern[key]), key)):
        if sum(item["category"] == "three_model_consensus" for item in selected) >= per_category:
            break
        selected.extend(select_diverse(three_by_pattern[pattern], 1, selected))
    three_count = sum(item["category"] == "three_model_consensus" for item in selected)
    if three_count < per_category:
        selected.extend(select_diverse(three, per_category - three_count, selected))

    weekend = [item for item in mp_only if item["anchor"].weekday() >= 5]
    weekday = [item for item in mp_only if item["anchor"].weekday() < 5]
    selected.extend(select_diverse(weekend, per_category // 2, selected))
    selected.extend(select_diverse(weekday, per_category - per_category // 2, selected))
    mp_count = sum(item["category"] == "matrix_profile_only" for item in selected)
    if mp_count < per_category:
        selected.extend(select_diverse(mp_only, per_category - mp_count, selected))

    long_candidates = event_candidates(events)
    for model in MODELS:
        eligible = [item for item in long_candidates[model] if has_full_model_context(item)]
        selected.extend(select_diverse(eligible, long_per_model, selected))

    for index, item in enumerate(selected, start=1):
        item["case_id"] = f"MR{index:03d}"
    return selected, population


def collect_scores(path: Path, cases: list[dict], context: timedelta):
    by_series: dict[str, list[dict]] = defaultdict(list)
    for item in cases:
        by_series[item["series_id"]].append(item)
    result: dict[str, dict[datetime, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (series_id, interval), rows in score_groups(path):
        relevant = by_series.get(series_id)
        if not relevant:
            continue
        at = timestamp(interval)
        for item in relevant:
            if item["anchor"] - context <= at < item["anchor"] + context:
                result[item["case_id"]][at] = rows
    for item in cases:
        anchor_rows = result[item["case_id"]].get(item["anchor"])
        if not anchor_rows:
            raise ValueError(f"No model scores at selected anchor: {item['case_id']}")
        item["available_models"] = {
            model for model, row in anchor_rows.items() if is_true(row["available"])
        }
        item["flagged_models"] = {
            model for model, row in anchor_rows.items() if is_true(row["is_anomaly"])
        }
        if item["category"] == "four_model_consensus" and len(item["flagged_models"]) != 4:
            raise ValueError(f"Four-model case changed: {item['case_id']}")
        if item["category"] == "three_model_consensus" and len(item["flagged_models"]) != 3:
            raise ValueError(f"Three-model case changed: {item['case_id']}")
        if item["category"] == "matrix_profile_only" and item["flagged_models"] != {"matrix_profile"}:
            raise ValueError(f"Matrix Profile-only case changed: {item['case_id']}")
        if item["category"] == "long_event" and item["source_model"] not in item["flagged_models"]:
            raise ValueError(f"Long-event start is not flagged: {item['case_id']}")
    return result


def collect_raw(raw_root: Path, cases: list[dict], context: timedelta):
    by_series: dict[str, list[dict]] = defaultdict(list)
    for item in cases:
        by_series[item["series_id"]].append(item)
    rows: list[dict] = []
    raw_fields: list[str] | None = None
    found: dict[str, bool] = {}
    for series_id, relevant in by_series.items():
        path = raw_root / f"{series_id}.csv"
        found[series_id] = path.is_file()
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
                raise ValueError(f"Raw file lacks timestamp: {path}")
            if raw_fields is None:
                raw_fields = list(reader.fieldnames)
            elif raw_fields != list(reader.fieldnames):
                raise ValueError(f"Raw column schema differs: {path}")
            for original in reader:
                at = timestamp(original["timestamp"])
                for item in relevant:
                    if item["anchor"] - context <= at < item["anchor"] + context:
                        rows.append({
                            "case_id": item["case_id"],
                            "series_id": series_id,
                            "minutes_from_anchor": int((at - item["anchor"]).total_seconds() / 60),
                            **original,
                        })
    rows.sort(key=lambda row: (row["case_id"], row["timestamp"]))
    return rows, raw_fields or ["timestamp", *RAW_METRICS], found


def raw_aggregates(rows: list[dict]):
    grouped: dict[tuple[str, datetime], list[dict]] = defaultdict(list)
    counts = Counter()
    for row in rows:
        at = timestamp(row["timestamp"])
        slot = at.replace(minute=(at.minute // 15) * 15, second=0, microsecond=0)
        grouped[(row["case_id"], slot)].append(row)
        counts[row["case_id"]] += 1
    aggregates = {}
    for key, values in grouped.items():
        record = {"raw_5min_count": len(values)}
        for metric in RAW_METRICS:
            finite = [number(row.get(metric)) for row in values]
            finite = [value for value in finite if value is not None]
            record[f"{metric}_mean"] = sum(finite) / len(finite) if finite else ""
            record[f"{metric}_min"] = min(finite) if finite else ""
            record[f"{metric}_max"] = max(finite) if finite else ""
        aggregates[key] = record
    return aggregates, counts


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(cases: list[dict], scores_by_case: dict, raw_rows: list[dict],
                  raw_fields: list[str], raw_found: dict, context: timedelta):
    aggregates, raw_counts = raw_aggregates(raw_rows)
    case_rows: list[dict] = []
    timeline_rows: list[dict] = []
    case_fields = [
        "case_id", "category", "selection_reason", "series_id", "device_id",
        "anchor_start", "context_start", "context_end", "source_model",
        "event_id", "event_start", "event_end", "event_duration_minutes",
        "available_model_count", "flagged_model_count", "flagged_models",
        "raw_source_found", "raw_5min_rows", "raw_coverage_ratio",
        "review_status", "review_label", "review_notes",
    ]
    timeline_fields = [
        "case_id", "category", "series_id", "interval_start", "interval_end",
        "minutes_from_anchor", "is_anchor_interval", "raw_5min_count",
        "raw_expected_5min_count",
    ]
    for metric in RAW_METRICS:
        timeline_fields.extend((f"{metric}_mean", f"{metric}_min", f"{metric}_max"))
    for model in MODELS:
        case_fields.extend(
            f"{model}_{field}" for field in (
                "available", "is_anomaly", "max_percentile", "mean_percentile",
                "max_score", "mean_score", "data_status", "source_resolution_minutes",
                "source_file",
            )
        )
        timeline_fields.extend(
            f"{model}_{field}" for field in (
                "available", "is_anomaly", "max_percentile", "mean_percentile",
                "max_score", "mean_score", "data_status", "source_resolution_minutes",
            )
        )

    for item in cases:
        case_id = item["case_id"]
        anchor = item["anchor"]
        start = anchor - context
        end = anchor + context
        category = item["category"]
        if category == "long_event":
            reason = f"{item['source_model']}连续标记{item['event_duration_minutes']}分钟；取事件起点"
        elif category == "matrix_profile_only":
            day_type = "周末" if anchor.weekday() >= 5 else "工作日"
            reason = f"四模型均可用，仅Matrix Profile标记；{day_type}"
        elif category == "three_model_consensus":
            reason = "四模型均可用，其中三模型同时标记"
        else:
            reason = "四模型均可用且同时标记"
        count = raw_counts[case_id]
        expected = int((2 * context).total_seconds() / 300)
        case_row = {
            "case_id": case_id,
            "category": category,
            "selection_reason": reason,
            "series_id": item["series_id"],
            "device_id": item["device_id"],
            "anchor_start": formatted(anchor),
            "context_start": formatted(start),
            "context_end": formatted(end),
            "source_model": item["source_model"],
            "event_id": item["event_id"],
            "event_start": item["event_start"],
            "event_end": item["event_end"],
            "event_duration_minutes": item["event_duration_minutes"],
            "available_model_count": len(item["available_models"]),
            "flagged_model_count": len(item["flagged_models"]),
            "flagged_models": "|".join(sorted(item["flagged_models"])),
            "raw_source_found": raw_found.get(item["series_id"], False),
            "raw_5min_rows": count,
            "raw_coverage_ratio": count / expected if expected else "",
            "review_status": "unreviewed",
            "review_label": "",
            "review_notes": "",
        }
        anchor_rows = scores_by_case[case_id][anchor]
        for model in MODELS:
            source = anchor_rows.get(model)
            for field in (
                "available", "is_anomaly", "max_percentile", "mean_percentile",
                "max_score", "mean_score", "data_status", "source_resolution_minutes",
                "source_file",
            ):
                case_row[f"{model}_{field}"] = source.get(field, "") if source else ""
        case_rows.append(case_row)

        at = start
        while at < end:
            score_rows = scores_by_case[case_id].get(at, {})
            raw = aggregates.get((case_id, at), {})
            timeline = {
                "case_id": case_id,
                "category": category,
                "series_id": item["series_id"],
                "interval_start": formatted(at),
                "interval_end": formatted(at + timedelta(minutes=15)),
                "minutes_from_anchor": int((at - anchor).total_seconds() / 60),
                "is_anchor_interval": at == anchor,
                "raw_5min_count": raw.get("raw_5min_count", 0),
                "raw_expected_5min_count": 3,
            }
            for metric in RAW_METRICS:
                for statistic in ("mean", "min", "max"):
                    field = f"{metric}_{statistic}"
                    timeline[field] = raw.get(field, "")
            for model in MODELS:
                source = score_rows.get(model)
                for field in (
                    "available", "is_anomaly", "max_percentile", "mean_percentile",
                    "max_score", "mean_score", "data_status", "source_resolution_minutes",
                ):
                    timeline[f"{model}_{field}"] = source.get(field, "") if source else ""
            timeline_rows.append(timeline)
            at += timedelta(minutes=15)
    return case_fields, case_rows, timeline_fields, timeline_rows, raw_fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--long-per-model", type=int, default=2)
    parser.add_argument("--context-hours", type=int, default=24)
    args = parser.parse_args()
    if args.per_category < 1 or args.long_per_model < 1 or args.context_hours < 1:
        parser.error("Sample counts and context hours must be positive")

    root = args.root.resolve()
    scores = root / "runs/ensemble/all_models_scores_15min.csv"
    events = root / "runs/ensemble/all_models_events.csv"
    raw_root = root / "data/processed/meter_csvs"
    output = (args.output_dir or root / "runs/ensemble/manual_review_august_v1").resolve()
    for path in (scores, events):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output contains files; choose a new --output-dir: {output}")

    context = timedelta(hours=args.context_hours)
    cases, population = choose_cases(
        scores, events, args.per_category, args.long_per_model, context
    )
    if not cases:
        raise ValueError("No review cases meet the selection rules")
    scores_by_case = collect_scores(scores, cases, context)
    raw_rows, raw_fields, raw_found = collect_raw(raw_root, cases, context)
    case_fields, case_rows, timeline_fields, timeline_rows, raw_fields = build_outputs(
        cases, scores_by_case, raw_rows, raw_fields, raw_found, context
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "review_cases.csv", case_fields, case_rows)
    write_csv(output / "review_timeline_15min.csv", timeline_fields, timeline_rows)
    write_csv(
        output / "review_raw_5min.csv",
        ["case_id", "series_id", "minutes_from_anchor", *raw_fields],
        raw_rows,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_scores": str(scores),
        "input_events": str(events),
        "input_raw_directory": str(raw_root),
        "timezone": "Australia/Sydney (timestamps preserve source UTC offset)",
        "context_hours_before_and_after": args.context_hours,
        "sample_requires_full_august_model_context": True,
        "selection": {
            "four_model_consensus": "Exactly four flags among four available models; severity and device diversity",
            "three_model_consensus": "Exactly three flags among four available models; severity and device diversity",
            "matrix_profile_only": "Only Matrix Profile flags among four available models; deterministic hash sample, half weekend",
            "long_event": "Longest derived events at least six hours; up to two per model, device diversity",
        },
        "candidate_population": dict(population),
        "selected_cases_by_category": dict(Counter(row["category"] for row in case_rows)),
        "selected_long_events_by_model": dict(Counter(
            row["source_model"] for row in case_rows if row["category"] == "long_event"
        )),
        "case_count": len(case_rows),
        "timeline_15min_rows": len(timeline_rows),
        "raw_5min_rows": len(raw_rows),
        "raw_missing_cases": [row["case_id"] for row in case_rows if not row["raw_source_found"]],
        "warning": "Model flags and derived events are candidates, not verified fault labels. Matrix Profile is 30-minute data repeated on the 15-minute grid.",
    }
    (output / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **{
        "cases": len(case_rows), "timeline_rows": len(timeline_rows),
        "raw_rows": len(raw_rows), "categories": manifest["selected_cases_by_category"],
        "raw_missing_cases": manifest["raw_missing_cases"],
    }}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
