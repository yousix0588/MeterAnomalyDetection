import argparse
from datetime import date
from pathlib import Path

from anomalies.multiscale.batch import detect_batch, load_local_channels, mark_common_mode
from anomalies.multiscale.cloud_batch import (
    detect_merged_batch,
    discover_merged_channels,
    populate_missing_day_statuses,
)
from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.detector import MultiscaleAnomalyDetector
from anomalies.multiscale.output import (
    write_detection_outputs,
    write_evidence_plots,
    write_shard_outputs,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run high-precision unsupervised multi-scale Matrix Profile detection",
    )
    parser.add_argument("--data-dir", default="1july-8Aug_data")
    parser.add_argument("--repository", default="meter_repository.json")
    parser.add_argument("--input-layout", choices=("daily", "merged"), default="daily")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="anomaly_detections_multiscale")
    parser.add_argument("--baseline-start", type=_date, default=date(2026, 7, 1))
    parser.add_argument("--baseline-end", type=_date, default=date(2026, 7, 30))
    parser.add_argument("--detection-start", type=_date, default=date(2026, 7, 31))
    parser.add_argument("--detection-end", type=_date, default=date(2026, 8, 6))
    parser.add_argument("--history-mode", choices=("fixed", "rolling"), default="fixed")
    parser.add_argument("--rolling-history-days", type=int, default=30)
    parser.add_argument("--meter-ids", nargs="+", default=None)
    parser.add_argument("--channel-keys", nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matching channels for a quick trial",
    )
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-mode", choices=("standard", "shard"), default="standard")
    return parser


def run(args: argparse.Namespace) -> int:
    config = DetectionConfig(
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        detection_start=args.detection_start,
        detection_end=args.detection_end,
        history_mode=args.history_mode,
        rolling_history_days=args.rolling_history_days,
    )
    output_dir = Path(args.output_dir)
    if args.input_layout == "merged":
        manifest = args.manifest or str(Path(args.data_dir) / "_manifest.csv")
        inputs, failures = discover_merged_channels(
            args.data_dir,
            manifest,
            args.repository,
            selected_channel_keys=set(args.channel_keys) if args.channel_keys else None,
            meter_ids=set(args.meter_ids) if args.meter_ids else None,
            categories=set(args.categories) if args.categories else None,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            limit=args.limit,
        )
        results = failures + detect_merged_batch(inputs, config, args.workers)
        populate_missing_day_statuses(results, config)
        if args.output_mode == "shard":
            paths = write_shard_outputs(results, output_dir)
        else:
            mark_common_mode(results, config)
            paths = write_detection_outputs(results, output_dir, args.include_candidates)
    else:
        inputs, skipped = load_local_channels(
            args.data_dir,
            args.repository,
            meter_ids=set(args.meter_ids) if args.meter_ids else None,
            categories=set(args.categories) if args.categories else None,
            limit=args.limit,
        )
        detector = MultiscaleAnomalyDetector(config)
        results = detect_batch(inputs, detector)
        mark_common_mode(results, config)
        if args.output_mode == "shard":
            paths = write_shard_outputs(results, output_dir)
        else:
            paths = write_detection_outputs(results, output_dir, args.include_candidates)
        if not args.no_plots:
            write_evidence_plots(results, inputs, output_dir)
        if skipped:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "load_skips.txt").write_text(
                "\n".join(skipped) + "\n",
                encoding="utf-8",
            )

    detected = sum(result.status.value == "detected" for result in results)
    print(f"Processed {len(results)} channels; {detected} have reportable events")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
