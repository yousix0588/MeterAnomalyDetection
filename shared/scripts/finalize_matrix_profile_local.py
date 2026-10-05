"""Rebuild derived outputs from a fully completed local detection run.

Does not rerun detection, change input data, or edit raw shard files. Records
the detection signature separately from the current finalization code hash.
"""
import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time

import run_matrix_profile_local as local
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/matrix_profile/results/september_fixed_july_local")
    args = parser.parse_args()
    output = local.resolve(args.run_dir)
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Older interrupted metadata omitted the prepared-input path until completion.
    inputs = Path(metadata.get("prepared_input_dir", local.ROOT / "data/processed/matrix_profile" / output.name))
    keys = sorted(pd.read_csv(inputs / "_manifest.csv").meter_channel.astype(str))
    if len(keys) != metadata["channels"]:
        raise ValueError("Manifest channel count differs from detection metadata")
    marker = json.loads((inputs / "input_signature.json").read_text())
    if marker["signature"] != metadata["signature"]:
        raise ValueError("Prepared input provenance does not match detection metadata")
    shards = output / "shards"
    for position, key in enumerate(keys):
        part = shards / f"part-{position:03d}"
        completed = json.loads((part / "complete.json").read_text())
        if completed != dict(signature=metadata["signature"], series_id=key):
            raise ValueError(f"Incomplete or incompatible channel: {key}")
        if not all((part / filename).exists() for filename in local.SHARD_FILES):
            raise ValueError(f"Missing shard file: {key}")
    config = local.DetectionConfig(
        baseline_start=date.fromisoformat(metadata["baseline_start"]),
        baseline_end=date.fromisoformat(metadata["baseline_end"]),
        detection_start=date.fromisoformat(metadata["detection_start"]),
        detection_end=date.fromisoformat(metadata["detection_end_exclusive"])-timedelta(days=1),
        timezone=metadata["timezone"],
    )
    started = time.monotonic()
    raw_event_count = sum(len(pd.read_csv(shards/f"part-{i:03d}/events.csv")) for i in range(len(keys)))
    local.aggregate_shards(shards, output, len(keys))
    counts = local.export_aligned(shards, keys, output, config)
    statuses = pd.read_csv(output / "ALL_CHANNEL_STATUS.csv")
    daily = pd.read_csv(output / "ALL_DAY_STATUS.csv")
    metadata.update(counts, status="complete", prepared_input_dir=str(inputs),
                    channel_status_counts=statuses.status.value_counts().to_dict(),
                    day_status_counts=daily.status.value_counts().to_dict(),
                    reportable_events=len(pd.read_csv(output / "ALL_EVENTS.csv")),
                    candidate_events=len(pd.read_csv(output / "ALL_CANDIDATES.csv")),
                    finalization_seconds=round(time.monotonic()-started, 1))
    metadata["coalesced_reportable_duplicates"] = raw_event_count - metadata["reportable_events"]
    metadata["detection_signature"] = metadata["signature"]
    metadata["finalization_code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in (
        Path(__file__), Path(local.__file__), local.MP_SRC/"anomalies/multiscale/aggregate.py",
        local.ROOT/"shared/scripts/build_ensemble_features.py"))).hexdigest()
    metadata["finalization_notes"] = "Reused original complete detection shards; identical refined event intervals coalesced with both records retained in evidence. No score recomputation."
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
