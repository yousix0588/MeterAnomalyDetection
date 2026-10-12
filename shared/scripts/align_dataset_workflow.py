"""Connect existing dataset preparation, canonical scores and review queues.

This is an I/O adapter, not a training or detector entry point. Original data,
model outputs, frozen ECDFs and triage snapshots are read-only. All paths are
project-relative, including the provenance written for cloud portability.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import heapq
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/data_tools"))
sys.path.insert(0, str(ROOT / "shared/scripts"))

import numpy as np
import pandas as pd
from build_meter_csvs import ArchiveReader
from meter_input import COLUMNS, index_target, normalize, prepare_inputs
import build_ensemble_features as ensemble

QUEUES = ("priority_alert", "operating_change_review", "data_quality")


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def relative(path):
    return resolve(path).relative_to(ROOT.resolve()).as_posix()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def boundary(value, timezone):
    value = pd.Timestamp(value)
    return value.tz_localize(timezone) if value.tz is None else value.tz_convert(timezone)


def load_profile(path):
    profile = json.loads(resolve(path).read_text(encoding="utf-8"))
    if profile["version"] != 1: raise ValueError("Unsupported dataset profile version")
    time = profile["time"]
    timezone = time["timezone"]
    periods = {key: boundary(time[key], timezone) for key in (
        "baseline_start_inclusive", "train_end_exclusive", "context_start_inclusive",
        "test_start_inclusive", "test_end_exclusive")}
    if not (periods["baseline_start_inclusive"] < periods["train_end_exclusive"]
            <= periods["context_start_inclusive"] < periods["test_start_inclusive"] < periods["test_end_exclusive"]):
        raise ValueError("Baseline, context-only and test roles must be disjoint and ordered")
    if time["input_interval_minutes"] != 5 or time["ensemble_interval_minutes"] != 15:
        raise ValueError("Existing contracts require five-minute input and fifteen-minute output")
    if set(profile["models"]) != ensemble.ALLOWED_MODELS:
        raise ValueError("The profile must declare all four models, using null for missing outputs")
    policy = profile["window_policy"]
    if (policy["neural_window_minutes"], policy["neural_stride_minutes"],
            policy["neural_window_points"], policy["neural_stride_points"]) != (1440, 15, 288, 3):
        raise ValueError("The established neural window is 24 hours with a 15-minute stride")
    if policy["matrix_profile_windows_minutes"] != [120, 360, 1440] or policy["matrix_profile_stride_minutes"] != 30:
        raise ValueError("Do not silently change the existing multi-scale MP detector")
    if policy["rpca_daily_points"] != 288: raise ValueError("RPCA retains its daily 288-point basis")
    if (periods["test_start_inclusive"] - periods["context_start_inclusive"]).total_seconds() < 86400:
        raise ValueError("Neural context must contain at least 24 real hours")
    return profile, periods


def verify_model_period(profile, periods):
    model = profile["models"]["matrix_profile"]
    meta = json.loads((resolve(model["run_directory"]) / "run_metadata.json").read_text())
    expected = dict(baseline_start=periods["baseline_start_inclusive"].date().isoformat(),
                    baseline_end=(periods["train_end_exclusive"] - pd.Timedelta(days=1)).date().isoformat(),
                    detection_start=periods["test_start_inclusive"].date().isoformat(),
                    detection_end_exclusive=periods["test_end_exclusive"].date().isoformat(),
                    timezone=profile["time"]["timezone"], history_mode="fixed", status="complete")
    for key, value in expected.items():
        if meta.get(key) != value: raise ValueError(f"Saved MP run disagrees with profile: {key}")
    source = resolve(profile["data"]["prepared_source_directory"])
    marker = json.loads((source / "input_signature.json").read_text())
    if marker["signature"] != meta["signature"]:
        raise ValueError("Prepared input provenance differs from the saved detector run")
    return meta


def _role_masks(frame, periods):
    ts = frame.timestamp
    baseline = ts.ge(periods["baseline_start_inclusive"]) & ts.lt(periods["train_end_exclusive"])
    context = ts.ge(periods["context_start_inclusive"]) & ts.lt(periods["test_start_inclusive"])
    test = ts.ge(periods["test_start_inclusive"]) & ts.lt(periods["test_end_exclusive"])
    if not (baseline | context | test).all():
        raise ValueError("Input contains timestamps outside the declared roles")
    return baseline, context, test


def prepare_dataset(profile, periods):
    """Reuse the completed strict input preparation; append only context-only rows."""
    source = resolve(profile["data"]["prepared_source_directory"])
    target = resolve(profile["data"]["shared_input_directory"])
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Shared dataset must be separate from the original prepared input")
    relative(target)
    injected = resolve(profile["data"]["injected_directory"])
    for candidate in (source, resolve(profile["data"]["raw_target_source"]), resolve(profile["data"]["original_history_directory"]), target):
        if injected in candidate.parents or candidate == injected:
            raise ValueError("Injected measurements cannot be the baseline, context or shared input source")
    marker = target / "dataset_metadata.json"
    if target.exists() and any(target.iterdir()):
        if not marker.exists(): raise FileExistsError("Nonempty shared input directory lacks dataset provenance")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if (metadata["dataset_id"] != profile["dataset_id"] or metadata["time"] != profile["time"]
                or metadata["window_policy"] != profile["window_policy"]
                or metadata["prepared_source_directory"] != relative(source)):
            raise ValueError("Existing shared input has a different dataset/period; use a new profile path")
        for key, sha in metadata["prepared_input_sha256"].items():
            if digest(source / f"{key}.csv") != sha: raise ValueError("Original prepared inputs changed")
        for key, sha in metadata["shared_input_sha256"].items():
            if digest(target / f"{key}.csv") != sha: raise ValueError("Shared inputs changed after preparation")
        for path, sha in metadata["context_source_sha256"].items():
            if digest(resolve(path)) != sha: raise ValueError("Context-only source changed after preparation")
        if digest(target/"_manifest.csv") != metadata["manifest_sha256"]:
            raise ValueError("Shared dataset manifest changed after preparation")
        print(f"Reuse verified shared input: {relative(target)}", flush=True)
        return metadata
    if not source.exists():
        # Fresh datasets reuse the same original ArchiveReader/strict preparation.
        history = resolve(profile["data"]["original_history_directory"])
        config = SimpleNamespace(timezone=profile["time"]["timezone"],
            baseline_start=periods["baseline_start_inclusive"].date(),
            baseline_end=(periods["train_end_exclusive"]-pd.Timedelta(days=1)).date(),
            detection_start=periods["test_start_inclusive"].date(),
            detection_end=(periods["test_end_exclusive"]-pd.Timedelta(days=1)).date())
        with ArchiveReader(str(resolve(profile["data"]["raw_target_source"]))) as reader:
            indexed = index_target(reader, config.detection_start, config.detection_end+timedelta(days=1))
            keys = sorted(set(pd.read_csv(history/"_manifest.csv").meter_channel) | set(indexed))
            prepare_inputs(reader, indexed, keys, history, source, config)
        # Preparation alone does not manufacture a completed model run/signature.
    manifest = pd.read_csv(source / "_manifest.csv")
    keys = sorted(manifest.meter_channel.astype(str))
    if len(keys) != len(set(keys)): raise ValueError("Duplicate channels in prepared manifest")
    target.mkdir(parents=True, exist_ok=True)
    rows, prepared_hashes, shared_hashes, context_hashes = [], {}, {}, {}
    raw = resolve(profile["data"]["raw_target_source"])
    with ArchiveReader(str(raw)) as reader:
        context_index = index_target(reader, periods["context_start_inclusive"].date(), periods["test_start_inclusive"].date())
        for position, key in enumerate(keys, 1):
            input_path = source / f"{key}.csv"
            prepared_hashes[key] = digest(input_path)
            prepared = normalize(pd.read_csv(input_path), profile["time"]["timezone"])
            baseline, existing_context, test = _role_masks(prepared, periods)
            if existing_context.any(): raise ValueError("Saved MP preparation must not contain target-only context")
            context_frames = []
            for day, name in context_index.get(key, []):
                with reader.open(name) as handle:
                    frame = normalize(pd.read_csv(handle), profile["time"]["timezone"])
                if not frame.timestamp.dt.date.eq(day).all(): raise ValueError("Context date-folder mismatch")
                context_frames.append(frame)
                if reader.kind == "dir": context_hashes[relative(raw/name)] = digest(raw/name)
            context = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame(columns=COLUMNS)
            combined = normalize(pd.concat([prepared, context], ignore_index=True), profile["time"]["timezone"])
            b, c, t = _role_masks(combined, periods)
            # Existing July and September values must survive exactly; context
            # is neither historical ECDF data nor a target observation.
            pd.testing.assert_frame_equal(combined.loc[~c].reset_index(drop=True), prepared.reset_index(drop=True))
            combined.to_csv(target / f"{key}.csv", index=False)
            shared_hashes[key] = digest(target / f"{key}.csv")
            context_expected = len(pd.date_range(periods["context_start_inclusive"], periods["test_start_inclusive"], freq="5min", inclusive="left"))
            test_expected = len(pd.date_range(periods["test_start_inclusive"], periods["test_end_exclusive"], freq="5min", inclusive="left"))
            rows.append(dict(meter_channel=key, n_rows=len(combined), baseline_rows=int(b.sum()),
                             context_only_rows=int(c.sum()), test_rows=int(t.sum()),
                             baseline_days_present=combined.loc[b,"timestamp"].dt.date.nunique(),
                             target_days_present=combined.loc[t,"timestamp"].dt.date.nunique(),
                             context_missing_5m_points=context_expected-int(c.sum()),
                             target_missing_5m_points=test_expected-int(t.sum()),
                             context_has_24h_observed_power=int(combined.loc[c,"pRealKw"].notna().sum())==context_expected,
                             baseline_missing_power_points=int(combined.loc[b,"pRealKw"].isna().sum()),
                             target_missing_power_points=int(combined.loc[t,"pRealKw"].isna().sum())))
            if position % 40 == 0 or position == len(keys): print(f"Shared input {position}/{len(keys)}", flush=True)
    if not raw.is_dir(): context_hashes[relative(raw)] = digest(raw)
    pd.DataFrame(rows).to_csv(target / "_manifest.csv", index=False)
    metadata = dict(dataset_id=profile["dataset_id"], time=profile["time"], schema_columns=COLUMNS,
                    window_policy=profile["window_policy"], channels=len(keys),
                    prepared_source_directory=relative(source), original_training_history_directory=relative(profile["data"]["original_history_directory"]),
                    prepared_input_sha256=prepared_hashes, shared_input_sha256=shared_hashes, context_source_sha256=context_hashes,
                    source_roles=dict(baseline="frozen July reference; never refit from target/context",
                                      context="window-only measurements; excluded from training/calibration and exported scores",
                                      test="target observations only", injected="separate paired evaluation, not production inputs"),
                    no_interpolation=True, signed_power_recomputed=False,
                    manifest_sha256=digest(target/"_manifest.csv"),
                    note="This derived input is not a substitute for the full original training corpus or saved scaler/ECDF.")
    metadata["dataset_signature"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (target / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def score_series(path, model, start, end, timezone, chunksize=100000):
    """Bounded-memory channel stream; source CSVs must be sorted by series_id."""
    current, parts, previous, selected_total = None, [], None, 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if ensemble.REQUIRED_COLUMNS.difference(chunk.columns): raise ValueError("Missing canonical score fields")
        if not chunk.model.eq(model).all(): raise ValueError("Configured model differs from score file")
        if not chunk.series_id.astype(str).is_monotonic_increasing:
            raise ValueError("Score input must be sorted by series_id")
        for key, group in chunk.groupby("series_id", sort=False):
            key = str(key)
            if previous is not None and key < previous: raise ValueError("Non-adjacent repeated/unsorted series")
            previous = key
            if current is not None and key != current:
                block = ensemble.validate(pd.concat(parts, ignore_index=True), timezone)
                block = block.loc[block.interval_start.ge(start) & block.interval_start.lt(end)]
                selected_total += len(block)
                if not block.empty: yield current, block
                parts = []
            current = key; parts.append(group)
    if parts:
        block = ensemble.validate(pd.concat(parts, ignore_index=True), timezone)
        block = block.loc[block.interval_start.ge(start) & block.interval_start.lt(end)]
        selected_total += len(block)
        if not block.empty: yield current, block
    if not selected_total: raise ValueError(f"Score input contains no target-period rows: {path}")


def merged_scores(inputs, start, end, timezone):
    streams = [iter(score_series(path, model, start, end, timezone)) for model, path in inputs]
    pending = []
    for number, stream in enumerate(streams):
        item = next(stream, None)
        if item is not None: heapq.heappush(pending, (item[0], number, item[1]))
    while pending:
        key = pending[0][0]; blocks = []
        while pending and pending[0][0] == key:
            _, number, block = heapq.heappop(pending); blocks.append(block)
            following = next(streams[number], None)
            if following is not None: heapq.heappush(pending, (following[0], number, following[1]))
        yield key, ensemble.validate(pd.concat(blocks, ignore_index=True), timezone)


def _point_mask(events, index, priority=False):
    result = np.zeros(len(index), dtype=bool)
    for row in events.to_dict("records"):
        intervals = json.loads(row["priority_segments"]) if priority else [dict(start_time=row["start_time"], end_time=row["end_time"])]
        for interval in intervals:
            begin, finish = pd.Timestamp(interval["start_time"]), pd.Timestamp(interval["end_time"])
            if begin.tz is None or finish.tz is None or finish <= begin: raise ValueError("Invalid source review interval")
            result |= (index < finish) & (index+pd.Timedelta(minutes=5) > begin)
    return result


def read_review_flags(path, timezone="Australia/Sydney", **kwargs):
    """Avoid object/string truthiness when nullable CSV flags are imported."""
    columns=["model_available","triage_available","source_formal_event_overlap"]+[f"{q}_overlap" for q in QUEUES]
    result=pd.read_csv(path,dtype={column:"boolean" for column in columns},**kwargs)
    def convert(frame):
        for column in ("interval_start","interval_end"):
            frame[column]=pd.to_datetime(frame[column],utc=True).dt.tz_convert(timezone)
        return frame
    return (convert(chunk) for chunk in result) if isinstance(result,pd.io.parsers.TextFileReader) else convert(result)


def validate_saved_calibration(part, model, periods):
    if model=="matrix_profile":
        expected=f"fixed_{periods['baseline_start_inclusive'].date()}_{(periods['train_end_exclusive']-pd.Timedelta(days=1)).date()}_v2"
        if not part.calibration_version.eq(expected).all(): raise ValueError("MP frozen calibration differs from profile")
    else:
        expected=f"train_pre_{periods['train_end_exclusive'].date()}_"
        if not part.calibration_version.astype(str).str.startswith(expected).all():
            raise ValueError(f"{model} frozen training cutoff differs from the dataset profile")


def review_grid(events, native_scores, start, end, timezone):
    """Three independent nullable flags: an interval is not an exclusive class."""
    index5 = pd.date_range(start, end, freq="5min", inclusive="left")
    index15 = pd.date_range(start, end, freq="15min", inclusive="left")
    if len(index5) != 3*len(index15): raise ValueError("Unsupported partial interval grid")
    available = native_scores.set_index("interval_start").available.reindex(index15).fillna(False).astype(bool)
    result = pd.DataFrame(dict(interval_start=index15, interval_end=index15+pd.Timedelta(minutes=15),
                               model="matrix_profile", model_available=available.to_numpy(), triage_available=available.to_numpy()))
    full = _point_mask(events, index5).reshape(-1,3).any(axis=1)
    result["source_formal_event_overlap"] = pd.array(full, dtype="boolean")
    for queue in QUEUES:
        points = _point_mask(events.loc[events.review_queue.eq(queue)], index5, priority=queue=="priority_alert")
        counts = points.reshape(-1,3).sum(axis=1)*5
        result[f"{queue}_overlap"] = pd.array(counts>0, dtype="boolean")
        if queue=="priority_alert": result["priority_core_minutes"] = counts.astype(float)
    result["review_state"] = "no_source_event"
    result.loc[full, "review_state"] = "source_event_noncore_context"
    for queue in ("operating_change_review", "data_quality", "priority_alert"):
        result.loc[result[f"{queue}_overlap"].fillna(False), "review_state"] = queue
    unknown = ~result.model_available
    for column in ["source_formal_event_overlap"]+[f"{queue}_overlap" for queue in QUEUES]:
        result.loc[unknown,column] = pd.NA
    result.loc[unknown,"priority_core_minutes"] = np.nan
    result.loc[unknown,"review_state"] = "unavailable_model_score"
    result["review_evidence_resolution_minutes"] = 5
    result["model_source_resolution_minutes"] = 30
    result["alignment_method"] = "any_overlap_from_real_5m_event_or_priority_core"
    return result


def load_review(profile, periods):
    directory = resolve(profile["review"]["triage_directory"])
    metadata = json.loads((directory / "triage_summary.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete": raise ValueError("Review snapshot is incomplete")
    for key, value in (("detection_start",periods["test_start_inclusive"]),
                       ("detection_end_exclusive",periods["test_end_exclusive"]),
                       ("baseline_start",periods["baseline_start_inclusive"]),
                       ("baseline_end",periods["train_end_exclusive"]-pd.Timedelta(days=1))):
        if metadata[key] != value.date().isoformat(): raise ValueError("Review snapshot period/calibration mismatch")
    for path, sha in metadata["original_production_sha256"].items():
        if digest(resolve(path.replace("\\","/"))) != sha: raise ValueError("Review snapshot belongs to different production outputs")
    if digest(resolve(profile["review"]["policy_config"])) != metadata["policy_config_sha256"]:
        raise ValueError("Review policy file differs from the executed snapshot")
    events = pd.read_csv(directory / "events_all_classified.csv")
    original = pd.read_csv(resolve(profile["models"]["matrix_profile"]["run_directory"]) / "ALL_EVENTS.csv")
    if events.event_id.duplicated().any() or len(events)!=len(original): raise ValueError("Review IDs are not lossless")
    pd.testing.assert_frame_equal(events[original.columns].set_index("event_id").sort_index(), original.set_index("event_id").sort_index())
    if not events.review_queue.isin(QUEUES).all(): raise ValueError("Unknown review queue")
    events["series_id"] = events.device_id.astype(str)+"_"+events.channel_idx.astype(int).astype(str)
    for column in ("start_time","end_time"):
        values = pd.to_datetime(events[column],utc=True).dt.tz_convert(profile["time"]["timezone"])
        if (values.lt(periods["test_start_inclusive"]) | values.gt(periods["test_end_exclusive"])).any():
            raise ValueError("Review events outside target period")
    return events, metadata


def canonical_formal_events(events, scores):
    records = []
    for row in events.to_dict("records"):
        start, end = pd.Timestamp(row["start_time"]), pd.Timestamp(row["end_time"])
        context = scores.loc[scores.interval_start.lt(end) & scores.interval_end.gt(start) & scores.available]
        records.append({**row, "model":"matrix_profile", "start":start, "end":end,
                        "max_raw_score":context.max_score.max(), "mean_raw_score":context.mean_score.mean(),
                        "max_score_percentile":context.max_percentile.max(),
                        "threshold":scores.threshold.iloc[0] if len(scores) else np.nan,
                        "interval_count":len(context), "calibration_version":scores.calibration_version.iloc[0] if len(scores) else "",
                        "event_layer":"native_formal_event_with_review_classification",
                        "evidence_resolution_minutes":5, "model_source_resolution_minutes":30})
    return pd.DataFrame(records)


def export_dataset(profile, periods, dataset, output):
    if output.exists() and any(output.iterdir()): raise FileExistsError("Use a fresh aligned output directory")
    events, review_meta = load_review(profile, periods)
    inputs = []
    for model,item in profile["models"].items():
        configured=item["score_input"]
        if configured:
            for value in configured if isinstance(configured,list) else [configured]:
                inputs.extend((model,path) for path in ensemble.discover_score_files(resolve(value)))
    if not inputs: raise ValueError("No saved canonical model outputs configured")
    for _,path in inputs:
        if path in output.parents or output==path.parent: raise ValueError("Output must not overwrite source model scores")
    protected = {relative(path):digest(path) for _,path in inputs}
    directory = resolve(profile["review"]["triage_directory"])
    for name in ("events_all_classified.csv","review_units.csv","event_memberships.csv","triage_summary.json"):
        protected[relative(directory/name)] = digest(directory/name)
    output.mkdir(parents=True,exist_ok=True); review_out=output/"review"; review_out.mkdir()
    (output/"workflow_metadata.json").write_text(json.dumps(dict(status="running",dataset_id=profile["dataset_id"])),encoding="utf-8")
    long_columns=[]
    for _,path in inputs:
        long_columns.extend(column for column in pd.read_csv(path,nrows=0).columns if column not in long_columns)
    wide_columns=["series_id","interval_start"]+[f"{model}_{metric}" for model in sorted(ensemble.ALLOWED_MODELS) for metric in (
        "max_percentile","mean_percentile","max_score","mean_score","available",
        "source_resolution_minutes","data_status","aggregation_method")]
    event_groups={key:group for key,group in events.groupby("series_id")}
    empty=events.iloc[:0]
    audit_rows=[]; exported_ids=set(); score_count=available_count=flag_count=0; first_formal=True
    for position,(key,long) in enumerate(merged_scores(inputs,periods["test_start_inclusive"],periods["test_end_exclusive"],profile["time"]["timezone"]),1):
        if key not in dataset["shared_input_sha256"]: raise ValueError(f"Scores lack a matching dataset channel: {key}")
        for model,part in long.groupby("model"):
            validate_saved_calibration(part,model,periods)
        first=position==1; mode="w" if first else "a"
        long=long.sort_values(["interval_start","model"]).reindex(columns=long_columns)
        long.to_csv(output/"all_models_scores_15min.csv",index=False,header=first,mode=mode)
        wide=ensemble.build_wide(long,periods["test_start_inclusive"],periods["test_end_exclusive"]).reindex(columns=wide_columns)
        mp=long.loc[long.model.eq("matrix_profile")].copy()
        flag_grid=review_grid(event_groups.get(key,empty),mp,periods["test_start_inclusive"],periods["test_end_exclusive"],profile["time"]["timezone"])
        flag_grid.insert(1,"series_id",key)
        flag_grid.to_csv(review_out/"interval_flags_15min.csv",index=False,header=first,mode=mode)
        annotations=flag_grid.drop(columns=["interval_end","model","series_id","model_source_resolution_minutes"])
        annotations=annotations.rename(columns={column:f"matrix_profile_{column}" for column in annotations if column!="interval_start"})
        wide=wide.merge(annotations,on="interval_start",validate="one_to_one")
        wide.to_csv(output/"ensemble_features_15min.csv",index=False,header=first,mode=mode)
        # Keep the old score-derived event contract and its semantics intact.
        ensemble.build_events(long).to_csv(output/"all_models_events.csv",index=False,header=first,mode=mode)
        summary=ensemble.build_run_summary(long)
        summary.to_csv(output/"run_summary.csv",index=False,header=first,mode=mode)
        audit_rows.extend(summary.to_dict("records"))
        formal=canonical_formal_events(event_groups.get(key,empty),mp)
        if not formal.empty:
            formal.to_csv(review_out/"model_events.csv",index=False,header=first_formal,mode="w" if first_formal else "a")
            first_formal=False; exported_ids.update(formal.event_id)
        score_count+=len(long); available_count+=int(long.available.sum()); flag_count+=int((long.available&long.is_anomaly).sum())
        if position%40==0: print(f"Canonical/review alignment {position} channels",flush=True)
    if exported_ids!=set(events.event_id): raise ValueError("Not all source formal events matched the canonical dataset")
    units=pd.read_csv(directory/"review_units.csv"); members=pd.read_csv(directory/"event_memberships.csv")
    units.insert(0,"model","matrix_profile"); units["start"]=units.start_time; units["end"]=units.end_time
    members.insert(0,"model","matrix_profile")
    if set(members.event_id)!=exported_ids or members.event_id.duplicated().any(): raise ValueError("Review membership IDs differ from exported events")
    units.to_csv(review_out/"review_units.csv",index=False); members.to_csv(review_out/"event_memberships.csv",index=False)
    readiness=[]
    summaries=pd.DataFrame(audit_rows)
    for model in sorted(ensemble.ALLOWED_MODELS):
        present=summaries.loc[summaries.model.eq(model)]
        readiness.append(dict(model=model,status="saved_scores_aligned" if len(present) else "not_run_for_this_dataset",
                              canonical_rows=int(present.intervals.sum()) if len(present) else 0,
                              available_rows=int(present.available_intervals.sum()) if len(present) else 0))
    pd.DataFrame(readiness).to_csv(output/"model_readiness.csv",index=False)
    for path,sha in protected.items():
        if digest(resolve(path))!=sha: raise RuntimeError("An original score/review file changed during alignment")
    metadata=dict(status="complete",dataset_id=profile["dataset_id"],time=profile["time"],window_policy=profile["window_policy"],
                  shared_input_directory=relative(profile["data"]["shared_input_directory"]),dataset_signature=dataset["dataset_signature"],
                  original_training_history_directory=relative(profile["data"]["original_history_directory"]),
                  score_rows=score_count,available_score_rows=available_count,threshold_flag_rows=flag_count,
                  models=readiness,source_formal_events=len(events),review_units=len(units),review_queue_counts=review_meta["queue_counts"],
                  source_sha256=protected,adapter_sha256=digest(Path(__file__)),
                  event_layers=dict(all_models_events="candidate threshold segments; not confirmed faults or priority alerts",
                                    review_model_events="lossless native formal events, canonical keys and queue classification",
                                    review_intervals="real five-minute event/core overlap, projected independently to fifteen minutes"),
                  limitations=["Missing model outputs remain unavailable, not normal", "Context-only rows never enter ECDF fitting or target score exports",
                               "The three review queues are not verified fault/normal labels", "Neural September inference still needs saved weights/scaler/ECDF and temporal-gap/context handling",
                               "Do not replace original neural training history with this July-reference derived dataset", "Injected inputs remain a separate paired evaluation"])
    (output/"workflow_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (output/"dataset_profile.json").write_text(json.dumps(profile,indent=2),encoding="utf-8")
    (output/"README.txt").write_text(
        "Dataset workflow alignment\n\nThe original four canonical CSV names and score meanings are retained.\n"
        "Review queues and real five-minute priority cores are sidecars under review/.\n"
        "The wide feature table adds nullable matrix_profile_* review annotations.\n"
        "no_source_event is not a normal label; unavailable_model_score has blank review flags.\n"
        "Other three September models have not been run; their available fields remain False.\n"
        "See workflow_metadata.json and the repository README for periods, paths and limitations.\n",encoding="utf-8")
    return metadata


def review_selection_reason(queue):
    return {"priority_alert":"Selected from the priority queue using real five-minute transition evidence; not a confirmed fault",
            "operating_change_review":"Selected from operating changes that need verification; not a normal label",
            "data_quality":"Selected from suspected flatline/data-quality events; not a confirmed power fault"}[queue]


def export_manual_review(profile, periods, dataset, output):
    """Reuse the existing manual-review collectors/output schema for new queues.

No four-model agreement is invented for a single-model September run. Queue
membership is kept distinct from raw model flags, and human labels stay blank.
"""
    import build_manual_review_set as manual
    context=manual.timedelta(hours=24)
    destination=output/"manual_review"
    if destination.exists() and any(destination.iterdir()): raise FileExistsError("Manual review output is not empty")
    workflow_meta=json.loads((output/"workflow_metadata.json").read_text())
    if (workflow_meta["status"]!="complete" or workflow_meta["dataset_signature"]!=dataset["dataset_signature"]):
        raise ValueError("Aligned output does not match the prepared dataset")
    events=pd.read_csv(output/"review/model_events.csv").set_index("event_id",drop=False)
    units=pd.read_csv(output/"review/review_units.csv")
    quota=profile["review"].get("sample_per_queue",5)
    if quota<1: raise ValueError("Manual sample quota must be positive")
    input_dir=resolve(profile["data"]["shared_input_directory"])
    cached_times={}; selected=[]; context_delta=pd.Timedelta(hours=24)
    for queue in QUEUES:
        candidates=[]
        for unit in units.loc[units.review_queue.eq(queue)].itertuples():
            for source_id in json.loads(unit.event_ids):
                source=events.loc[source_id]
                cores=json.loads(source.priority_segments)
                moment=pd.Timestamp(cores[0]["start_time"]) if queue=="priority_alert" else pd.Timestamp(source.start)
                anchor=moment.floor("15min")
                if not (periods["test_start_inclusive"]<=anchor-context_delta and anchor+context_delta<=periods["test_end_exclusive"]): continue
                key=str(source.series_id)
                if key not in cached_times:
                    timestamps=pd.read_csv(input_dir/f"{key}.csv",usecols=["timestamp"]).timestamp
                    cached_times[key]=pd.DatetimeIndex(pd.to_datetime(timestamps,utc=True))
                expected=pd.date_range(anchor-context_delta,anchor+context_delta,freq="5min",inclusive="left")
                if not expected.isin(cached_times[key]).all(): continue
                candidates.append(dict(category=queue,series_id=key,device_id=key.rsplit("_",1)[0],
                    anchor=anchor.to_pydatetime(),available_models=set(),flagged_models=set(),priority=1.,
                    tie_break=manual.stable_number(profile["dataset_id"],queue,unit.review_unit_id,source_id),
                    source_model="matrix_profile",event_id=source_id,event_start=str(source.start),event_end=str(source.end),
                    event_duration_minutes=int(source.duration_minutes),review_unit_id=unit.review_unit_id))
        candidates.sort(key=lambda row:row["tie_break"])
        selected.extend(manual.select_diverse(candidates,quota,selected))
    if not selected: raise ValueError("No complete 48-hour review contexts available")
    for number,row in enumerate(selected,1): row["case_id"]=f"SR{number:03d}"
    scores_path=output/"all_models_scores_15min.csv"
    score_rows=manual.collect_scores(scores_path,selected,context)
    raw_rows,raw_fields,raw_found=manual.collect_raw(input_dir,selected,context)
    case_fields,cases,timeline_fields,timeline,raw_fields=manual.build_outputs(
        selected,score_rows,raw_rows,raw_fields,raw_found,context)
    case_fields.extend(["review_queue","review_unit_id","dataset_id"])
    selected_by_id={row["case_id"]:row for row in selected}
    for row in cases:
        selected_case=selected_by_id[row["case_id"]]
        row.update(review_queue=selected_case["category"],review_unit_id=selected_case["review_unit_id"],dataset_id=profile["dataset_id"],
                   selection_reason=review_selection_reason(selected_case["category"]))
    destination.mkdir(parents=True)
    manual.write_csv(destination/"review_cases.csv",case_fields,cases)
    manual.write_csv(destination/"review_timeline_15min.csv",timeline_fields,timeline)
    manual.write_csv(destination/"review_raw_5min.csv",["case_id","series_id","minutes_from_anchor",*raw_fields],raw_rows)
    manifest=dict(dataset_id=profile["dataset_id"],dataset_signature=dataset["dataset_signature"],
        score_source=relative(scores_path),raw_source=relative(input_dir),context_hours_before_and_after=24,
        requested_per_queue=quota,selected_per_queue=dict(manual.Counter(row["category"] for row in cases)),
        case_count=len(cases),timeline_15min_rows=len(timeline),raw_5min_rows=len(raw_rows),
        source_tools=["shared/scripts/build_manual_review_set.py: collect_scores, collect_raw, build_outputs, select_diverse"],
        implementation_sha256=digest(Path(__file__)),
        warnings=["Queue membership is not a fault label", "Only currently available models are shown; no four-model agreement is claimed",
                  "Timeline flags are original model thresholds, not priority-core flags; queue and core information are separate",
                  "Human review_status is unreviewed and review_label/review_notes are blank",
                  "Samples require complete timestamp context; this does not certify every measurement as valid"])
    (destination/"review_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Manual review sample: {len(cases)} cases, {manifest['selected_per_queue']}",flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",default="configs/datasets/september_2026.json")
    parser.add_argument("--stage",choices=("prepare","export","review","all"),default="all")
    parser.add_argument("--output-dir",help="New aligned directory; relative to group14reshape")
    args=parser.parse_args(argv)
    profile,periods=load_profile(args.profile)
    if args.stage in ("export","review","all"): verify_model_period(profile,periods)
    dataset=prepare_dataset(profile,periods)
    if args.stage=="prepare":
        print(f"Prepared/verified {dataset['channels']} input channels; no model execution.",flush=True); return
    output=resolve(args.output_dir or profile["output"]["directory"])
    if args.stage=="review":
        export_manual_review(profile,periods,dataset,output); return
    metadata=export_dataset(profile,periods,dataset,output)
    if args.stage=="all": export_manual_review(profile,periods,dataset,output)
    print(json.dumps({key:metadata[key] for key in ("status","dataset_id","score_rows","available_score_rows","threshold_flag_rows","source_formal_events","review_units")},indent=2),flush=True)
    print(f"Aligned output: {relative(output)}",flush=True)


if __name__=="__main__": main()
