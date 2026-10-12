"""Create conservative review queues from saved Matrix Profile events.

No model is retrained or rescored. Production outputs are read-only. Frozen
July envelopes provide real five-minute priority evidence. Existing paired
injection results are evaluated separately, including downgraded coverage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "methods/matrix_profile/src"))

import numpy as np
import pandas as pd
from anomalies.multiscale.alert_triage import (
    AlertTriageConfig, QUEUES, _runs, build_review_units, classify_events, config_dict, interval_mask,
)
from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.magnitude_events import MagnitudeEventConfig, calibrate_magnitude_reference
from anomalies.multiscale.models import ChannelContext
from anomalies.multiscale.preprocessing import _normalise_index, prepare_channel

POLICIES = ("legacy", "transition_guarded", "envelope_confirmed")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    result = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_input(path, timezone):
    frame = pd.read_csv(path, usecols=lambda name: name in ("timestamp", "pRealKw", "iRMSMax"))
    if "timestamp" not in frame or frame.timestamp.duplicated().any():
        raise ValueError(f"Missing or duplicate timestamps: {path}")
    return _normalise_index(frame, timezone)


def context_for(events):
    row = events.iloc[0]
    return ChannelContext(str(row.device_id), int(row.channel_idx), str(row.organization),
                          int(row.category_id), str(row.category_name))


def classify_with_reference(events, frame, config, policy):
    if events.empty:
        return classify_events(events, {}, {}, policy), "no_source_events"
    context = context_for(events)
    preparation = prepare_channel(frame, config, context.category_name)
    if preparation.data is None:
        return classify_events(events, {}, {}, policy, "reference_unavailable:" + preparation.status.value), preparation.status.value
    prepared = preparation.data
    magnitude_policy = MagnitudeEventConfig(
        minimum_effect_kw=policy.minimum_effect_kw, relative_effect=policy.relative_effect,
        slot_radius_points=policy.slot_radius_points, envelope_quantile=policy.envelope_quantile,
        jump_quantile=policy.jump_quantile, minimum_reference_days=policy.minimum_reference_days,
        minimum_duration_minutes=policy.minimum_duration_minutes)
    refs, cache = {}, {}
    for day in sorted({pd.Timestamp(value).date() for value in events.start_time}):
        group = "all" if context.category_name.casefold() == "solar generation" else day.weekday() >= 5
        if group not in cache:
            # No target observations, injection labels or target-wide quantiles enter calibration.
            cache[group] = calibrate_magnitude_reference(prepared, day, context, config, magnitude_policy)
        refs[day] = cache[group]
    return classify_events(events, prepared.target_5m, refs, policy), preparation.status.value


def september_worker(task):
    key, records, columns, input_path, config, policy = task
    events = pd.DataFrame(records, columns=columns)
    path = Path(input_path) / f"{key}.csv"
    before_hash = digest(path)
    frame = load_input(path, config.timezone)
    classified, status = classify_with_reference(events, frame, config, policy)
    if digest(path) != before_hash:
        raise RuntimeError(f"Source input changed during triage: {path}")
    return classified, dict(series_id=key, preparation_status=status, input_sha256=before_hash,
                           input_file=str(path.relative_to(ROOT)))


def export_queues(classified, output, policy):
    units, membership = build_review_units(classified, policy)
    if set(membership.event_id) != set(classified.event_id) or len(membership) != len(classified):
        raise AssertionError("Triage memberships lost or duplicated source events")
    classified = classified.merge(membership[["event_id", "review_unit_id"]], on="event_id", validate="one_to_one")
    classified.to_csv(output / "events_all_classified.csv", index=False)
    units.to_csv(output / "review_units.csv", index=False)
    membership.to_csv(output / "event_memberships.csv", index=False)
    for queue in QUEUES:
        classified.loc[classified.review_queue.eq(queue)].to_csv(output / f"{queue}_events.csv", index=False)
        units.loc[units.review_queue.eq(queue)].to_csv(output / f"{queue}_units.csv", index=False)
    intervals = []
    for row in classified.loc[classified.review_queue.eq("priority_alert")].to_dict("records"):
        for core in json.loads(row["priority_segments"]):
            intervals.append(dict(event_id=row["event_id"], review_unit_id=row["review_unit_id"],
                                  model="matrix_profile", series_id=f"{row['device_id']}_{int(row['channel_idx'])}",
                                  source_resolution_minutes=5, **core))
    interval_columns = ["event_id", "review_unit_id", "model", "series_id", "source_resolution_minutes", "start_time",
                        "end_time", "duration_minutes", "peak_excess_kw", "onset_jump_kw",
                        "current_confirmation_fraction", "multiscale_agreement", "direction"]
    pd.DataFrame(intervals, columns=interval_columns).to_csv(output / "priority_intervals.csv", index=False)
    counts = pd.DataFrame([dict(review_queue=queue, source_events=int(classified.review_queue.eq(queue).sum()),
                                review_units=int(units.review_queue.eq(queue).sum())) for queue in QUEUES])
    counts.to_csv(output / "queue_counts.csv", index=False)
    by_category = classified.groupby(["organization", "category_name", "review_queue"], dropna=False).size().reset_index(name="source_events")
    by_category.to_csv(output / "queue_by_category.csv", index=False)
    return classified, units, counts


def injection_worker(task):
    key, validation_dir, policy, output_dir, expected_original_hash, expected_injected_hash = task
    source = Path(validation_dir) / "channels" / key
    original_path = ROOT / "data/processed/meter_csvs" / f"{key}.csv"
    injected_path = ROOT / "data/injected_meters" / f"{key}.csv"
    if digest(original_path) != expected_original_hash or digest(injected_path) != expected_injected_hash:
        raise ValueError(f"Injection inputs differ from the completed validation: {key}")
    original = load_input(original_path, "Australia/Sydney")
    modified = load_input(injected_path, "Australia/Sydney").reindex(original.index)
    if not np.isfinite(modified.pRealKw.to_numpy()).any():
        raise ValueError("No aligned injected power")
    known = (modified.pRealKw - original.pRealKw).abs().gt(1e-4)
    if known.loc[known.index.month == 7].any():
        raise ValueError("Known edits contaminate the historical reference")
    replay = original.copy()
    common = list(original.columns.intersection(modified.columns))
    replay.loc[known, common] = modified.loc[known, common]
    pd.testing.assert_frame_equal(replay.loc[~known], original.loc[~known])
    labels = pd.read_csv(source / "changed_points.csv")
    index = pd.DatetimeIndex(pd.to_datetime(labels.timestamp, utc=True)).tz_convert("Australia/Sydney")
    changed = labels.is_injected.to_numpy(dtype=bool)
    eligible = labels.eligible.to_numpy(dtype=bool)
    if not np.array_equal(known.reindex(index).to_numpy(dtype=bool), changed):
        raise ValueError("Saved injection labels no longer match paired power differences")
    config = DetectionConfig(detection_start=date(2026, 8, 1), detection_end=date(2026, 8, 31))
    destination = Path(output_dir) / "injection_channels" / key
    destination.mkdir(parents=True, exist_ok=True)
    summary, runs = [], []
    for event_policy in POLICIES:
        results, masks, counts = {}, {}, {}
        for version, frame in (("original", original), ("injected", replay)):
            events = pd.read_csv(source / f"{version}_{event_policy}_events.csv")
            classified, status = classify_with_reference(events, frame, config, policy)
            classified.to_csv(destination / f"{version}_{event_policy}_classified.csv", index=False)
            results[version] = classified
            units, _ = build_review_units(classified, policy)
            masks[version] = dict(all=interval_mask(classified, index),
                                  priority=interval_mask(classified, index, queue="priority_alert", priority_cores=True),
                                  quality=interval_mask(classified, index, queue="data_quality"),
                                  review=interval_mask(classified, index, queue="operating_change_review"))
            masks[version]["action"] = masks[version]["priority"] | masks[version]["quality"]
            counts[version] = dict(source_events=len(events), priority_events=int(classified.review_queue.eq("priority_alert").sum()),
                                   priority_units=int(units.review_queue.eq("priority_alert").sum()),
                                   quality_events=int(classified.review_queue.eq("data_quality").sum()),
                                   review_events=int(classified.review_queue.eq("operating_change_review").sum()))
        before, after = masks["original"], masks["injected"]
        changed_eligible = changed & eligible
        record = dict(series_id=key, event_policy=event_policy,
                      eligible_changed_5m_points=int(changed_eligible.sum()))
        for version in counts:
            record.update({f"{version}_{field}": value for field, value in counts[version].items()})
        for queue in ("all", "priority", "quality", "review", "action"):
            record[f"injected_{queue}_covered_changed_5m_points"] = int((after[queue] & changed_eligible).sum())
            record[f"original_{queue}_covered_changed_5m_points"] = int((before[queue] & changed_eligible).sum())
            record[f"new_{queue}_covered_changed_5m_points"] = int((after[queue] & ~before[queue] & changed_eligible).sum())
        record["covered_points_not_in_priority"] = int((after["all"] & ~after["priority"] & changed_eligible).sum())
        record["covered_points_not_in_priority_or_quality"] = int((after["all"] & ~after["action"] & changed_eligible).sum())
        record["new_priority_covered_unchanged_5m_points"] = int((after["priority"] & ~before["priority"] & ~changed & eligible).sum())
        summary.append(record)
        for first, stop in _runs(changed, index):
            valid = bool(eligible[first:stop].all())
            run = dict(series_id=key, event_policy=event_policy, start_time=index[first].isoformat(),
                       end_time=(index[stop-1] + pd.Timedelta(minutes=5)).isoformat(),
                       changed_5m_points=stop-first, eligible=valid)
            for queue in ("all", "priority", "quality", "review", "action"):
                run[f"injected_{queue}_overlap"] = bool(after[queue][first:stop].any()) if valid else None
                run[f"injected_{queue}_covered_points"] = int(after[queue][first:stop].sum()) if valid else None
            run["downgraded_out_of_priority_and_quality"] = bool(after["all"][first:stop].any() and not after["action"][first:stop].any()) if valid else None
            runs.append(run)
    if digest(original_path) != expected_original_hash or digest(injected_path) != expected_injected_hash:
        raise RuntimeError("Paired source files changed during triage")
    return summary, runs


def evaluate_injections(validation, output, policy, workers):
    audit = pd.read_csv(validation / "input_audit.csv")
    tasks = []
    for row in audit.itertuples():
        if (validation / "channels" / row.series_id / "changed_points.csv").exists():
            tasks.append((row.series_id, str(validation), policy, str(output), row.original_sha256, row.injected_sha256))
    summaries, segments = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(injection_worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(pending), 1):
            summary, runs = future.result()
            summaries.extend(summary); segments.extend(runs)
            print(f"Injection triage {number}/{len(tasks)}: {pending[future]}", flush=True)
    summary = pd.DataFrame(summaries)
    runs = pd.DataFrame(segments)
    summary.to_csv(output / "injection_paired_summary.csv", index=False)
    runs.to_csv(output / "injection_changed_runs.csv", index=False)
    runs.loc[runs.downgraded_out_of_priority_and_quality.eq(True)].to_csv(output / "injection_downgraded_runs.csv", index=False)
    aggregate = summary.groupby("event_policy").sum(numeric_only=True).reset_index()
    original_totals = pd.read_csv(validation / "aggregate_comparison.csv").set_index("event_policy")
    for i, row in aggregate.iterrows():
        if int(row.injected_all_covered_changed_5m_points) != int(original_totals.loc[row.event_policy, "injected_covered_changed_5m_points"]):
            raise AssertionError("Lossless source-event coverage differs from previous paired validation")
        aggregate.loc[i, "all_known_changed_5m_points"] = int(audit.known_changed_5m_points.sum())
        aggregate.loc[i, "unscored_changed_5m_points"] = int(audit.known_changed_5m_points.sum()) - row.eligible_changed_5m_points
        group = runs.loc[runs.event_policy.eq(row.event_policy) & runs.eligible.eq(True)]
        aggregate.loc[i, "eligible_changed_runs"] = len(group)
        for queue in ("all", "priority", "quality", "review", "action"):
            aggregate.loc[i, f"{queue}_coverage_percent"] = 100 * row[f"injected_{queue}_covered_changed_5m_points"] / row.eligible_changed_5m_points
            aggregate.loc[i, f"{queue}_overlapped_runs"] = int(group[f"injected_{queue}_overlap"].sum())
        aggregate.loc[i, "downgraded_runs"] = int(group.downgraded_out_of_priority_and_quality.sum())
    aggregate.to_csv(output / "injection_triage_comparison.csv", index=False)
    audit.to_csv(output / "injection_input_audit.csv", index=False)
    return aggregate, dict(paired_channels=len(tasks), unscored_channels=len(audit)-len(tasks),
                           all_known_changed_5m_points=int(audit.known_changed_5m_points.sum()))


def render_report(output, counts, units, classified, injection, metadata):
    names = dict(priority_alert="优先告警", operating_change_review="运行变化待核验", data_quality="数据质量问题")
    table = counts.copy()
    table.review_queue = table.review_queue.map(names)
    table.columns = ["队列", "保留的原始事件", "合并后的核验单元"]
    priority_categories = units.loc[units.review_queue.eq("priority_alert")].groupby("category_name").size().rename_axis("优先队列类别").reset_index(name="核验单元")
    before = metadata["source_event_count"]
    priority = int(counts.loc[counts.review_queue.eq("priority_alert"), "review_units"].iloc[0])
    quality = int(counts.loc[counts.review_queue.eq("data_quality"), "review_units"].iloc[0])
    comparison = injection[["event_policy", "injected_source_events", "injected_priority_events", "injected_quality_events",
                            "all_coverage_percent", "priority_coverage_percent", "action_coverage_percent",
                            "all_overlapped_runs", "action_overlapped_runs", "eligible_changed_runs", "downgraded_runs"]].copy()
    comparison.event_policy = comparison.event_policy.map(dict(legacy="原路径", transition_guarded="补漏路径：转变限制", envelope_confirmed="补漏路径：较宽松"))
    comparison.columns = ["来源路径", "原事件", "优先事件", "质量事件", "全部保留覆盖 (%)", "仅优先核心覆盖 (%)",
                          "优先核心＋质量覆盖 (%)", "全部覆盖区段", "优先＋质量区段", "有效改变区段", "仅留待核验区段"]
    reasons = classified.loc[classified.review_queue.eq("operating_change_review"), "triage_reasons"].map(json.loads).explode().value_counts().rename_axis("未进入优先队列的原因").reset_index(name="事件数（可重复计数）")
    top = classified.loc[classified.review_queue.eq("priority_alert"), ["device_id", "channel_idx", "event_id", "priority_core_minutes", "peak_outside_envelope_kw", "priority_segments"]].head(15)
    top.columns = ["设备", "通道", "原事件ID", "优先核心分钟", "峰值超历史范围(kW)", "真实5分钟告警核心"]
    policy = metadata["policy"]
    injection_audit = metadata["injection_audit"]
    eligible_points = int(injection.eligible_changed_5m_points.iloc[0])
    all_points = injection_audit["all_known_changed_5m_points"]
    report = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Matrix Profile 告警减负与核验分流</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1280px;margin:32px auto;padding:16px;color:#202b39;line-height:1.7}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:16px 0}}th,td{{padding:8px;border-bottom:1px solid #dbe2e9;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#edf2f7}}.note{{border-left:4px solid #d39b36;background:#fff7e8;padding:12px 18px}}a{{color:#245c9e}}code{{background:#eef3f7;padding:2px 5px}}</style></head><body>
<h1>减少优先告警，不把所有变化当故障</h1>
<p>检测期 {metadata['detection_start']}（含）至 {metadata['detection_end_exclusive']}（不含）的原始 {before:,} 条正式事件全部保留。新工程规则生成 <strong>{priority:,} 个优先告警核验单元</strong>，另有 {quality:,} 个数据质量核验单元。优先队列相对原先把每条正式事件都当告警的处理负担减少 {100*(1-priority/before):.2f}%；这是任务队列变化，不是证明真实误报减少。</p>
<p class="note">优先告警仍不是已确认故障。运行变化待核验不等于正常，数据质量问题也不等于电力故障。旧 15 分钟高分区间和分数完全未改变，不再用它们直接统计故障告警。较早的补漏实验仍保留但不启用。</p>
<h2>一、三类输出与去重后的工作量</h2>{table.to_html(index=False)}
<p>每条原事件恰好归入一个队列、一个核验单元。合并只处理同通道同类型同方向的重叠／相邻记录，或原先已有的共同变化组中真正重叠、同队列同方向的记录。共同变化组只是核验便利，不证明同一物理故障。所有成员 ID 可回溯。</p>
{priority_categories.to_html(index=False)}
<p class="note">太阳能的季节、天气、云遮和运行状态变化也可能满足功率突变规则。优先队列中太阳能占比较高，必须做天气／设备状态核验；高优先级不代表高故障概率。此处没有天气数据，不能自动排除正常云遮，也不能宣称逆变器故障。</p>
<h2>二、优先告警的固定规则</h2>
<p>参考固定为 {metadata['baseline_start']} 至 {metadata['baseline_end']} 的原始观测；工作日／周末分开，太阳能沿用全部历史。历史同时段允许 ±{policy['slot_radius_points']*5} 分钟排程偏差，采用 {100*policy['envelope_quantile']:.0f}%–{100*(1-policy['envelope_quantile']):.0f}% 历史范围。不用目标月份分布重新选择阈值。</p>
<p>功率超出该范围的幅度必须超过 <code>max({policy['minimum_effect_kw']} kW, {100*policy['relative_effect']:.0f}% × 历史 |功率| P95)</code>，连续至少 {policy['minimum_duration_minutes']} 分钟，峰值至少达到该幅度门槛的 {policy['peak_effect_multiplier']:.0f} 倍；同一连续区段开始时的实际五分钟突变须超出冻结的历史跳变门槛，并由原事件的短／中尺度共同高分或电流超历史范围支持。缺少证据、长时间缓慢漂移或纯形状变化暂留待核验。</p>
<p>电流是佐证，不是独立故障真值；模型的 high/medium 标签也不是故障概率。以上是可审计的保守工程规则，尚未通过独立正常／故障标签集校准。</p>
<p>优先时间范围使用实际满足规则的 5 分钟连续核心，保存在 priority_intervals.csv；不把整个 2/6/24 小时窗口或原事件区间都当作优先告警，也不复制 30 分钟分数制造高分辨率。</p>
<h2>三、减少告警的代价：原始／注入配对检查</h2>
<p>沿用已有八月人工修改配对验证，{injection_audit['paired_channels']} 个通道可评分，{injection_audit['unscored_channels']} 个通道仍不可评分。有效改变点 {eligible_points:,} 个，全体改变点 {all_points:,} 个，其中 {all_points-eligible_points:,} 个不可评分。以下覆盖率的分母为有效改变点；区段重叠不等于真实故障召回率。只分析原验证包含的修改日期，不是完整八月背景。</p>
{comparison.to_html(index=False, float_format=lambda value:f'{value:.2f}')}
<p class="note">“全部保留覆盖”应与之前的验证完全相同；“仅优先核心覆盖”可能大幅下降，这是保守告警的代价。多数已有人工修改表现为平线，被单独归为数据质量问题。若运行时只看 priority_alert_units.csv 而忽略质量与待核验队列，会漏看这些已知修改。injection_downgraded_runs.csv 列出原有事件覆盖、但优先核心和质量队列均不覆盖的改变区段，不能把它们解释为正常。</p>
<h2>四、降级原因</h2>{reasons.to_html(index=False)}
<h2>五、部分优先事件（按来源顺序，不是故障概率排名）</h2>{top.to_html(index=False)}
<h2>文件入口</h2><p><a href="priority_alert_units.csv">优先告警核验单元</a> · <a href="priority_intervals.csv">真实时间核心</a> · <a href="operating_change_review_units.csv">运行变化待核验单元</a> · <a href="data_quality_units.csv">数据质量核验单元</a> · <a href="events_all_classified.csv">完整{before:,}条分流台账</a> · <a href="event_memberships.csv">ID对应关系</a> · <a href="injection_triage_comparison.csv">注入对比</a> · <a href="injection_downgraded_runs.csv">被降级的注入区段</a> · <a href="triage_summary.json">配置、计数与输入哈希</a></p>
<p>生产分数／事件只读，源 CSV 未改写，其他三个模型与四模型对齐协议未改动。未标注的未修改点不能当已确认正常点，因此不计算精确率、F1 或真实误报率。下一步应人工核验优先队列，同时抽查待核验队列；不能只以条数变少证明检测改进。</p></body></html>'''
    (output / "report.html").write_text(report, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/matrix_profile/results/september_fixed_july_local")
    parser.add_argument("--input-dir", help="Prepared original five-minute July/September CSVs")
    parser.add_argument("--validation-dir", default="runs/analysis/matrix_profile_existing_injections_20261005")
    parser.add_argument("--config", default="configs/matrix_profile_alert_triage.json")
    parser.add_argument("--output-dir", default="runs/analysis/matrix_profile_alert_triage_20261005")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--finalize-only", action="store_true", help="Re-render a completed snapshot without reclassifying events")
    args = parser.parse_args(argv)
    if args.workers < 1: parser.error("--workers must be positive")
    run, validation, output = resolve(args.run_dir), resolve(args.validation_dir), resolve(args.output_dir)
    if args.finalize_only:
        metadata = json.loads((output / "triage_summary.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete": raise ValueError("Snapshot is incomplete")
        executed_hashes = {key.replace("\\", "/"): value for key, value in metadata["implementation_sha256"].items()}
        for relative in ("methods/matrix_profile/src/anomalies/multiscale/alert_triage.py",
                         "methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py"):
            if digest(ROOT / relative) != executed_hashes[relative]:
                raise ValueError("Classification or calibration code changed; use a fresh run")
        if digest(resolve(args.config)) != metadata["policy_config_sha256"]:
            raise ValueError("Triage policy changed; use a fresh run")
        counts = pd.read_csv(output / "queue_counts.csv")
        units = pd.read_csv(output / "review_units.csv")
        classified = pd.read_csv(output / "events_all_classified.csv")
        injection = pd.read_csv(output / "injection_triage_comparison.csv")
        metadata["report_render_code_sha256"] = digest(Path(__file__))
        metadata["report_render_notes"] = "Report-only refresh; executed classification hashes and results preserved"
        render_report(output, counts, units, classified, injection, metadata)
        (output / "triage_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report-only refresh: {output / 'report.html'}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; no previous results are overwritten")
    config_path = resolve(args.config)
    policy = AlertTriageConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    run_meta = json.loads((run / "run_metadata.json").read_text())
    if run_meta["history_mode"] != "fixed":
        raise ValueError("This triage runner currently requires a fixed historical baseline")
    config = DetectionConfig(baseline_start=date.fromisoformat(run_meta["baseline_start"]),
                             baseline_end=date.fromisoformat(run_meta["baseline_end"]),
                             detection_start=date.fromisoformat(run_meta["detection_start"]),
                             detection_end=(pd.Timestamp(run_meta["detection_end_exclusive"])-pd.Timedelta(days=1)).date(),
                             timezone=run_meta["timezone"])
    input_dir = resolve(args.input_dir) if args.input_dir else ROOT / "data/processed/matrix_profile" / run.name
    protected = [run / name for name in ("ALL_EVENTS.csv", "ALL_SCORES_30MIN.csv", "ALL_SCORES_15MIN.csv", "run_metadata.json")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    events = pd.read_csv(run / "ALL_EVENTS.csv")
    if events.empty: raise ValueError("No saved source events")
    if events.event_id.isna().any() or events.event_id.duplicated().any(): raise ValueError("Non-unique source event IDs")
    output.mkdir(parents=True, exist_ok=True)
    groups = events.groupby(["device_id", "channel_idx"], sort=True)
    tasks = [(f"{device}_{channel}", group.to_dict("records"), list(events.columns), str(input_dir), config, policy)
             for (device, channel), group in groups]
    print(f"Lossless review triage: {len(events)} source events, {len(tasks)} channels; frozen history", flush=True)
    frames, audits = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(september_worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(pending), 1):
            classified, audit = future.result(); frames.append(classified); audits.append(audit)
            if number % 10 == 0 or number == len(tasks): print(f"September triage {number}/{len(tasks)}", flush=True)
    classified = pd.concat(frames, ignore_index=True).set_index("event_id", drop=False).loc[events.event_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(classified[list(events.columns)].reset_index(drop=True), events.reset_index(drop=True))
    classified, units, counts = export_queues(classified, output, policy)
    pd.DataFrame(audits).to_csv(output / "september_input_audit.csv", index=False)
    print(counts.to_string(index=False), flush=True)
    injection, injection_audit = evaluate_injections(validation, output, policy, args.workers)
    for path in protected:
        if digest(path) != hashes[str(path.relative_to(ROOT))]: raise RuntimeError("Production file changed during triage")
    metadata = dict(status="complete", policy=config_dict(policy), source_event_count=len(events),
                    review_unit_count=len(units), queue_counts=counts.to_dict("records"),
                    baseline_start=run_meta["baseline_start"], baseline_end=run_meta["baseline_end"],
                    detection_start=run_meta["detection_start"], detection_end_exclusive=run_meta["detection_end_exclusive"],
                    original_production_sha256=hashes, policy_config_sha256=digest(config_path),
                    implementation_sha256={str(path.relative_to(ROOT)): digest(path) for path in (
                        Path(__file__), ROOT / "methods/matrix_profile/src/anomalies/multiscale/alert_triage.py",
                        ROOT / "methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py")},
                    injection_audit=injection_audit, no_retraining=True, no_score_recomputation=True,
                    priority_time_source="five_minute_continuous_evidence_cores",
                    limitations=["Engineering triage, not validated fault probabilities", "Review queue is not a normal label",
                                 "Current correlation is supporting evidence, not independent ground truth",
                                 "Paired injection evaluation only includes dates containing known edits",
                                 "Unscored channels remain unavailable, not normal", "No precision or F1 without labelled normal observations"])
    (output / "triage_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    render_report(output, counts, units, classified, injection, metadata)
    print(f"Complete: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
