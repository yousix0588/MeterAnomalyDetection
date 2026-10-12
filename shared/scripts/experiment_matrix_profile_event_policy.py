"""Isolated opt-in amplitude event comparison using a completed baseline pilot.

This does NOT change production scoring, admission, thresholds, or event routing.
Raw July and cleaned July-August retain their previous frozen references.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/scripts"))
import experiment_matrix_profile_baselines as baseline
import numpy as np
import pandas as pd
from anomalies.multiscale.magnitude_events import (
    build_magnitude_events, calibrate_magnitude_reference, event_point_mask, MagnitudeEventConfig,
)

VARIANTS = ("raw_july", "clean_july_august")
POLICIES = ("legacy", "envelope_confirmed", "transition_guarded")


def policies_for_day(scores, target, prepared, context, config, reference):
    legacy = baseline.day_events(scores, target, prepared, context, config)
    formal = [event for event in legacy if event.confidence is not baseline.Confidence.CANDIDATE]
    quality = [event for event in formal if event.event_type.value == "data_quality"]
    freeze, _ = baseline.sensor_freezes(target)
    result = {"legacy": formal}
    for policy in POLICIES[1:]:
        supplemental = build_magnitude_events(scores, target, reference, context, config,
                                               legacy_events=formal, quality_events=quality,
                                               excluded_mask=freeze.to_numpy(), mode=policy)
        result[policy] = formal + supplemental
    return result


def covers(events, start, end):
    return any(event.start_time < end and event.end_time > start for event in events)


def annotate_scores(native, events):
    result = native.copy()
    result["event_is_anomaly"] = False
    result["supplemental_event_is_anomaly"] = False
    for event in events:
        overlap = (result.interval_start < event.end_time) & (result.interval_end > event.start_time)
        result.loc[overlap, "event_is_anomaly"] = True
        if event.evidence.get("supplemental"):
            result.loc[overlap, "supplemental_event_is_anomaly"] = True
    return result


def worker(task):
    key, context, pilot_path, original_path, output_path = task
    pilot, output = Path(pilot_path), Path(output_path)
    original = baseline._normalise_index(pd.read_csv(original_path), "Australia/Sydney")
    combined = baseline._normalise_index(pd.read_csv(pilot/"prepared_inputs"/f"{key}.csv"), "Australia/Sydney")
    frame = pd.concat([original, combined.loc[combined.index.month == 8]]).sort_index()
    assert not frame.index.has_duplicates
    frozen, _ = baseline.sensor_freezes(frame)
    july = frame.loc[frame.index.month == 7, "pRealKw"]
    probe_scale = max(1., float(july.abs().quantile(.95)))
    summaries, probes, days = [], [], []
    for variant in VARIANTS:
        first, last, clean = baseline.VARIANTS[variant]
        config = baseline.local.DetectionConfig(baseline_start=first, baseline_end=last,
                                               detection_start=date(2026,9,1), detection_end=date(2026,9,30))
        source, _ = baseline.mask_history(frame, config, frozen) if clean else (frame.copy(), 0)
        preparation = baseline.prepare_channel(source, config, context.category_name)
        prepared = preparation.data
        cache, reference_cache, rows = {}, {}, []
        events = {policy: [] for policy in POLICIES}
        day_policies, day_scores = {}, {}
        if prepared is not None:
            for day, target in sorted(prepared.target_5m.items()):
                scores = baseline.score_day(prepared, day, context, config, cache)
                day_scores[day] = scores
                group = "all" if context.category_name.casefold() == "solar generation" else day.weekday() >= 5
                if group not in reference_cache:
                    reference_cache[group] = calibrate_magnitude_reference(prepared, day, context, config)
                found = policies_for_day(scores, target, prepared, context, config, reference_cache[group])
                day_policies[day] = found
                for policy in POLICIES:
                    events[policy].extend(found[policy])
                rows.extend(baseline._interval_score_records(scores, context, config))
        statuses = [dict(date=str(day), status="scored" if day in day_scores else
                        ("missing_target" if prepared else preparation.status.value))
                    for day in pd.date_range("2026-09-01", "2026-09-30").date]
        native = baseline.local.complete_grid(pd.DataFrame(rows, columns=baseline.INTERVAL_SCORE_COLUMNS),
                                              key, pd.DataFrame(statuses), config)
        old = pd.read_csv(pilot/"variants"/variant/key/"scores_30min.csv")
        np.testing.assert_allclose(pd.to_numeric(native.max_score).to_numpy(dtype=float),
                                   pd.to_numeric(old.max_score).to_numpy(dtype=float),
                                   atol=1e-8, rtol=0, equal_nan=True)
        assert np.array_equal(native.available, old.available)
        expected = pd.read_csv(pilot/"variants"/variant/key/"events.csv")
        assert set(baseline.event_frame(events["legacy"]).event_id) == set(expected.event_id)
        for policy in POLICIES:
            destination = output/"variants"/variant/policy/key
            destination.mkdir(parents=True, exist_ok=True)
            formal = baseline.event_frame(events[policy])
            formal.to_csv(destination/"events.csv", index=False)
            # Recompute event overlap on each grid. Do not repeat a real 5m
            # event flag onto the wrong 15m half of a 30m score interval.
            canonical = baseline._upsample_interval_frame(native)
            for grid, name in ((native, "scores_30min.csv"), (canonical, "scores_15min.csv")):
                marked = annotate_scores(grid, events[policy])
                marked["event_policy"] = policy
                marked["baseline_variant"] = variant
                marked.to_csv(destination/name, index=False)
            dq = int(formal.event_type.eq("data_quality").sum())
            supplemental = sum(bool(event.evidence.get("supplemental")) for event in events[policy])
            summaries.append(dict(series_id=key, baseline_variant=variant, event_policy=policy,
                                  preparation_status=preparation.status.value, reason=preparation.reason,
                                  available_intervals=int(native.available.sum()),
                                  score_flagged_intervals=int(native.is_anomaly.sum()),
                                  formal_events=len(formal), data_quality_events=dq,
                                  behavior_events=len(formal)-dq, supplemental_events=supplemental))
        days.extend(dict(series_id=key, baseline_variant=variant, **record) for record in statuses)
        if prepared is None:
            continue
        probe_day = date(2026,9,15) if date(2026,9,15) in day_scores else min(day_scores)
        target = prepared.target_5m[probe_day]
        group = "all" if context.category_name.casefold() == "solar generation" else probe_day.weekday() >= 5
        ref = reference_cache[group]
        for kind in ("spike_15m", "level_shift_6h", "multisensor_freeze_4h"):
            modified, start, end = baseline.controlled_probe(target, kind, probe_scale)
            if modified is None or modified.loc[(modified.index>=start)&(modified.index<end)].isna().any().any():
                continue
            probe = replace(prepared, target_5m={**prepared.target_5m, probe_day:modified},
                            target_30m={**prepared.target_30m, probe_day:baseline.resample_probe(modified, config)})
            scores = baseline.score_day(probe, probe_day, context, config, cache)
            found = policies_for_day(scores, modified, probe, context, config, ref)
            inside = (target.index>=start)&(target.index<end)
            for policy in POLICIES:
                before = day_policies[probe_day][policy]
                after = found[policy]
                before_mask = event_point_mask(before, target.index)
                after_mask = event_point_mask(after, target.index)
                newly = after_mask & ~before_mask
                background = covers(before, start, end)
                after_overlap = covers(after, start, end)
                probes.append(dict(series_id=key, baseline_variant=variant, event_policy=policy,
                                   kind=kind, start=start, end=end, amplitude_scale_kw=probe_scale,
                                   background_formal_overlap=background, after_formal_overlap=after_overlap,
                                   new_overlap_without_background=after_overlap and not background,
                                   newly_covered_inside_5m_points=int((newly&inside).sum()),
                                   newly_covered_outside_5m_points=int((newly&~inside).sum()),
                                   after_quality_overlap=covers([event for event in after if event.event_type.value=="data_quality"],start,end),
                                   after_supplemental_overlap=covers([event for event in after if event.evidence.get("supplemental")],start,end)))
                pd.testing.assert_frame_equal(probe.history_5m[min(probe.history_5m)], prepared.history_5m[min(prepared.history_5m)])
            # Store complete probe events for independent overlap verification.
            for policy in POLICIES:
                destination = output/"probes"/variant/policy/key/kind
                destination.mkdir(parents=True, exist_ok=True)
                baseline.event_frame(found[policy]).to_csv(destination/"events.csv",index=False)
                baseline.event_frame(day_policies[probe_day][policy]).to_csv(destination/"background_events.csv",index=False)
            probe_file = output/"probe_inputs"/key/f"{kind}.csv"
            probe_file.parent.mkdir(exist_ok=True, parents=True)
            if not probe_file.exists():
                modified.to_csv(probe_file, index_label="timestamp")
                target.to_csv(probe_file.with_name("background.csv"),index_label="timestamp")
    checkpoint = output/"completed_channels"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint/f"{key}.json").write_text(json.dumps(dict(summary=summaries, probes=probes, days=days),default=str),encoding="utf-8")
    return summaries, probes, days


def render_report(output, aggregate, checks, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    for variant,color in (("raw_july","#708a9c"),("clean_july_august","#275d87")):
        group=aggregate.loc[aggregate.baseline_variant.eq(variant)].set_index("event_policy").reindex(POLICIES)
        offset = -.18 if variant=="raw_july" else .18
        axes[0].bar(np.arange(3)+offset,group.common_behavior_events,width=.35,label=variant,color=color)
    axes[0].set_xticks(range(3),["Legacy", "Envelope + evidence", "Transition guarded"])
    axes[0].set_ylabel("Behavior events on shared scored days")
    axes[0].legend(fontsize=9)
    power=checks.loc[checks.kind.ne("multisensor_freeze_4h")]
    totals=[]
    for position,(policy,color) in enumerate(zip(POLICIES,("#708a9c","#ba9b4c","#275d87"))):
        group=power.loc[power.event_policy.eq(policy)]
        count=int(group.after_formal_overlap.sum())
        total=int(group.cases.sum())
        totals.append(total)
        axes[1].bar(position,count,color=color)
        axes[1].text(position,count+.5,f"{count}/{total}",ha="center")
    axes[1].set_xticks(range(3),["Legacy","Envelope + evidence","Transition guarded"])
    axes[1].set_ylabel("Power probes overlapping a formal event")
    axes[1].tick_params(axis="x",labelrotation=10)
    axes[1].set_ylim(0,max([1]+totals)*1.15)
    fig.suptitle("Opt-in amplitude event policies (unlabelled pilot, not accuracy)")
    fig.savefig(output/"event_policy_comparison.png",dpi=160)
    plt.close(fig)
    view=aggregate[["baseline_variant","event_policy","common_formal_events","common_behavior_events",
                    "common_supplemental_events","available_intervals","unavailable_channels"]]
    probe_view=checks.loc[checks.kind.ne("multisensor_freeze_4h"),
                          ["baseline_variant","event_policy","kind","cases","background_formal_overlap",
                           "after_formal_overlap","new_overlap_without_background","newly_covered_outside_5m_points"]]
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多尺度事件路径优化试验</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1200px;margin:30px auto;padding:20px;line-height:1.7}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:7px;border-bottom:1px solid #ddd}}th{{background:#eef3f7}}img{{max-width:100%}}</style>
<h1>九月多尺度 Matrix Profile 事件路径优化</h1>
<p><strong>结论：带限制的幅值路径提高了受控扰动覆盖，但幅值路径本身会增加事件。</strong>共同覆盖范围中，原七月原路径为127个正式事件，清洗七至八月加严格幅值路径为102个，减少19.69%；这里同时改变了历史基线，不能把减量全部归因于新事件路径。清洗七至八月内部，原路径78个变为102个，新增24个。宽松路径新增95个，不作为当前首选。没有真实正常标签，无法证明误报减少。</p>
<p>对同一批11个通道比较原始七月、清洗七至八月两个历史参考。共同有效原生30分钟区间 {metadata['common_available_intervals']:,} 个。历史校准、2/6/24小时窗口、分数阈值和准入规则均未变。新路径仅为显式调用的试验模块，未启用到生产检测器。</p>
<p>独立幅值候选要求五分钟真实功率超过历史同时间邻域范围与绝对/相对幅度下限，并有电流、局部突变或变点证据。严格路径还要求较大的超界幅度及起始附近的转变或形态与电流共同支持。历史范围只读取九月之前的数据。冻结与缺测区段不转成业务故障。</p>
<img src="event_policy_comparison.png"><h2>相同覆盖范围上的背景事件</h2>{view.to_html(index=False)}
<h2>相同受控扰动</h2><p>事件与注入区段重叠可能来自原背景告警，不是召回率。new_overlap_without_background 单独统计原区段无事件、扰动后出现事件的案例。新增覆盖的区段外五分钟点另存，不能把窗口拖尾当精确定位。4小时多传感器冻结在原七月的11个、清洗七至八月的10个有效案例上均保持数据质量事件重叠，未增加注入区段内的幅值补充事件。</p>{probe_view.to_html(index=False)}
<h2>输出定义与限制</h2><p>标准 raw_score/max_score/mean_score/is_anomaly/available 字段保持原义。event_is_anomaly 单独表示正式事件覆盖，supplemental_event_is_anomaly 表示新增路径覆盖。原生分数仍是30分钟，15分钟仍标注 upsampled_from_30m；补充事件边界来自真实五分钟观测，15分钟事件标记重新计算而非复制。三个策略是同一模型的对照，不得当作三个独立模型进入集成。</p>
<p>没有真实故障/正常标签，新增事件不能称为已确认故障，背景数量也不是误报率。严谨减量须与灵敏度和覆盖一起评估。部分参考下不可评分通道保持不可用，未放松准入规则。本次不覆盖旧结果，不改变默认基线。</p>
<p><a href="aggregate_comparison.csv">事件对照</a> · <a href="probe_results.csv">逐扰动结果</a> · <a href="trial_summary.csv">通道与覆盖</a> · <a href="day_status.csv">日状态</a></p></html>'''
    (output/"report.html").write_text(report,encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir",default="runs/analysis/matrix_profile_baseline_pilot_20261005_v2")
    parser.add_argument("--output-dir",default="runs/analysis/matrix_profile_event_policy_20261005")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--finalize-only",action="store_true",
                        help="Validate and finalize all completed channel checkpoints without re-scoring")
    args=parser.parse_args(argv)
    pilot,output=baseline.local.resolve(args.pilot_dir),baseline.local.resolve(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.finalize_only:
        raise FileExistsError("Use a fresh isolated output directory")
    output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((pilot/"experiment_summary.json").read_text(encoding="utf-8"))
    run=ROOT/"runs/matrix_profile/results/september_fixed_july_local"
    original_hashes={name:baseline.digest(run/name) for name in prior["original_sha256"]}
    assert original_hashes==prior["original_sha256"]
    run_metadata=json.loads((run/"run_metadata.json").read_text())
    inputs=Path(run_metadata["prepared_input_dir"])
    repository=json.loads((ROOT/"methods/matrix_profile/src/meter_repository.json").read_text())
    tasks=[]
    for key in prior["channels"]:
        device,channel=key.rsplit("_",1)
        context=baseline._context_from_metadata(repository,device,int(channel))
        tasks.append((key,context,str(pilot),str(inputs/f"{key}.csv"),str(output)))
    print(f"Event policy pilot: {len(tasks)} channels x 2 frozen references x 3 policies",flush=True)
    summary,probes,days=[],[],[]
    if args.finalize_only:
        for key in prior["channels"]:
            checkpoint=json.loads((output/"completed_channels"/f"{key}.json").read_text(encoding="utf-8"))
            summary.extend(checkpoint["summary"]); probes.extend(checkpoint["probes"]); days.extend(checkpoint["days"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(worker,task):task[0] for task in tasks}
            for number,future in enumerate(as_completed(pending),1):
                result,checks,statuses=future.result()
                summary.extend(result); probes.extend(checks); days.extend(statuses)
                print(f"Completed {number}/{len(tasks)}: {pending[future]}",flush=True)
    # Re-validate persisted controls, including resumed/finalized checkpoints.
    for key in prior["channels"]:
        for variant in VARIANTS:
            old_scores=pd.read_csv(pilot/"variants"/variant/key/"scores_30min.csv")
            expected=set(pd.read_csv(pilot/"variants"/variant/key/"events.csv").event_id)
            for policy in POLICIES:
                directory=output/"variants"/variant/policy/key
                scored=pd.read_csv(directory/"scores_30min.csv")
                assert scored.interval_start.equals(old_scores.interval_start)
                for column in ("raw_score","max_score","mean_score","max_percentile","mean_percentile"):
                    np.testing.assert_allclose(pd.to_numeric(scored[column]),pd.to_numeric(old_scores[column]),
                                               atol=1e-8,rtol=0,equal_nan=True)
                assert scored.available.equals(old_scores.available)
                assert scored.is_anomaly.equals(old_scores.is_anomaly)
                found=pd.read_csv(directory/"events.csv")
                assert expected <= set(found.event_id)
                if policy=="legacy":
                    assert expected==set(found.event_id)
    summary=pd.DataFrame(summary); probes=pd.DataFrame(probes); days=pd.DataFrame(days)
    summary.to_csv(output/"trial_summary.csv",index=False)
    probes.to_csv(output/"probe_results.csv",index=False)
    days.to_csv(output/"day_status.csv",index=False)
    common=(days.groupby(["series_id","date"]).status.apply(lambda series:series.eq("scored").all()))
    common_days=set(common[common].index)
    aggregate=[]
    for variant in VARIANTS:
        for policy in POLICIES:
            rows=[]
            for key in prior["channels"]:
                events=pd.read_csv(output/"variants"/variant/policy/key/"events.csv")
                rows.extend(record for record in events.to_dict("records")
                            if (key,record["start_time"][:10]) in common_days)
            quality=sum(record["event_type"]=="data_quality" for record in rows)
            supplemental=sum(json.loads(record["evidence"]).get("supplemental",False) for record in rows)
            group=summary.loc[summary.baseline_variant.eq(variant)&summary.event_policy.eq(policy)]
            aggregate.append(dict(baseline_variant=variant,event_policy=policy,
                                  common_formal_events=len(rows),common_behavior_events=len(rows)-quality,
                                  common_supplemental_events=int(supplemental),
                                  available_intervals=int(group.available_intervals.sum()),
                                  unavailable_channels=int(group.available_intervals.eq(0).sum())))
    aggregate=pd.DataFrame(aggregate)
    aggregate.to_csv(output/"aggregate_comparison.csv",index=False)
    checks=probes.groupby(["baseline_variant","event_policy","kind"]).agg(
        cases=("series_id","size"),background_formal_overlap=("background_formal_overlap","sum"),
        after_formal_overlap=("after_formal_overlap","sum"),
        new_overlap_without_background=("new_overlap_without_background","sum"),
        newly_covered_inside_5m_points=("newly_covered_inside_5m_points","sum"),
        newly_covered_outside_5m_points=("newly_covered_outside_5m_points","sum"),
        quality_overlap=("after_quality_overlap","sum"),supplemental_overlap=("after_supplemental_overlap","sum")).reset_index()
    checks.to_csv(output/"probe_summary.csv",index=False)
    assert {name:baseline.digest(run/name) for name in original_hashes}==original_hashes
    metadata=dict(analysis_only=True,production_changed=False,channels=prior["channels"],
                  variants=VARIANTS,policies=POLICIES,common_available_intervals=len(common_days)*48,
                  original_sha256=original_hashes,policy_config=asdict(MagnitudeEventConfig()),
                  experiment_code_sha256=baseline.digest(__file__),
                  policy_code_sha256=baseline.digest(ROOT/"methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py"),
                  baseline_cutoff="2026-08-31",windows_hours=[2,6,24],score_threshold=.995,
                  validation="Same-baseline original scores and legacy event IDs reproduced; original output hashes unchanged",
                  limitation="Purposive unlabelled pilot and controlled stresses, not real-world accuracy")
    (output/"experiment_summary.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    render_report(output,aggregate,checks,metadata)
    print(aggregate.to_string(index=False),flush=True)
    print(checks.to_string(index=False),flush=True)
    print(f"Report: {output/'report.html'}",flush=True)


if __name__=="__main__":
    main()
