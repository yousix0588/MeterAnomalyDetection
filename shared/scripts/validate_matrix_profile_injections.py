"""Validate opt-in event paths on existing August power edits, without retraining.

July reference observations always come from the original channel file. Only
declared edited points are replayed into an in-memory copy. Other original
measurements remain intact, including signed-power columns outside the edit.
Unmodified observations have no verified normal labels: report background
alarms, not false positives or precision.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"shared/scripts"))
import experiment_matrix_profile_event_policy as experiment
import numpy as np
import pandas as pd
from anomalies.multiscale.magnitude_events import _contiguous_groups

baseline=experiment.baseline


def localized_replay(original, injected, tolerance=1e-4):
    if original.index.has_duplicates or injected.index.has_duplicates:
        raise ValueError("Duplicate injection/original timestamps")
    if len(injected.index.difference(original.index)):
        raise ValueError("Injection timestamps absent from original")
    aligned=injected.reindex(original.index)
    reference=aligned.pRealKw_original
    observed=aligned.pRealKw
    if not np.allclose(reference,original.pRealKw,rtol=1e-7,atol=tolerance,equal_nan=True):
        raise ValueError("Embedded original power does not match the independent original file")
    mask=(observed-reference).abs().gt(tolerance)
    replay=original.copy()
    common=[column for column in original if column in aligned]
    replay.loc[mask,common]=aligned.loc[mask,common]
    pd.testing.assert_frame_equal(replay.loc[~mask],original.loc[~mask])
    changes={column:int((~np.isclose(aligned[column],original[column],rtol=1e-7,
                                   atol=tolerance,equal_nan=True)&~mask).sum())
             for column in common if column!="pRealKw"}
    return replay,mask,changes


def worker(task):
    key,context,original_path,injected_path,output_path=task
    original=baseline._normalise_index(pd.read_csv(original_path),"Australia/Sydney")
    injected=baseline._normalise_index(pd.read_csv(injected_path),"Australia/Sydney")
    replay,known,outside=localized_replay(original,injected)
    if known.loc[known.index.month==7].any():
        raise ValueError("Declared power injections contaminate July reference")
    output=Path(output_path)/"channels"/key
    output.mkdir(parents=True,exist_ok=True)
    config=baseline.local.DetectionConfig(detection_start=date(2026,8,1),detection_end=date(2026,8,31))
    preparation=baseline.prepare_channel(original,config,context.category_name)
    modified_preparation=baseline.prepare_channel(replay,config,context.category_name)
    audit=dict(series_id=key,preparation_status=preparation.status.value,reason=preparation.reason,
               known_changed_5m_points=int(known.sum()),outside_field_changes=sum(outside.values()),
               outside_columns=json.dumps({column:count for column,count in outside.items() if count},sort_keys=True),
               original_sha256=baseline.digest(original_path),injected_sha256=baseline.digest(injected_path))
    if preparation.data is None:
        return audit,[],[],[]
    assert modified_preparation.status==preparation.status
    prepared,modified=preparation.data,modified_preparation.data
    for day in prepared.history_5m:
        pd.testing.assert_frame_equal(prepared.history_5m[day],modified.history_5m[day])
    dates=sorted(set(original.index[known].date))
    cache,refs={},{}
    all_events={version:{policy:[] for policy in experiment.POLICIES} for version in ("original","injected")}
    all_rows={version:[] for version in all_events}
    statuses=[]
    for day in dates:
        if day not in prepared.target_5m or day not in modified.target_5m:
            statuses.append(dict(series_id=key,date=str(day),status="missing_target",reason="One paired day is invalid"))
            continue
        statuses.append(dict(series_id=key,date=str(day),status="scored",reason="Paired original/injected day"))
        group="all" if context.category_name.casefold()=="solar generation" else day.weekday()>=5
        if group not in refs:
            refs[group]=experiment.calibrate_magnitude_reference(prepared,day,context,config)
        for version,data in (("original",prepared),("injected",modified)):
            scores=baseline.score_day(data,day,context,config,cache)
            found=experiment.policies_for_day(scores,data.target_5m[day],data,context,config,refs[group])
            all_rows[version].extend(baseline._interval_score_records(scores,context,config))
            for policy in experiment.POLICIES:
                all_events[version][policy].extend(found[policy])
    evaluation_index=original.index[(original.index.month==8)&np.isin(original.index.date,dates)]
    valid_dates={pd.Timestamp(record["date"]).date() for record in statuses if record["status"]=="scored"}
    eligible=np.isin(evaluation_index.date,list(valid_dates))
    changed=known.reindex(evaluation_index).to_numpy(dtype=bool)
    point_labels=pd.DataFrame(dict(timestamp=evaluation_index,is_injected=changed,eligible=eligible))
    point_labels.to_csv(output/"changed_points.csv",index=False)
    summary,segments=[],[]
    for version in all_events:
        native=pd.DataFrame(all_rows[version],columns=baseline.INTERVAL_SCORE_COLUMNS)
        # Only paired edited days are scored. No claims about all-month background.
        if not native.empty:
            native.to_csv(output/f"{version}_scores_30min.csv",index=False)
            baseline._upsample_interval_frame(native).to_csv(output/f"{version}_scores_15min.csv",index=False)
        for policy in experiment.POLICIES:
            baseline.event_frame(all_events[version][policy]).to_csv(output/f"{version}_{policy}_events.csv",index=False)
    for policy in experiment.POLICIES:
        before=all_events["original"][policy]; after=all_events["injected"][policy]
        before_mask=experiment.event_point_mask(before,evaluation_index)
        after_mask=experiment.event_point_mask(after,evaluation_index)
        novel=after_mask&~before_mask
        changed_eligible=changed&eligible
        summary.append(dict(series_id=key,event_policy=policy,eligible_changed_5m_points=int(changed_eligible.sum()),
                            original_covered_changed_5m_points=int((before_mask&changed_eligible).sum()),
                            injected_covered_changed_5m_points=int((after_mask&changed_eligible).sum()),
                            newly_covered_changed_5m_points=int((novel&changed_eligible).sum()),
                            newly_covered_unchanged_5m_points=int((novel&~changed&eligible).sum()),
                            original_formal_events=len(baseline.event_frame(before)),
                            injected_formal_events=len(baseline.event_frame(after))))
        for first,stop in _contiguous_groups(changed,evaluation_index):
            start=evaluation_index[first]; end=evaluation_index[stop-1]+pd.Timedelta(minutes=5)
            segment_eligible=bool(eligible[first:stop].all())
            segments.append(dict(series_id=key,event_policy=policy,start=start,end=end,
                                 changed_5m_points=stop-first,eligible=segment_eligible,
                                 original_overlap=bool(before_mask[first:stop].any()) if segment_eligible else None,
                                 injected_overlap=bool(after_mask[first:stop].any()) if segment_eligible else None,
                                 newly_covered_points=int(novel[first:stop].sum()) if segment_eligible else None,
                                 covered_changed_points=int(after_mask[first:stop].sum()) if segment_eligible else None))
    return audit,summary,segments,statuses


def render_report(output,audit,aggregate,metadata):
    labels={"legacy":"原路径","envelope_confirmed":"幅值与证据","transition_guarded":"幅值与转变限制"}
    display=aggregate[["event_policy","eligible_changed_5m_points","injected_covered_changed_5m_points",
                       "changed_point_coverage_percent","eligible_changed_runs","injected_overlapped_runs",
                       "newly_overlapped_runs"]].copy()
    display.event_policy=display.event_policy.map(labels)
    display.columns=["事件路径","有效改变点","覆盖改变点","改变点覆盖率 (%)","有效连续改变区段","重叠区段","原无重叠而新增"]
    for column in ("有效改变点","覆盖改变点","有效连续改变区段","重叠区段","原无重叠而新增"):
        display[column]=display[column].astype(int)
    table=display.to_html(index=False,float_format=lambda value:f"{value:.2f}")
    background=aggregate[["event_policy","original_formal_events","injected_formal_events",
                          "newly_covered_changed_5m_points","newly_covered_unchanged_5m_points"]].copy()
    background.event_policy=background.event_policy.map(labels)
    background.columns=["事件路径","原始数据事件","注入数据事件","改变点新增覆盖","未改变点新增覆盖"]
    availability=aggregate[["event_policy","all_declared_changed_5m_points","unscored_changed_5m_points",
                            "all_changed_point_coverage_percent"]].copy()
    availability.event_policy=availability.event_policy.map(labels)
    availability.columns=["事件路径","全体已知改变点","不可评分改变点","全体改变点覆盖率 (%)"]
    legacy=aggregate.loc[aggregate.event_policy.eq("legacy")].iloc[0]
    guarded=aggregate.loc[aggregate.event_policy.eq("transition_guarded")].iloc[0]
    gain=guarded.changed_point_coverage_percent-legacy.changed_point_coverage_percent
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>已有注入数据的事件路径验证</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1300px;margin:35px auto;padding:20px;line-height:1.7}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}th{{background:#eef3f7}}</style>
<h1>已有八月注入数据的配对验证</h1><p>共 {len(audit)} 个文件，基线固定原始七月，不重新训练，不读取八月测试分布进行校准。只对包含已知功率修改的日期做原始/注入配对评分。不可评分通道保留原状态，不作为漏检或正常。</p>
<p>带转变限制的路径将实际改变点覆盖率从 {legacy.changed_point_coverage_percent:.2f}% 提高到 {guarded.changed_point_coverage_percent:.2f}%（增加 {gain:.2f} 个百分点）。连续改变区段重叠从 {int(legacy.injected_overlapped_runs)}/{int(legacy.eligible_changed_runs)} 提高到 {int(guarded.injected_overlapped_runs)}/{int(guarded.eligible_changed_runs)}。这证明该组已知修改上的覆盖改善，不证明真实故障精确率提高。</p>
<p>根据 pRealKw 与 pRealKw_original 的差异标记实际改变的五分钟点，并与独立原始文件核对。对每个文件构造隔离重放：只在已知改变点复制注入值，区段外全部保留原值。正向/反向功率的区段外修改不进入基线或重放。原始和注入文件未写入。</p>
<h2>已知改变点与连续改变区段</h2>{table}
<h2>原始背景与新增覆盖</h2>{background.to_html(index=False)}
<h2>不可评分通道也保留在全体分母中</h2>{availability.to_html(index=False,float_format=lambda value:f"{value:.2f}")}
<p>覆盖率只描述这组人工修改点，不代表真实故障召回率。相邻改变点构成一个 changed run，不等同于人工设计的一个完整案例。original_overlap 表示原背景已有事件；新覆盖单独列出。未改变的点没有已确认正常标签，因此不计算精确率、F1或真实误报率。newly_covered_unchanged_5m_points 反映新增背景覆盖/时间扩散。</p>
<h2>输入与准入审计</h2>{audit[["series_id","preparation_status","known_changed_5m_points","outside_field_changes"]].to_html(index=False)}
<p><a href="input_audit.csv">完整输入审计</a> · <a href="aggregate_comparison.csv">汇总</a> · <a href="changed_runs.csv">逐连续改变区段</a> · <a href="paired_summary.csv">通道配对结果</a> · <a href="day_status.csv">配对日状态</a></p></html>'''
    (output/"report.html").write_text(report,encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="runs/analysis/matrix_profile_existing_injections_20261005")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args(argv)
    output=baseline.local.resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    output.mkdir(parents=True,exist_ok=True)
    repository=json.loads((ROOT/"methods/matrix_profile/src/meter_repository.json").read_text())
    tasks=[]
    for path in sorted((ROOT/"data/injected_meters").glob("*.csv")):
        if path.name.startswith("_"): continue
        device,channel=path.stem.rsplit("_",1)
        context=baseline._context_from_metadata(repository,device,int(channel))
        if context is None: raise ValueError(f"No metadata: {path.stem}")
        tasks.append((path.stem,context,str(ROOT/"data/processed/meter_csvs"/path.name),str(path),str(output)))
    audits,summaries,segments,statuses=[],[],[],[]
    print(f"Existing injection validation: {len(tasks)} paired files; frozen original July",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,task):task[0] for task in tasks}
        for number,future in enumerate(as_completed(pending),1):
            audit,summary,runs,days=future.result()
            audits.append(audit); summaries.extend(summary); segments.extend(runs); statuses.extend(days)
            print(f"Completed {number}/{len(tasks)}: {pending[future]} ({audit['preparation_status']})",flush=True)
    audit=pd.DataFrame(audits); summary=pd.DataFrame(summaries); runs=pd.DataFrame(segments)
    audit.to_csv(output/"input_audit.csv",index=False)
    summary.to_csv(output/"paired_summary.csv",index=False)
    runs.to_csv(output/"changed_runs.csv",index=False)
    pd.DataFrame(statuses).to_csv(output/"day_status.csv",index=False)
    aggregate=summary.groupby("event_policy").sum(numeric_only=True).reset_index()
    aggregate["changed_point_coverage_percent"]=100*aggregate.injected_covered_changed_5m_points/aggregate.eligible_changed_5m_points
    aggregate["all_declared_changed_5m_points"]=int(audit.known_changed_5m_points.sum())
    aggregate["unscored_changed_5m_points"]=aggregate.all_declared_changed_5m_points-aggregate.eligible_changed_5m_points
    aggregate["all_changed_point_coverage_percent"]=100*aggregate.injected_covered_changed_5m_points/aggregate.all_declared_changed_5m_points
    eligible=runs.loc[runs.eligible]
    for index,row in aggregate.iterrows():
        group=eligible.loc[eligible.event_policy.eq(row.event_policy)]
        aggregate.loc[index,"eligible_changed_runs"]=len(group)
        aggregate.loc[index,"original_overlapped_runs"]=int(group.original_overlap.sum())
        aggregate.loc[index,"injected_overlapped_runs"]=int(group.injected_overlap.sum())
        aggregate.loc[index,"newly_overlapped_runs"]=int((group.injected_overlap.astype(bool)&~group.original_overlap.astype(bool)).sum())
    aggregate.to_csv(output/"aggregate_comparison.csv",index=False)
    for row in audit.itertuples():
        assert baseline.digest(ROOT/"data/injected_meters"/f"{row.series_id}.csv")==row.injected_sha256
        assert baseline.digest(ROOT/"data/processed/meter_csvs"/f"{row.series_id}.csv")==row.original_sha256
    metadata=dict(analysis_only=True,source_files_unchanged=True,baseline_start="2026-07-01",baseline_end="2026-07-31",
                  detection_start="2026-08-01",detection_end="2026-08-31",input_files=len(audit),
                  status_counts=audit.preparation_status.value_counts().to_dict(),
                  replay="Known pRealKw change points only; independent originals outside edits",
                  experiment_code_sha256=baseline.digest(__file__),
                  policy_code_sha256=baseline.digest(ROOT/"methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py"),
                  windows_hours=[2,6,24],score_threshold=.995)
    (output/"experiment_summary.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    render_report(output,audit,aggregate,metadata)
    print(aggregate.to_string(index=False),flush=True)
    print(f"Report: {output/'report.html'}",flush=True)


if __name__=="__main__":
    main()
