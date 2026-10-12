"""Isolated baseline sensitivity pilot using the existing multiscale detector.

No production code, inputs, thresholds or September results are overwritten.
All references end before September; 2h/6h/24h windows are unchanged.
The exact multi-sensor freeze rule identifies SUSPECTED stale readings, not
confirmed communication faults. Controlled probes are sensitivity checks,
not the user's injected dataset and not a real-world accuracy benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date
import hashlib
import html
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/scripts"))
import run_matrix_profile_local as local
import numpy as np
import pandas as pd
from anomalies.multiscale.aggregate import _coalesce_refined_events
from anomalies.multiscale.cloud_batch import _context_from_metadata
from anomalies.multiscale.data_quality import detect_flatline_events, overlaps_quality_event
from anomalies.multiscale.detector import _interval_score_records
from anomalies.multiscale.events import build_day_events
from anomalies.multiscale.models import ChannelStatus, Confidence
from anomalies.multiscale.output import EVENT_COLUMNS, INTERVAL_SCORE_COLUMNS, _csv_value, _upsample_interval_frame
from anomalies.multiscale.preprocessing import _normalise_index, prepare_channel
from anomalies.multiscale.scoring import score_day

SENSORS = ["pRealKw", "vRMSMin", "vRMSMax", "iRMSMin", "iRMSMax"]
VARIANTS = {
    "raw_july": (date(2026, 7, 1), date(2026, 7, 31), False),
    "clean_july": (date(2026, 7, 1), date(2026, 7, 31), True),
    "raw_july_august": (date(2026, 7, 1), date(2026, 8, 31), False),
    "clean_july_august": (date(2026, 7, 1), date(2026, 8, 31), True),
    "clean_august": (date(2026, 8, 1), date(2026, 8, 31), True),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sensor_freezes(frame, minimum_points=24, tolerance=1e-9):
    """Require all five sensors finite and unchanged across contiguous 5m rows.

    Excludes near-zero power to avoid treating normal night shutdown/solar
    inactivity as frozen data. Cross-midnight runs remain one candidate.
    """
    mask = pd.Series(False, index=frame.index)
    records = []
    if not all(column in frame for column in SENSORS) or frame.empty:
        return mask, records
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Freeze input must have a sorted unique timestamp index")
    values = frame[SENSORS].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    adjacent = np.diff(frame.index.asi8) == pd.Timedelta(minutes=5).value
    same = np.isclose(values[1:], values[:-1], atol=tolerance, rtol=0).all(axis=1)
    boundaries = np.r_[0, np.flatnonzero(~(adjacent & same & finite[1:] & finite[:-1])) + 1, len(frame)]
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        segment = values[first:last]
        if (last-first < minimum_points or not finite[first:last].all()
                or abs(segment[0, 0]) <= .05 or np.ptp(segment, axis=0).max() > tolerance):
            continue
        mask.iloc[first:last] = True
        records.append(dict(start=frame.index[first], end=frame.index[last-1]+pd.Timedelta(minutes=5),
                            points=last-first, duration_minutes=(last-first)*5,
                            pRealKw=float(segment[0, 0]), issue="suspected_multisensor_freeze"))
    return mask, records


def mask_history(frame, config, freeze_mask):
    cleaned = frame.copy()
    history = (cleaned.index.date >= config.baseline_start) & (cleaned.index.date <= config.baseline_end)
    affected = history & freeze_mask.to_numpy()
    cleaned.loc[affected, :] = np.nan
    # Never interpolate these long spans or change target observations.
    pd.testing.assert_frame_equal(cleaned.loc[~history], frame.loc[~history])
    return cleaned, int(affected.sum())


def freeze_overlap(mask, intervals):
    # Empty resampling bins introduce NaN and can promote booleans to floats.
    # Normalize explicitly before boolean inversion/combination.
    return mask.resample("30min").max().reindex(intervals).fillna(False).astype(bool).to_numpy()


def choose_channels(distribution, audit):
    selected = {}
    for category in ("Grid connect", "Air conditioning", "Lighting & Power", "Solar generation"):
        group = distribution.loc[distribution.category_name.eq(category)].sort_values(["flag_rate", "series_id"])
        if group.empty:
            continue
        for row, reason in ((group.iloc[-1], "highest flag rate in category"),
                            (group.iloc[len(group)//2], "middle-ranked flag rate in category")):
            selected[row.series_id] = reason
    for row in audit.sort_values(["july_frozen_points", "series_id"], ascending=[False, True]).head(2).itertuples():
        if row.july_frozen_points:
            selected.setdefault(row.series_id, "largest suspected July freeze exposure")
    for key in ("DD43710150291_0", "DD63710147032_4"):
        if distribution.series_id.eq(key).any():
            selected.setdefault(key, "previous review / low-load guard case")
    return selected


def event_frame(events):
    rows = [{key: _csv_value(value) for key, value in event.to_dict().items()} for event in events]
    return _coalesce_refined_events(pd.DataFrame(rows, columns=EVENT_COLUMNS))


def day_events(scores, target, prepared, context, config):
    baseline = float(pd.concat([day.pRealKw for day in prepared.history_5m.values()]).median())
    quality = detect_flatline_events(scores.day, target, prepared.history_5m, context)
    behavior = build_day_events(scores, target, baseline, context, config)
    return quality + [event for event in behavior if not overlaps_quality_event(event, quality)]


def controlled_probe(target, kind, scale):
    """Change only a known interval; preserve every field outside that interval."""
    result = target.copy()
    start = target.index[144]  # local noon, independent of September score rank
    duration = {"spike_15m": 15, "level_shift_6h": 360, "multisensor_freeze_4h": 240}[kind]
    end = start + pd.Timedelta(minutes=duration)
    affected = (target.index >= start) & (target.index < end)
    if kind == "multisensor_freeze_4h":
        if not np.isfinite(target.loc[start, SENSORS].to_numpy(dtype=float)).all() or abs(target.loc[start, "pRealKw"]) <= .05:
            return None, start, end
        result.loc[affected, :] = target.loc[start].to_numpy()
    else:
        amplitude = 4*scale if kind == "spike_15m" else 2*scale
        power = target.loc[affected, "pRealKw"] + amplitude
        result.loc[affected, "pRealKw"] = power
        for column, values in (("pRealPositiveKw", power.clip(lower=0)),
                               ("pRealNegativeKw", -power.clip(upper=0))):
            if column in result:
                result.loc[affected, column] = values
        # Deterministic correlated-current stress, not an electrical simulation.
        if "iRMSMax" in result:
            original = target.loc[affected, "pRealKw"].abs().clip(lower=.05)
            ratio = (power.abs()/original).clip(upper=20)
            result.loc[affected, "iRMSMax"] = target.loc[affected, "iRMSMax"]*ratio
            if "iRMSMin" in result:
                result.loc[affected, "iRMSMin"] = target.loc[affected, "iRMSMin"]*ratio
    pd.testing.assert_frame_equal(result.loc[~affected], target.loc[~affected])
    return result, start, end


def trial_worker(task):
    key, context, input_path, original_path, output_path = task
    original = _normalise_index(pd.read_csv(original_path), "Australia/Sydney")
    combined = _normalise_index(pd.read_csv(input_path), "Australia/Sydney")
    # Load July and September directly from the exact production input. Avoid
    # adding a CSV serialization round trip to the reproduction control.
    frame = pd.concat([original, combined.loc[combined.index.month == 8]]).sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate trial input timestamps: {key}")
    freeze_mask, _ = sensor_freezes(frame)
    summaries, probes, statuses = [], [], []
    for variant, (first, last, clean) in VARIANTS.items():
        config = local.DetectionConfig(baseline_start=first, baseline_end=last,
                                       detection_start=date(2026,9,1), detection_end=date(2026,9,30))
        source, removed = mask_history(frame, config, freeze_mask) if clean else (frame.copy(), 0)
        preparation = prepare_channel(source, config, context.category_name)
        prepared = preparation.data
        rows, events, candidates, day_status = [], [], [], []
        cache, original_days = {}, {}
        if prepared is not None:
            for day, target in sorted(prepared.target_5m.items()):
                scores = score_day(prepared, day, context, config, cache)
                original_days[day] = scores
                rows.extend(_interval_score_records(scores, context, config))
                found = day_events(scores, target, prepared, context, config)
                events.extend(event for event in found if event.confidence is not Confidence.CANDIDATE)
                candidates.extend(event for event in found if event.confidence is Confidence.CANDIDATE)
            valid_days = len(prepared.history_5m)
            july_days = sum(day.month == 7 for day in prepared.history_5m)
            august_days = valid_days-july_days
        else:
            valid_days = july_days = august_days = 0
        for day in pd.date_range("2026-09-01", "2026-09-30").date:
            missing_status = ChannelStatus.MISSING_TARGET.value if prepared is not None else preparation.status.value
            day_status.append(dict(date=str(day), status="scored" if day in original_days else missing_status))
        native = local.complete_grid(pd.DataFrame(rows, columns=INTERVAL_SCORE_COLUMNS), key,
                                     pd.DataFrame(day_status), config)
        native["variant"] = variant
        native["suspected_freeze_overlap"] = freeze_overlap(freeze_mask,native.interval_start)
        native["load_behavior_flag"] = native.is_anomaly & ~native.suspected_freeze_overlap
        native["baseline_cleaning"] = "exclude_suspected_multisensor_freezes" if clean else "none"
        output = Path(output_path)/variant/key
        output.mkdir(parents=True, exist_ok=True)
        native.to_csv(output/"scores_30min.csv", index=False)
        _upsample_interval_frame(native).to_csv(output/"scores_15min.csv", index=False)
        formal = event_frame(events)
        formal.to_csv(output/"events.csv", index=False)
        event_frame(candidates).to_csv(output/"candidates.csv", index=False)
        dq = int(formal.event_type.eq("data_quality").sum())
        summaries.append(dict(series_id=key, variant=variant, category_name=context.category_name,
                              preparation_status=preparation.status.value, reason=preparation.reason,
                              history_days=valid_days, july_days=july_days, august_days=august_days,
                              baseline_removed_5m_points=removed, available_intervals=int(native.available.sum()),
                              flagged_intervals=int(native.is_anomaly.sum()),
                              load_behavior_flagged_intervals=int(native.load_behavior_flag.sum()),
                              formal_events=len(formal), data_quality_events=dq, behavior_events=len(formal)-dq))
        statuses.extend(dict(series_id=key, variant=variant, **row) for row in day_status)
        if prepared is None:
            continue
        probe_day = date(2026,9,15) if date(2026,9,15) in prepared.target_5m else min(prepared.target_5m)
        target = prepared.target_5m[probe_day]
        # Use the SAME July-derived amplitude for every variant, never a
        # variant-specific baseline that would change the stress input.
        july_power = frame.loc[(frame.index.date >= date(2026,7,1)) & (frame.index.date <= date(2026,7,31)), "pRealKw"]
        scale = max(1., float(july_power.abs().quantile(.95)))
        before = pd.DataFrame(_interval_score_records(original_days[probe_day], context, config))
        for kind in ("spike_15m", "level_shift_6h", "multisensor_freeze_4h"):
            modified, start, end = controlled_probe(target, kind, scale)
            if modified is None or modified.loc[(modified.index>=start)&(modified.index<end)].isna().any().any():
                probes.append(dict(series_id=key, variant=variant, kind=kind, eligible=False,
                                   reason="Probe interval missing values or freeze starts at near-zero power"))
                continue
            probe = replace(prepared, target_5m={**prepared.target_5m, probe_day:modified},
                            target_30m={**prepared.target_30m, probe_day:resample_probe(modified, config)})
            scores = score_day(probe, probe_day, context, config, cache)
            after = pd.DataFrame(_interval_score_records(scores, context, config))
            inside = (after.interval_start < end) & (after.interval_end > start)
            new = after.is_anomaly & ~before.is_anomaly
            detected = [event for event in day_events(scores, modified, probe, context, config)
                        if event.confidence is not Confidence.CANDIDATE and event.start_time < end and event.end_time > start]
            probe_mask, _ = sensor_freezes(modified)
            probes.append(dict(series_id=key, variant=variant, kind=kind, eligible=True,
                               start=start, end=end, amplitude_scale_kw=scale,
                               background_flagged_inside_bins=int((before.is_anomaly & inside).sum()),
                               after_flagged_inside_bins=int((after.is_anomaly & inside).sum()),
                               newly_flagged_inside_bins=int((new & inside).sum()),
                               newly_flagged_outside_bins=int((new & ~inside).sum()),
                               max_score_increase=float((after.max_score-before.max_score).loc[inside].max()),
                               formal_event_overlap=bool(detected),
                               multisensor_quality_overlap=bool(probe_mask.loc[(modified.index>=start)&(modified.index<end)].any())))
    checkpoint = Path(output_path).parent/"completed_channels"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint/f"{key}.json").write_text(json.dumps(dict(summaries=summaries,probes=probes,statuses=statuses),default=str),encoding="utf-8")
    return summaries, probes, statuses


def resample_probe(frame, config):
    from anomalies.multiscale.preprocessing import _resample_day
    return _resample_day(frame, config)


def paired_probe_comparisons(probes):
    """Pair identical eligible disturbances; never label overlap as recall."""
    eligible = probes.loc[probes.eligible.fillna(False)].copy()
    keys = ["series_id","kind","start","end"]
    baseline = eligible.loc[eligible.variant.eq("raw_july")]
    rows = []
    for variant in VARIANTS:
        if variant == "raw_july":
            continue
        paired = baseline.merge(eligible.loc[eligible.variant.eq(variant)],on=keys,suffixes=("_old","_new"),validate="one_to_one")
        for kind, group in paired.groupby("kind"):
            lost = group.formal_event_overlap_old & ~group.formal_event_overlap_new
            gained = ~group.formal_event_overlap_old & group.formal_event_overlap_new
            rows.append(dict(variant=variant,kind=kind,identical_eligible_cases=len(group),
                             old_formal_overlaps=int(group.formal_event_overlap_old.sum()),
                             new_formal_overlaps=int(group.formal_event_overlap_new.sum()),
                             lost_formal_overlaps=int(lost.sum()),gained_formal_overlaps=int(gained.sum()),
                             old_score_overlap_cases=int(group.after_flagged_inside_bins_old.gt(0).sum()),
                             new_score_overlap_cases=int(group.after_flagged_inside_bins_new.gt(0).sum()),
                             lost_case_ids="|".join(sorted(group.loc[lost,"series_id"]))))
    return pd.DataFrame(rows)


def render_report(output, summary, aggregate, probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"raw_july":"July raw", "clean_july":"July cleaned", "raw_july_august":"Jul–Aug raw",
              "clean_july_august":"Jul–Aug cleaned", "clean_august":"August cleaned"}
    fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].bar([labels[x] for x in aggregate.variant], aggregate.common_flag_rate*100, color="#366c99")
    axes[0].set(title="Threshold rate on the SAME available intervals", ylabel="Flagged native 30m intervals (%)")
    axes[1].bar([labels[x] for x in aggregate.variant], aggregate.available_intervals, color="#598856")
    axes[1].set(title="Coverage must not disappear into fewer flags", ylabel="Available native 30m intervals")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Multiscale Matrix Profile baseline sensitivity pilot", fontsize=15)
    fig.savefig(output/"pilot_comparison.png", dpi=160)
    plt.close(fig)
    eligible = probes.loc[probes.eligible.fillna(False)] if len(probes) else probes
    probe_table = (eligible.groupby(["variant","kind"]).agg(cases=("series_id","size"),
                   formal_overlap=("formal_event_overlap","sum"),
                   background_flagged_bins=("background_flagged_inside_bins","sum"),
                   newly_flagged_bins=("newly_flagged_inside_bins","sum"),
                   newly_flagged_outside_bins=("newly_flagged_outside_bins","sum"))
                   .reset_index()) if len(eligible) else pd.DataFrame()
    probe_table.to_csv(output/"probe_summary.csv", index=False)
    paired = paired_probe_comparisons(probes)
    paired.to_csv(output/"paired_probe_comparison.csv",index=False)
    display = aggregate[["variant","common_flag_rate","common_flag_reduction_percent","common_formal_events","common_behavior_events","unavailable_channels","available_intervals"]].copy()
    display["common_flag_rate"] *= 100
    display.columns = ["基线","共同区间超限率 (%)","超限区间减少 (%)","共同覆盖日正式事件","其中行为事件","不可评分通道","全部有效区间"]
    display["基线"] = display["基线"].map(labels)
    reduced = aggregate.loc[aggregate.variant.eq("clean_july_august"),"common_flag_reduction_percent"].iloc[0]
    cleaned_july = aggregate.loc[aggregate.variant.eq("clean_july"),"common_flag_reduction_percent"].iloc[0]
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多尺度基线对照试验</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1250px;margin:35px auto;padding:0 25px;line-height:1.7;color:#24313c}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef3f7}}img{{width:100%}}h2{{margin-top:30px}}</style>
<h1>九月多尺度 Matrix Profile 基线对照试验</h1>
<p><strong>结论：参考历史的选择明显影响告警数量，但暂不应替换默认基线。</strong>清洗后的七至八月历史在共同可评分区间上减少了 {reduced:.2f}% 的超限区间，但部分相同受控扰动不再生成与注入区段重叠的正式事件。减少告警不等于识别能力提高。仅清洗七月没有减量，变化为 {cleaned_july:.2f}%（负值表示增加）。</p>
<p>这是有意选取高告警、中等告警和历史冻值通道的小批量敏感性实验，不代表全体通道的总体改善。全部模型仍使用原有 2/6/24 小时窗口、原生 30 分钟评分和明确标记为复制的 15 分钟输出。全部历史参考及校准截止于 2026-08-31，不使用九月全局分布。</p>
<p>原始 GitHub 备份：reshape1 / 3ae351b。生产分数、事件和原始输入文件均未覆盖。原始七月试验逐区间复现检查及生产文件 SHA256 检查：{html.escape(str(summary['validation']))}。</p>
<h2>同一可评分区间的结果</h2><p>共同可评分区间数：{summary['common_available_intervals']:,}。改变基线后出现的不可评分通道或日期另列，不能算作异常减少。表中的正式事件只统计所有变体都完整可评分的通道日；全部事件数量另存于 aggregate_comparison.csv。</p>
{display.to_html(index=False, float_format=lambda x:f'{x:.2f}')}
<img src="pilot_comparison.png" alt="共同覆盖分数与可评分覆盖量对照">
<h2>清洗规则及局限</h2><p>仅把功率、电压最小/最大值、电流最小/最大值五列同步不变至少两小时且非近零功率的区间标为疑似冻值。它不是通讯故障的人工确认，也不是 DOCX 报告原算法的复现。清洗只屏蔽历史参考中的这些行，目标观测保持原样。仍保留 95% 有效日和至少 24 个历史日要求，未降低门槛凑足样本。数据质量候选另存，没有把它们自动认定正常。</p>
<h2>受控扰动敏感性检查</h2><p>本次未找到仓库内现成的 injected_meters 数据，因此另在真实九月背景的内存副本中构造 15 分钟尖峰、6 小时水平变化和 4 小时多测量列冻结。仅修改已知注入区段，每个变体使用完全相同的扰动。背景可能已超限；事件重叠不能作为召回率，新增超限与背景超限分别报告。未证明误报率、精确率或真实故障召回率改善。</p>
{probe_table.to_html(index=False)}
<h2>相同受控案例的配对结果</h2><p>下面只比较通道、扰动类型和注入时间完全相同的有效案例。正式事件重叠减少是敏感性风险提示，不是已证明的真实故障漏检率；原背景可能已有告警。</p>
{paired[["variant","kind","identical_eligible_cases","old_formal_overlaps","new_formal_overlaps","lost_formal_overlaps","old_score_overlap_cases","new_score_overlap_cases"]].to_html(index=False)}
<h2>下一步优先核查</h2><p>一、现有 build_day_events 的候选掩码主要由 2/6 小时形态分数生成，幅值、日形态和变点主要在候选之后确认。受控尖峰和水平变化仍产生区间超限，但未必进入正式事件；应先明确分数候选与正式事件的业务用途，再验证独立幅值/变点候选路径。二、两个试验通道在扩大或改用八月历史后被 constant_signal 准入规则排除；需要检查它们是否为稀疏启停负荷，不能把无分数当正常。三、更长历史可能同时吸收正常模式和持续异常；在原注入数据或人工标签上验证之前，不把最低告警率当作最优基线。</p>
<h2>详细结果</h2><p><a href="pilot_channels.csv">固定通道选择</a> · <a href="baseline_quality_audit.csv">全部原已评分通道的七月与九月冻值检查</a> · <a href="suspected_freeze_intervals.csv">疑似冻值区段</a> · <a href="trial_summary.csv">通道级结果与不可评分原因</a> · <a href="probes.csv">逐受控扰动结果</a> · <a href="day_status.csv">每日覆盖状态</a></p></html>'''
    (output/"report.html").write_text(report, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/matrix_profile/results/september_fixed_july_local")
    parser.add_argument("--distribution-dir", default="runs/analysis/matrix_profile_september_distribution")
    parser.add_argument("--output-dir", default="runs/analysis/matrix_profile_baseline_pilot_20261005_v2")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    run, output = local.resolve(args.run_dir), local.resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new isolated output directory; existing experiment results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run/"run_metadata.json").read_text())
    if meta["baseline_start"] != "2026-07-01" or meta["detection_start"] != "2026-09-01":
        raise ValueError("This pilot requires the original fixed-July September run")
    originals = {name:digest(run/name) for name in ("ALL_SCORES_30MIN.csv","ALL_SCORES_15MIN.csv","ALL_EVENTS.csv")}
    inputs = Path(meta["prepared_input_dir"])
    repository = json.loads((ROOT/"methods/matrix_profile/src/meter_repository.json").read_text())
    distribution = pd.read_csv(local.resolve(args.distribution_dir)/"by_channel.csv")
    audit, freezes = [], []
    for position, key in enumerate(sorted(distribution.series_id),1):
        frame = _normalise_index(pd.read_csv(inputs/f"{key}.csv", usecols=["timestamp"]+SENSORS),meta["timezone"])
        mask, records = sensor_freezes(frame)
        months = frame.index.month
        july, september = months==7, months==9
        audit.append(dict(series_id=key, july_frozen_points=int(mask.loc[july].sum()),
                          september_frozen_points=int(mask.loc[september].sum()),
                          july_days_with_freeze=int(pd.Index(frame.index.date[july & mask.to_numpy()]).nunique()),
                          september_days_with_freeze=int(pd.Index(frame.index.date[september & mask.to_numpy()]).nunique())))
        freezes.extend(dict(series_id=key,**row) for row in records)
        if position%40==0 or position==len(distribution):
            print(f"Freeze audit {position}/{len(distribution)} channels",flush=True)
    audit = pd.DataFrame(audit)
    audit.to_csv(output/"baseline_quality_audit.csv",index=False)
    pd.DataFrame(freezes,columns=["series_id","start","end","points","duration_minutes","pRealKw","issue"]).to_csv(output/"suspected_freeze_intervals.csv",index=False)
    selected = choose_channels(distribution,audit)
    selection = distribution.loc[distribution.series_id.isin(selected)].copy()
    selection["selection_reason"] = selection.series_id.map(selected)
    selection.to_csv(output/"pilot_channels.csv",index=False)
    prepared_dir = output/"prepared_inputs"
    prepared_dir.mkdir()
    tasks = []
    with local.ArchiveReader(meta["source"]) as reader:
        indexed = local.index_target(reader,date(2026,8,1),date(2026,9,1))
        for key in sorted(selected):
            frame = local.normalize(pd.read_csv(inputs/f"{key}.csv"),meta["timezone"])
            parts = [frame]
            for day,name in indexed.get(key,[]):
                with reader.open(name) as stream:
                    extra = local.normalize(pd.read_csv(stream),meta["timezone"])
                if not extra.timestamp.dt.date.eq(day).all():
                    raise ValueError(f"August folder/date mismatch: {name}")
                parts.append(extra)
            frame = local.normalize(pd.concat(parts,ignore_index=True),meta["timezone"])
            path = prepared_dir/f"{key}.csv"
            frame.to_csv(path,index=False)
            device,channel = key.rsplit("_",1)
            context = _context_from_metadata(repository,device,int(channel))
            if context is None:
                raise ValueError(f"Missing context: {key}")
            tasks.append((key,context,str(path),str(inputs/f"{key}.csv"),str(output/"variants")))
    print(f"Pilot: {len(tasks)} channels x {len(VARIANTS)} baseline variants",flush=True)
    summaries, probes, day_status = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(trial_worker,task):task[0] for task in tasks}
        for position,future in enumerate(as_completed(pending),1):
            result,checks,statuses = future.result()
            summaries.extend(result); probes.extend(checks); day_status.extend(statuses)
            print(f"Completed {position}/{len(tasks)}: {pending[future]}",flush=True)
    summary_frame,probe_frame = pd.DataFrame(summaries),pd.DataFrame(probes)
    summary_frame.to_csv(output/"trial_summary.csv",index=False)
    probe_frame.to_csv(output/"probes.csv",index=False)
    pd.DataFrame(day_status).to_csv(output/"day_status.csv",index=False)
    pieces = [pd.read_csv(output/"variants"/variant/key/"scores_30min.csv") for key in selected for variant in VARIANTS]
    scores = pd.concat(pieces,ignore_index=True)
    common = scores.groupby(["series_id","interval_start"]).available.all()
    common_keys = common[common].index
    common_days = pd.DataFrame(list(common_keys),columns=["series_id","interval_start"])
    common_days["date"] = common_days.interval_start.str.slice(0,10)
    fully_common_days = set(common_days.groupby(["series_id","date"]).size().loc[lambda x:x.eq(48)].index)
    original = pd.read_csv(run/"ALL_SCORES_30MIN.csv",usecols=["series_id","interval_start","available","max_score"])
    reference = original.set_index(["series_id","interval_start"])
    raw = scores.loc[scores.variant.eq("raw_july")].set_index(["series_id","interval_start"])
    expected = reference.reindex(raw.index)
    np.testing.assert_array_equal(raw.available,expected.available)
    np.testing.assert_allclose(raw.max_score,expected.max_score,atol=1e-8,rtol=0,equal_nan=True)
    original_events = pd.read_csv(run/"ALL_EVENTS.csv")
    for key in selected:
        device,channel = key.rsplit("_",1)
        ids = set(original_events.loc[original_events.device_id.eq(device)&original_events.channel_idx.eq(int(channel)),"event_id"])
        reproduction = pd.read_csv(output/"variants/raw_july"/key/"events.csv")
        if set(reproduction.event_id)!=ids:
            raise AssertionError(f"Original formal event IDs not reproduced: {key}")
    rows = []
    for variant in VARIANTS:
        subset = scores.loc[scores.variant.eq(variant)]
        matched = subset.set_index(["series_id","interval_start"]).reindex(common_keys)
        ch = summary_frame.loc[summary_frame.variant.eq(variant)]
        common_events = []
        for key in selected:
            frame = pd.read_csv(output/"variants"/variant/key/"events.csv")
            common_events.extend(row for row in frame.to_dict("records") if (key,str(row["start_time"])[:10]) in fully_common_days)
        common_quality = sum(row["event_type"]=="data_quality" for row in common_events)
        rows.append(dict(variant=variant,available_intervals=int(subset.available.sum()),
                         unavailable_channels=int(ch.available_intervals.eq(0).sum()),
                         flagged_intervals=int(subset.is_anomaly.sum()),
                         common_available_intervals=len(matched),common_flagged_intervals=int(matched.is_anomaly.sum()),
                         common_flag_rate=float(matched.is_anomaly.mean()),
                         common_load_flagged_intervals=int(matched.load_behavior_flag.sum()),
                         common_formal_events=len(common_events),common_behavior_events=len(common_events)-common_quality,
                         formal_events=int(ch.formal_events.sum()),behavior_events=int(ch.behavior_events.sum()),
                         data_quality_events=int(ch.data_quality_events.sum())))
    aggregate = pd.DataFrame(rows)
    base = int(aggregate.iloc[0].common_flagged_intervals)
    aggregate["common_flag_reduction_percent"] = (1-aggregate.common_flagged_intervals/base)*100 if base else np.nan
    aggregate.to_csv(output/"aggregate_comparison.csv",index=False)
    if {name:digest(run/name) for name in originals} != originals:
        raise AssertionError("Production output changed during experiment")
    summary = dict(analysis_only=True,production_changed=False,backup_commit="3ae351b",
                   channels=list(selected),variants=VARIANTS,original_sha256=originals,
                   experiment_code_sha256=digest(__file__),
                   production_code_sha256={str(path.relative_to(ROOT)):digest(path) for path in sorted(local.MP_SRC.rglob("*.py"))},
                   pilot_input_sha256={key:digest(prepared_dir/f"{key}.csv") for key in selected},
                   common_available_intervals=len(common_keys),validation="Original raw-July scores and formal event IDs reproduced; production SHA256 unchanged",
                   audit_channels=len(audit),july_channels_with_suspected_freeze=int(audit.july_frozen_points.gt(0).sum()),
                   september_channels_with_suspected_freeze=int(audit.september_frozen_points.gt(0).sum()),
                   controlled_probe_cases=int(probe_frame.eligible.fillna(False).sum()),
                   baseline_cutoff="2026-08-31",score_threshold_unchanged=.995,windows_hours=[2,6,24],
                   limitations=["Purposive pilot selection, not fleet-wide effect", "No confirmed normal/fault labels", "Controlled probes are not real-world accuracy", "Freeze rule is not the source DOCX algorithm"])
    (output/"experiment_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    render_report(output,summary,aggregate,probe_frame)
    print(aggregate.to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__ == "__main__":
    main()
