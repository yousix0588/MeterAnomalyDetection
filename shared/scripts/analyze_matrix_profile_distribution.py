"""Read-only September diagnostics and reporting-rule sensitivity analysis.

Writes a separate report directory. Does not alter detection scores, event
files, baseline calibration or production thresholds. Uses native 30m rows
so the repeated 15m rows are not treated as independent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "methods/matrix_profile/src"))
from anomalies.multiscale.scoring import select_reference_days


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def merge_reporting_units(events, gap_minutes=0):
    """Group same-channel/type/direction overlap/adjacency; retain original IDs.

    Direction refers only to the stored event mean vs stored July median,
    not a claim about fault mechanism. This is a reporting grouping only.
    """
    events = events.copy()
    difference = events.observed_mean_kw - events.baseline_median_kw
    events["direction"] = np.where(difference.isna(), "unknown", np.where(difference >= 0, "up", "down"))
    rows = []
    for key, group in events.groupby(["series_id", "event_type", "direction"], sort=True):
        current = None
        for event in group.sort_values("start").itertuples():
            if current is not None and event.start <= current["end"] + pd.Timedelta(minutes=gap_minutes):
                current["end"] = max(current["end"], event.end)
                current["event_ids"].append(event.event_id)
                current["confidences"].add(event.confidence)
            else:
                if current is not None:
                    rows.append(current)
                current = dict(series_id=key[0], event_type=key[1], direction=key[2],
                               start=event.start, end=event.end, event_ids=[event.event_id],
                               confidences={event.confidence})
        if current is not None:
            rows.append(current)
    for row in rows:
        row["original_event_count"] = len(row["event_ids"])
        row["event_ids"] = json.dumps(row["event_ids"])
        row["confidences"] = "|".join(sorted(row["confidences"]))
        row["span_minutes"] = (row["end"]-row["start"]).total_seconds()/60
    return pd.DataFrame(rows).sort_values(["series_id", "start"]).reset_index(drop=True)


def vector_same_slot(reference, target):
    """Exact vectorization of the existing v2 disjoint-midrank magnitude ECDF."""
    center = np.nanmedian(reference, axis=0)
    reference_deviation = np.abs(reference-center)
    target_deviation = np.abs(target-center)
    finite = np.isfinite(reference_deviation)
    tied = np.isclose(reference_deviation, target_deviation, rtol=1e-9, atol=1e-12) & finite
    lower = (reference_deviation < target_deviation) & ~tied & finite
    counts = finite.sum(axis=0)
    result = np.divide(lower.sum(axis=0)+.5*tied.sum(axis=0), counts,
                       out=np.full(288, np.nan), where=counts>0)
    result[~np.isfinite(target_deviation)] = np.nan
    return result


def magnitude_attribution(native, inputs, metadata, output):
    """Reconstruct magnitude from frozen July only; validate against saved mean.

    The saved mean/std plus reconstructed magnitude identify the two remaining
    shape scores up to permutation. Only symmetric AND/OR attribution is used.
    No new MP join, training or test-distribution calibration is performed.
    """
    repository = json.loads((ROOT/"methods/matrix_profile/src/meter_repository.json").read_text())
    baseline_start = pd.Timestamp(metadata["baseline_start"]).date()
    baseline_end = pd.Timestamp(metadata["baseline_end"]).date()
    timezone = metadata["timezone"]
    parts, drift = [], []
    keys = sorted(native.series_id.unique())
    for position, key in enumerate(keys, 1):
        raw = pd.read_csv(inputs/f"{key}.csv", usecols=["timestamp", "pRealKw"])
        raw.index = pd.to_datetime(raw.pop("timestamp"), utc=True).dt.tz_convert(timezone)
        days = {}
        for day, group in raw.groupby(raw.index.date):
            grid = pd.date_range(pd.Timestamp(day, tz=timezone), periods=288, freq="5min")
            values = group.pRealKw.reindex(grid).to_numpy(dtype=float)
            if np.isfinite(values).sum() >= 274:
                days[day] = values
        history = {day: values for day, values in days.items() if baseline_start <= day <= baseline_end}
        device, channel = key.rsplit("_", 1)
        category = repository[device]["channels"][int(channel)]["category_name"]
        scores = native.loc[native.series_id.eq(key)].copy()
        for day, day_scores in scores.groupby(scores.start.dt.date):
            reference = select_reference_days(history, day, category, 6)
            reference_values = np.vstack(list(reference.values()))
            percentiles = vector_same_slot(reference_values, days[day]).reshape(48, 6)
            magnitude_max = np.nanmax(percentiles, axis=1)
            magnitude_mean = np.nanmean(percentiles, axis=1)
            day_scores = day_scores.sort_values("start").copy()
            assert len(day_scores) == 48
            np.testing.assert_allclose(magnitude_mean, day_scores.magnitude_mean_percentile,
                                       rtol=1e-8, atol=1e-10, equal_nan=True)
            assert day_scores.valid_point_count.eq(3).all()
            shape_sum = 3*day_scores.mean_score.to_numpy()-magnitude_max
            shape_sum_squares = 3*(day_scores.score_std.to_numpy()**2+day_scores.mean_score.to_numpy()**2)-magnitude_max**2
            discriminant = 2*shape_sum_squares-shape_sum**2
            assert discriminant.min() >= -1e-8
            spread = np.sqrt(np.maximum(discriminant, 0))
            shape_max = (shape_sum+spread)/2
            shape_min = (shape_sum-spread)/2
            np.testing.assert_allclose(np.maximum(shape_max, magnitude_max), day_scores.max_score, atol=1e-7)
            shape_high = shape_max > .995+1e-9
            amplitude_high = magnitude_max > .995
            day_scores["magnitude_max_reconstructed"] = magnitude_max
            day_scores["shape_max_reconstructed"] = shape_max
            day_scores["shape_min_reconstructed"] = shape_min
            day_scores["trigger_source"] = np.select(
                [shape_high & amplitude_high, shape_high, amplitude_high],
                ["shape_and_magnitude", "shape_only", "magnitude_only"], default="below_threshold")
            day_scores["reference_days"] = len(reference)
            parts.append(day_scores)
        july = np.concatenate(list(history.values()))
        september = np.concatenate([days[day] for day in sorted(scores.start.dt.date.unique())])
        july_median = float(np.nanmedian(july))
        drift.append(dict(series_id=key, category_name=category, baseline_days=len(history),
                          target_days=scores.start.dt.date.nunique(), july_median_kw=july_median,
                          september_median_kw=float(np.nanmedian(september)),
                          median_shift_kw=float(np.nanmedian(september))-july_median,
                          july_p95_kw=float(np.nanquantile(july,.95)),
                          september_p95_kw=float(np.nanquantile(september,.95))))
        if position % 40 == 0 or position == len(keys):
            print(f"Attributed {position}/{len(keys)} scored channels", flush=True)
    pd.DataFrame(drift).to_csv(output/"july_september_power_comparison.csv", index=False)
    return pd.concat(parts, ignore_index=True).sort_values(["series_id", "start"]).reset_index(drop=True)


def sequence_event_count(frame, flag, gap_minutes=0):
    selected = frame.loc[flag, ["series_id", "start"]]
    if selected.empty:
        return 0
    difference = selected.groupby("series_id").start.diff()
    return int((difference.isna() | difference.gt(pd.Timedelta(minutes=30+gap_minutes))).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/matrix_profile/results/september_fixed_july_local")
    parser.add_argument("--output-dir", default="runs/analysis/matrix_profile_september_distribution")
    args = parser.parse_args()
    run, output = resolve(args.run_dir), resolve(args.output_dir)
    if run.resolve() == output.resolve():
        raise ValueError("Analysis outputs must be separate from detection outputs")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run/"run_metadata.json").read_text())
    native = pd.read_csv(run/"ALL_SCORES_30MIN.csv")
    native = native.loc[native.available].copy()
    native["start"] = pd.to_datetime(native.interval_start, utc=True).dt.tz_convert(metadata["timezone"])
    native = native.sort_values(["series_id", "start"]).reset_index(drop=True)
    events = pd.read_csv(run/"ALL_EVENTS.csv")
    events["series_id"] = events.device_id+"_"+events.channel_idx.astype(str)
    for column, original in (("start", "start_time"), ("end", "end_time")):
        events[column] = pd.to_datetime(events[original], utc=True).dt.tz_convert(metadata["timezone"])
    unified = pd.read_csv(run/"aligned/all_models_events.csv")
    incidents = pd.read_csv(run/"incident_summary.csv")
    statuses = pd.read_csv(run/"ALL_CHANNEL_STATUS.csv")
    statuses["series_id"] = statuses.device_id+"_"+statuses.channel_idx.astype(str)
    native = native.merge(statuses[["series_id", "category_name"]], on="series_id", validate="many_to_one")
    native = magnitude_attribution(native, Path(metadata["prepared_input_dir"]), metadata, output)
    source_counts = native.groupby("trigger_source").size().rename("native_30m_intervals")
    source_percent = (source_counts/len(native)*100).rename("percent_of_available")
    pd.concat([source_counts, source_percent], axis=1).to_csv(output/"trigger_attribution.csv")
    native[["series_id", "interval_start", "magnitude_max_reconstructed", "shape_max_reconstructed",
            "shape_min_reconstructed", "reference_days", "trigger_source"]].to_csv(output/"interval_attribution_30min.csv", index=False)
    flag = native.max_score > .995
    channel = native.groupby(["series_id", "category_name"]).agg(
        available_intervals=("available", "size"), flagged_intervals=("is_anomaly", "sum"),
        score_one_intervals=("max_score", lambda s:int(s.eq(1).sum())))
    channel["flag_rate"] = channel.flagged_intervals/channel.available_intervals
    event_counts = events.groupby("series_id").size()
    channel["formal_events"] = channel.index.get_level_values("series_id").map(event_counts).fillna(0).astype(int)
    channel.sort_values("flag_rate", ascending=False).to_csv(output/"by_channel.csv")
    daily = native.groupby(native.start.dt.strftime("%Y-%m-%d")).agg(
        available_intervals=("available", "size"), flagged_intervals=("is_anomaly", "sum"))
    daily["flag_rate"] = daily.flagged_intervals/daily.available_intervals
    daily["formal_events"] = daily.index.map(events.groupby(events.start.dt.strftime("%Y-%m-%d")).size()).fillna(0).astype(int)
    daily.to_csv(output/"by_date.csv", index_label="date")
    category = channel.groupby("category_name")[["available_intervals", "flagged_intervals", "formal_events"]].sum()
    category["scored_channels"] = channel.groupby("category_name").size()
    category["flag_rate"] = category.flagged_intervals/category.available_intervals
    category["formal_events_per_100_channel_days"] = category.formal_events/(category.available_intervals/48)*100
    category.to_csv(output/"by_category.csv")
    periodic = native.groupby([native.start.dt.weekday.ge(5).rename("is_weekend"),
                               native.start.dt.hour.rename("hour")]).agg(
        intervals=("available", "size"), flags=("is_anomaly", "sum"))
    periodic["flag_rate"] = periodic["flags"]/periodic["intervals"]
    periodic.to_csv(output/"by_weekend_hour.csv")
    quantiles = native[["max_score", "mean_score", "magnitude_mean_percentile"]].quantile([0,.1,.25,.5,.75,.9,.95,.99,1])
    quantiles.to_csv(output/"score_quantiles.csv", index_label="quantile")
    scenarios = []
    for threshold in (.995,.997,.999,.9999):
        high = native.max_score > threshold
        scenarios.append(dict(rule=f"threshold_gt_{threshold}", scope="threshold_exceedance",
                              native_flagged_intervals=int(high.sum()), aligned_flagged_intervals=int(high.sum())*2,
                              event_or_reporting_units=sequence_event_count(native, high),
                              interval_reduction_percent=(1-high.sum()/flag.sum())*100))
    for gap in (0,15,30):
        scenarios.append(dict(rule=f"threshold_gap_merge_{gap}m", scope="threshold_exceedance",
                              native_flagged_intervals=int(flag.sum()), aligned_flagged_intervals=int(flag.sum())*2,
                              event_or_reporting_units=sequence_event_count(native,flag,gap)))
    reporting = []
    for gap in (0,15,30):
        merged = merge_reporting_units(events, gap)
        assert merged.original_event_count.sum()==len(events)
        merged.to_csv(output/f"formal_reporting_units_gap_{gap}m.csv", index=False)
        reporting.append(dict(rule=f"same_channel_type_direction_merge_gap_{gap}m", units=len(merged),
                              reduction_percent=(1-len(merged)/len(events))*100, changes_detection=False))
    reporting.append(dict(rule="existing_common_mode_incidents", units=len(incidents),
                          reduction_percent=(1-len(incidents)/len(events))*100, changes_detection=False))
    protect = events.confidence.isin(["high", "physical"]) | events.event_type.eq("data_quality")
    for minimum in (30,60):
        keep = protect | events.duration_minutes.ge(minimum)
        reporting.append(dict(rule=f"priority_duration_ge_{minimum}m_protect_high_physical", units=int(keep.sum()),
                              reduction_percent=(1-keep.sum()/len(events))*100, changes_detection=False,
                              note="Priority list only; short events are NOT relabelled normal"))
    extra = events.corroborating_metrics.map(lambda value:any(metric in json.loads(value)
                                           for metric in ("iRMSMax", "change_point", "daily_profile")))
    weak = events.confidence.eq("medium") & ~extra & ~events.event_type.eq("data_quality")
    reporting.append(dict(rule="priority_with_extra_confirmation_for_medium", units=int((~weak).sum()),
                          reduction_percent=float(weak.mean()*100), changes_detection=False,
                          note="Do not discard weak shape anomalies without labels"))
    pd.DataFrame(scenarios).to_csv(output/"interval_threshold_scenarios.csv", index=False)
    pd.DataFrame(reporting).to_csv(output/"formal_reporting_scenarios.csv", index=False)
    summary = dict(analysis_only=True, production_changed=False, native_available=len(native),
                   aligned_available=len(native)*2, native_flagged=int(flag.sum()),
                   aligned_flagged=int(flag.sum())*2, flag_rate=float(flag.mean()),
                   score_exactly_one=int(native.max_score.eq(1).sum()),
                   saturation_fraction_of_flags=float(native.max_score.eq(1).sum()/flag.sum()),
                   trigger_counts=source_counts.to_dict(), formal_events=len(events),
                   unified_threshold_events=len(unified), confidence_counts=events.confidence.value_counts().to_dict(),
                   event_type_counts=events.event_type.value_counts().to_dict(),
                   duration_quantiles_minutes=events.duration_minutes.quantile([.1,.25,.5,.75,.9,.95]).to_dict(),
                   unified_single_30m_events=int(unified.duration_minutes.eq(30).sum()),
                   existing_incidents=len(incidents), formal_reporting_scenarios=reporting,
                   interval_scenarios=scenarios, input_run=str(run))
    weekend = native.start.dt.weekday.ge(5)
    summary["weekday_flag_rate"] = float(flag.loc[~weekend].mean())
    summary["weekend_flag_rate"] = float(flag.loc[weekend].mean())
    summary["attribution_validation"] = "July-only magnitude means and reconstructed overall maxima match every saved scored interval"
    (output/"analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    bins = [0,.9,.99,.995,.999,.999999,1.000001]
    hist = pd.cut(native.max_score,bins=bins,include_lowest=True).value_counts(sort=False)
    axes[0,0].bar(range(len(hist)),hist.to_numpy(),color="#366c99")
    axes[0,0].set_xticks(range(len(hist)),["0–.90",".90–.99",".99–.995",".995–.999",".999–<1","1"],rotation=25)
    axes[0,0].set(title="Native score distribution",ylabel="Available 30-minute intervals")
    duration_edges=[0,30,60,120,240,480,1440]
    duration_hist=pd.cut(events.duration_minutes,duration_edges,include_lowest=True).value_counts(sort=False)
    axes[0,1].bar(range(len(duration_hist)),duration_hist.to_numpy(),color="#598856")
    axes[0,1].set_xticks(range(len(duration_hist)),["≤30","30–60","60–120","120–240","240–480","480–1440"],rotation=25)
    axes[0,1].set(title="Formal event duration",xlabel="Minutes",ylabel="Events")
    channel.flag_rate.plot.hist(bins=np.linspace(0,1,21),ax=axes[0,2],color="#366c99")
    axes[0,2].set(title="Flag rate by scored channel",xlabel="Fraction of available intervals",ylabel="Channels")
    axes[1,0].plot(pd.to_datetime(daily.index),daily.flag_rate*100,color="#366c99")
    axes[1,0].set(title="Daily threshold flag rate",ylabel="Percent of available intervals")
    axes[1,0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=6))
    axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1,0].tick_params(axis="x",rotation=30)
    counts=events.confidence.value_counts().reindex(["medium","high","physical"])
    axes[1,1].bar(counts.index,counts.to_numpy(),color=["#b58b47","#366c99","#598856"])
    axes[1,1].set(title="Formal detector confidence",ylabel="Events")
    labels=["Original","Same-channel\nadjacency","Common-mode\nincidents","Priority\nextra evidence"]
    values=[len(events),reporting[0]["units"],len(incidents),int((~weak).sum())]
    axes[1,2].bar(labels,values,color=["#777777","#366c99","#598856","#b58b47"])
    for index,value in enumerate(values):axes[1,2].text(index,value+25,f"{value:,}",ha="center")
    axes[1,2].set(title="Reporting / priority what-ifs (not accuracy)",ylabel="Review units")
    fig.suptitle("September 2026 Matrix Profile diagnostics — frozen July baseline",fontsize=16)
    fig.savefig(output/"distribution_overview.png",dpi=160)
    plt.close(fig)
    fig,axis=plt.subplots(figsize=(9,4),constrained_layout=True)
    source_counts.reindex(["below_threshold","magnitude_only","shape_only","shape_and_magnitude"]).plot.bar(ax=axis,color="#366c99",rot=15)
    axis.set(title="Attribution of the native 30-minute threshold rule",ylabel="Available intervals",xlabel="")
    fig.savefig(output/"trigger_attribution.png",dpi=160)
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
