from dataclasses import replace
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/scripts"))
import experiment_matrix_profile_baselines as experiment


def sensor_frame(start="2026-07-01 23:00", points=36):
    index = pd.date_range(start, periods=points, freq="5min", tz="Australia/Sydney")
    return pd.DataFrame({"pRealKw":5.,"vRMSMin":230.,"vRMSMax":235.,
                         "iRMSMin":10.,"iRMSMax":12.,"pRealPositiveKw":5.,
                         "pRealNegativeKw":0.},index=index)


def test_freeze_crosses_midnight_but_does_not_bridge_missing_time():
    frame = sensor_frame()
    mask, runs = experiment.sensor_freezes(frame)
    assert mask.all() and len(runs)==1 and runs[0]["duration_minutes"]==180
    interrupted = frame.drop(frame.index[18])
    assert not experiment.sensor_freezes(interrupted)[0].any()


def test_normal_zero_power_and_voltage_variation_are_not_freezes():
    frame = sensor_frame()
    frame["pRealKw"] = 0
    assert not experiment.sensor_freezes(frame)[0].any()
    frame["pRealKw"] = 5
    frame["vRMSMax"] += np.arange(len(frame))*.001
    assert not experiment.sensor_freezes(frame)[0].any()


def test_freeze_overlap_remains_boolean_across_unobserved_month_gap():
    frame = pd.concat([sensor_frame("2026-07-01"),sensor_frame("2026-09-01")])
    mask,_ = experiment.sensor_freezes(frame)
    grid = pd.date_range("2026-08-31 23:30",periods=3,freq="30min",tz="Australia/Sydney")
    overlap = experiment.freeze_overlap(mask,grid)
    assert overlap.dtype==bool
    np.testing.assert_array_equal(overlap,[False,True,True])
    np.testing.assert_array_equal(~overlap,[True,False,False])


def test_history_cleaning_never_modifies_target_or_turns_missing_into_zero():
    frame = pd.concat([sensor_frame("2026-07-01"),sensor_frame("2026-09-01")])
    mask, _ = experiment.sensor_freezes(frame)
    config = experiment.local.DetectionConfig(detection_start=date(2026,9,1),detection_end=date(2026,9,30))
    cleaned, removed = experiment.mask_history(frame,config,mask)
    assert removed==36
    assert cleaned.loc[cleaned.index.month==7].isna().all().all()
    pd.testing.assert_frame_equal(cleaned.loc[cleaned.index.month==9],frame.loc[frame.index.month==9])


def test_power_probe_updates_signed_power_only_inside_known_interval():
    frame = sensor_frame("2026-09-15",288)
    modified,start,end = experiment.controlled_probe(frame,"spike_15m",2.)
    inside = (frame.index>=start)&(frame.index<end)
    assert inside.sum()==3
    pd.testing.assert_frame_equal(modified.loc[~inside],frame.loc[~inside])
    np.testing.assert_array_equal(modified.loc[inside,"pRealKw"],13.)
    np.testing.assert_array_equal(modified.loc[inside,"pRealPositiveKw"]-modified.loc[inside,"pRealNegativeKw"],modified.loc[inside,"pRealKw"])


def test_baseline_variants_are_frozen_before_september_and_keep_windows():
    original = experiment.local.DetectionConfig(detection_start=date(2026,9,1),detection_end=date(2026,9,30))
    for first,last,_ in experiment.VARIANTS.values():
        config = replace(original,baseline_start=first,baseline_end=last)
        assert config.baseline_end < config.detection_start
        assert (config.short_window_points,config.medium_window_points,config.daily_window_points)==(4,12,48)


def test_probe_pairing_does_not_hide_dropped_coverage_or_count_changed_dates():
    rows = []
    for variant,key,start,overlap in [("raw_july","A","2026-09-15",True),
                                     ("raw_july","B","2026-09-15",True),
                                     ("clean_july_august","A","2026-09-15",False),
                                     ("clean_july_august","B","2026-09-16",True)]:
        rows.append(dict(variant=variant,series_id=key,kind="spike_15m",start=start,end=start+" 00:15",
                         eligible=True,formal_event_overlap=overlap,after_flagged_inside_bins=1))
    result = experiment.paired_probe_comparisons(pd.DataFrame(rows))
    row = result.loc[result.variant.eq("clean_july_august")].iloc[0]
    assert row.identical_eligible_cases==1
    assert row.old_formal_overlaps==1 and row.new_formal_overlaps==0
    assert row.lost_formal_overlaps==1 and row.lost_case_ids=="A"
