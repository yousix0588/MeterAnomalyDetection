from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/matrix_profile/src"))
from anomalies.multiscale.config import DetectionConfig
from anomalies.multiscale.models import ChannelContext, DayScores, PreparedChannel
from anomalies.multiscale.magnitude_events import (
    _contiguous_groups, build_magnitude_events, calibrate_magnitude_reference,
    event_point_mask, MagnitudeEventConfig,
)


def setup_case():
    config = DetectionConfig(detection_start=date(2026, 9, 1), detection_end=date(2026, 9, 30))
    context = ChannelContext("test", 0, "test", 1, "Lighting & Power")
    power = 5 + np.sin(np.arange(288) / 288 * 2 * np.pi)
    history = {}
    for offset in range(31):
        day = date(2026, 7, 1) + timedelta(days=offset)
        index = pd.date_range(str(day), periods=288, freq="5min", tz=config.timezone)
        history[day] = pd.DataFrame(dict(pRealKw=power, iRMSMax=power * 2), index=index)
    day = date(2026, 9, 15)
    index = pd.date_range(str(day), periods=288, freq="5min", tz=config.timezone)
    target = pd.DataFrame(dict(pRealKw=power, iRMSMax=power * 2), index=index)
    scores = DayScores(day, np.zeros(48), np.zeros(48), 0., False,
                       pd.Series(100., index=index), {"iRMSMax": pd.Series(100., index=index)}, [], {})
    prepared = PreparedChannel(history, {day: target}, {}, {})
    reference = calibrate_magnitude_reference(prepared, day, context, config)
    return prepared, target, scores, reference, context, config


def test_frozen_reference_never_reads_target_and_rejects_leakage():
    prepared, target, scores, ref, context, config = setup_case()
    changed = replace(prepared, target_5m={scores.day: target * 100})
    other = calibrate_magnitude_reference(changed, scores.day, context, config)
    np.testing.assert_array_equal(ref.upper_kw, other.upper_kw)
    with pytest.raises(ValueError):
        ref.upper_kw[0] = 100
    bad = replace(prepared, history_5m={**prepared.history_5m, scores.day: target})
    with pytest.raises(ValueError, match="before detection"):
        calibrate_magnitude_reference(bad, scores.day, context, config)


def test_percentile_alone_does_not_generate_amplitude_event():
    _, target, scores, ref, context, config = setup_case()
    assert not build_magnitude_events(scores, target, ref, context, config)


def test_three_point_spike_has_exact_boundaries_without_shape_candidate():
    _, target, scores, ref, context, config = setup_case()
    modified = target.copy()
    modified.iloc[144:147] += 20
    events = build_magnitude_events(scores, modified, ref, context, config)
    assert len(events) == 1
    assert events[0].start_time == target.index[144]
    assert events[0].end_time == target.index[147]
    assert events[0].duration_minutes == 15
    assert events[0].dominant_scale == "point_5m"
    assert "local_jump" in events[0].corroborating_metrics


def test_missing_point_and_single_spike_cannot_be_bridged_to_15_minutes():
    _, target, scores, ref, context, config = setup_case()
    modified = target.copy()
    modified.iloc[144:149] += 20
    modified.iloc[146] = np.nan
    assert not build_magnitude_events(scores, modified, ref, context, config)
    modified = target.copy()
    modified.iloc[144] += 20
    assert not build_magnitude_events(scores, modified, ref, context, config)


def test_six_hour_shift_is_localized_and_legacy_coverage_not_duplicated():
    _, target, scores, ref, context, config = setup_case()
    modified = target.copy()
    modified.iloc[144:216] += 20
    events = build_magnitude_events(scores, modified, ref, context, config)
    assert len(events) == 1 and events[0].duration_minutes == 360
    assert event_point_mask(events, target.index).sum() == 72
    assert not build_magnitude_events(scores, modified, ref, context, config, legacy_events=events)


def test_quality_suppression_does_not_create_business_fault():
    _, target, scores, ref, context, config = setup_case()
    modified = target.copy()
    modified.iloc[144:147] += 20
    events = build_magnitude_events(scores, modified, ref, context, config)
    assert not build_magnitude_events(scores, modified, ref, context, config, quality_events=events)
    mask = np.zeros(288, dtype=bool)
    mask[144:147] = True
    assert not build_magnitude_events(scores, modified, ref, context, config, excluded_mask=mask)


def test_transition_guard_rejects_whole_day_offset_without_onset_evidence():
    _, target, scores, ref, context, config = setup_case()
    modified = target + 20
    assert not build_magnitude_events(scores, modified, ref, context, config)
    assert build_magnitude_events(scores, modified, ref, context, config, mode="envelope_confirmed")


def test_missing_timestamps_split_groups_and_invalid_policy_rejected():
    index = pd.date_range("2026-09-01", periods=6, freq="5min").delete(3)
    assert _contiguous_groups(np.ones(5, dtype=bool), index) == [(0, 3), (3, 5)]
    with pytest.raises(ValueError):
        MagnitudeEventConfig(minimum_effect_kw=0)


def test_tiny_high_percentile_change_does_not_bypass_effect_guard():
    _, target, scores, ref, context, config = setup_case()
    modified = target.copy()
    modified.iloc[144:147] += .02
    assert not build_magnitude_events(scores, modified, ref, context, config)


def test_real_point_event_flags_are_not_repeated_to_wrong_15m_half():
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "shared/scripts"))
    import experiment_matrix_profile_event_policy as trial
    start = pd.Timestamp("2026-09-15 12:00",tz="Australia/Sydney")
    native=pd.DataFrame(dict(interval_start=[start],interval_end=[start+pd.Timedelta(minutes=30)]))
    event=SimpleNamespace(start_time=start+pd.Timedelta(minutes=15),
                          end_time=start+pd.Timedelta(minutes=30),evidence={"supplemental":True})
    assert trial.annotate_scores(native,[event]).event_is_anomaly.tolist()==[True]
    canonical=pd.DataFrame(dict(interval_start=[start,start+pd.Timedelta(minutes=15)],
                                interval_end=[start+pd.Timedelta(minutes=15),start+pd.Timedelta(minutes=30)]))
    assert trial.annotate_scores(canonical,[event]).event_is_anomaly.tolist()==[False,True]


def test_existing_injection_replay_preserves_signed_power_outside_edits():
    sys.path.insert(0, str(ROOT / "shared/scripts"))
    import validate_matrix_profile_injections as validation
    _, original, _, _, _, _ = setup_case()
    original["pRealPositiveKw"] = original.pRealKw
    original["pRealNegativeKw"] = 0.
    injected = original.copy()
    injected["pRealKw_original"] = original.pRealKw
    injected.iloc[144:147,injected.columns.get_loc("pRealKw")] += 20
    injected["pRealPositiveKw"] += 100  # Known whole-range recomputation hazard
    replay,mask,changes = validation.localized_replay(original,injected)
    assert mask.sum()==3 and changes["pRealPositiveKw"]==285
    pd.testing.assert_frame_equal(replay.loc[~mask],original.loc[~mask])


def test_existing_injection_replay_rejects_mismatched_ground_truth():
    sys.path.insert(0, str(ROOT / "shared/scripts"))
    import validate_matrix_profile_injections as validation
    _, original, _, _, _, _ = setup_case()
    injected=original.copy()
    injected["pRealKw_original"]=original.pRealKw+1
    with pytest.raises(ValueError,match="independent original"):
        validation.localized_replay(original,injected)


def test_existing_injection_replay_rejects_unknown_timestamps():
    sys.path.insert(0, str(ROOT / "shared/scripts"))
    import validate_matrix_profile_injections as validation
    _, original, _, _, _, _ = setup_case()
    injected=original.copy()
    injected["pRealKw_original"]=original.pRealKw
    injected.index += pd.Timedelta(days=1)
    with pytest.raises(ValueError,match="absent from original"):
        validation.localized_replay(original,injected)
