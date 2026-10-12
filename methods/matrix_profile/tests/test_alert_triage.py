from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/matrix_profile/src"))
from anomalies.multiscale.alert_triage import (
    AlertTriageConfig, build_review_units, classify_events, event_evidence, interval_mask,
)
from anomalies.multiscale.magnitude_events import FrozenMagnitudeReference


def case():
    index = pd.date_range("2026-09-15", periods=288, freq="5min", tz="Australia/Sydney")
    target = pd.DataFrame(dict(pRealKw=np.full(288, 5.), iRMSMax=np.full(288, 10.)), index=index)
    ref = FrozenMagnitudeReference((date(2026, 7, 1), date(2026, 7, 2)),
          np.full(288, 4.), np.full(288, 6.), np.full(288, 5.), np.full(288, 23), .5, 2.,
          np.full(288, 8.), np.full(288, 12.), .5)
    event = dict(event_id="one", device_id="meter", channel_idx=0, organization="test",
                 category_id=1, category_name="test", start_time=index[140].isoformat(),
                 end_time=index[212].isoformat(), duration_minutes=360, event_type="shape_change",
                 confidence="high", short_percentile=100., medium_percentile=100.,
                 evidence="{}", incident_id="event-one")
    return target, ref, event


def test_saturated_percentile_is_not_a_priority_alert():
    target, ref, event = case()
    found = event_evidence(event, target, ref)
    assert found["review_queue"] == "operating_change_review"
    assert json.loads(found["priority_segments"]) == []


def test_real_spike_has_point_core_not_six_hour_window_halo():
    target, ref, event = case()
    target.iloc[147:150, 0] += 20
    classified = classify_events(pd.DataFrame([event]), {target.index[0].date(): target},
                                 {target.index[0].date(): ref})
    assert classified.review_queue.iloc[0] == "priority_alert"
    assert classified.priority_core_minutes.iloc[0] == 15
    assert interval_mask(classified, target.index).sum() == 72
    assert interval_mask(classified, target.index, queue="priority_alert", priority_cores=True).sum() == 3
    bins = pd.date_range(target.index[144], periods=3, freq="15min")
    np.testing.assert_array_equal(interval_mask(classified, bins, 15, "priority_alert", True), [False, True, False])
    assert classified.start_time.iloc[0] == event["start_time"]
    assert classified.evidence.iloc[0] == event["evidence"]


def test_negative_power_effect_can_be_prioritized():
    target, ref, event = case()
    target.iloc[147:160, 0] -= 20
    found = event_evidence(event, target, ref)
    assert found["review_queue"] == "priority_alert" and found["direction"] == "down"
    # Priority direction comes from actual cores, not the mostly unchanged
    # six-hour source window's median.
    event.update(start_time=target.index[147].isoformat(), end_time=target.index[160].isoformat())
    assert event_evidence(event, target, ref)["direction"] == "down"


def test_missing_model_confirmation_needs_real_current_support():
    target, ref, event = case()
    event.update(short_percentile=None, medium_percentile=None)
    target.iloc[147:150, 0] += 20
    assert event_evidence(event, target, ref)["review_queue"] == "operating_change_review"
    target.iloc[147:150, 1] += 30
    assert event_evidence(event, target, ref)["review_queue"] == "priority_alert"


def test_slow_drift_is_review_not_normal_or_priority():
    target, ref, event = case()
    target.iloc[140:212, 0] += np.arange(72) / 10
    found = event_evidence(event, target, ref)
    assert found["peak_outside_envelope_kw"] > 1.
    assert found["review_queue"] == "operating_change_review"
    assert "no_abrupt_onset_above_frozen_jump_guard" in json.loads(found["triage_reasons"])


@pytest.mark.parametrize("points", [1, 2])
def test_sub_15_minute_spikes_remain_review(points):
    target, ref, event = case()
    target.iloc[147:147+points, 0] += 20
    assert event_evidence(event, target, ref)["review_queue"] == "operating_change_review"


def test_missing_point_is_not_interpolated_or_bridged():
    target, ref, event = case()
    target.iloc[147:152, 0] += 20
    target.iloc[149, 0] = np.nan
    assert event_evidence(event, target, ref)["review_queue"] == "operating_change_review"


def test_opposite_two_point_spikes_cannot_form_a_15_minute_core():
    target, ref, event = case()
    target.iloc[147:149, 0] += 20
    target.iloc[149:151, 0] -= 20
    assert event_evidence(event, target, ref)["review_queue"] == "operating_change_review"


def test_coming_out_of_a_quality_gap_does_not_supply_onset_evidence():
    target, ref, event = case()
    target.iloc[147:150, 0] += 20
    excluded = np.zeros(288, dtype=bool)
    excluded[146] = True
    assert event_evidence(event, target, ref, quality_mask=excluded)["review_queue"] == "operating_change_review"


def test_quality_is_separate_and_cannot_validate_a_business_fault():
    target, ref, event = case()
    target.iloc[147:150, 0] += 20
    quality = {**event, "event_id": "quality", "event_type": "data_quality", "confidence": "physical",
               "start_time": target.index[147].isoformat(), "end_time": target.index[150].isoformat(),
               "evidence": '{"quality_issue":"flatline"}'}
    source = pd.DataFrame([event, quality])
    found = classify_events(source, {target.index[0].date(): target}, {target.index[0].date(): ref})
    assert found.review_queue.tolist() == ["operating_change_review", "data_quality"]
    pd.testing.assert_frame_equal(found[list(source.columns)], source)


def test_unavailable_reference_never_becomes_normal_or_priority():
    _, _, event = case()
    found = classify_events(pd.DataFrame([event]), {}, {})
    assert found.review_queue.iloc[0] == "operating_change_review"
    assert "missing_target_or_frozen_reference" in found.triage_reasons.iloc[0]


def test_future_reference_and_invalid_grids_are_rejected():
    target, ref, event = case()
    with pytest.raises(ValueError, match="precede"):
        event_evidence(event, target, replace(ref, dates=(target.index[0].date(),)))
    with pytest.raises(ValueError, match="288"):
        event_evidence(event, target.iloc[:-1], ref)


def test_ground_truth_columns_do_not_change_classification():
    target, ref, event = case()
    target.iloc[147:150, 0] += 20
    before = event_evidence(event, target, ref)
    target["pRealKw_original"] = -1e9
    target["is_injected"] = True
    assert event_evidence(event, target, ref) == before


def test_grouping_preserves_ids_and_splits_queue_direction_and_type():
    target, _, base = case()
    rows = []
    for i, (queue, direction, event_type, channel) in enumerate([
        ("operating_change_review", "up", "shape_change", 0),
        ("operating_change_review", "up", "shape_change", 0),
        ("operating_change_review", "down", "shape_change", 1),
        ("priority_alert", "up", "shape_change", 2),
        ("data_quality", "unknown", "data_quality", 3),
        ("operating_change_review", "up", "shape_change", 4),
    ]):
        rows.append({**base, "event_id": str(i), "channel_idx": channel, "review_queue": queue,
                     "direction": direction, "event_type": event_type, "quality_issue": "unspecified",
                     "priority_segments": "[]", "incident_id": "common-test",
                     "start_time": target.index[140+i].isoformat(), "end_time": target.index[160+i].isoformat()})
    units, membership = build_review_units(pd.DataFrame(rows))
    assert len(units) == 4
    assert set(membership.event_id) == set(str(i) for i in range(6))
    assert membership.event_id.is_unique
    grouped = units.loc[units.source_event_count.eq(3)].iloc[0]
    assert grouped.grouping_kind == "common_mode_review" and grouped.channel_count == 2


def test_positive_merge_gap_is_explicit_and_recorded():
    target, _, base = case()
    first = {**base, "review_queue": "operating_change_review", "direction": "up", "quality_issue": "unspecified", "priority_segments": "[]"}
    second = {**first, "event_id": "two", "start_time": target.index[215].isoformat(), "end_time": target.index[220].isoformat(), "incident_id": "event-two"}
    assert len(build_review_units(pd.DataFrame([first, second]))[0]) == 2
    units, _ = build_review_units(pd.DataFrame([first, second]), AlertTriageConfig(merge_gap_minutes=15))
    assert len(units) == 1 and units.contains_gaps.iloc[0]


def test_duplicate_ids_are_rejected():
    _, _, event = case()
    with pytest.raises(ValueError, match="unique"):
        classify_events(pd.DataFrame([event, event]), {}, {})


def test_report_refresh_handles_windows_hash_paths(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "shared/scripts"))
    import triage_matrix_profile_alerts as runner
    config = tmp_path / "policy.json"
    config.write_text("{}", encoding="utf-8")
    implementation = {}
    for relative in ("methods/matrix_profile/src/anomalies/multiscale/alert_triage.py",
                     "methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py"):
        implementation[relative.replace("/", "\\")] = runner.digest(ROOT / relative)
    metadata = dict(status="complete", implementation_sha256=implementation,
                    policy_config_sha256=runner.digest(config))
    (tmp_path / "triage_summary.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in ("queue_counts", "review_units", "events_all_classified", "injection_triage_comparison"):
        pd.DataFrame({"example": [1]}).to_csv(tmp_path / f"{name}.csv", index=False)
    called = []
    monkeypatch.setattr(runner, "render_report", lambda *args: called.append(True))
    runner.main(["--finalize-only", "--output-dir", str(tmp_path), "--config", str(config)])
    assert called == [True]
    result = json.loads((tmp_path / "triage_summary.json").read_text(encoding="utf-8"))
    assert result["implementation_sha256"] == implementation
    assert "report_render_code_sha256" in result


def test_report_refresh_rejects_changed_policy(tmp_path):
    sys.path.insert(0, str(ROOT / "shared/scripts"))
    import triage_matrix_profile_alerts as runner
    config = tmp_path / "policy.json"
    config.write_text("{}", encoding="utf-8")
    implementation = {relative: runner.digest(ROOT / relative) for relative in (
        "methods/matrix_profile/src/anomalies/multiscale/alert_triage.py",
        "methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py")}
    (tmp_path / "triage_summary.json").write_text(json.dumps(dict(
        status="complete", implementation_sha256=implementation, policy_config_sha256="wrong")), encoding="utf-8")
    with pytest.raises(ValueError, match="policy changed"):
        runner.main(["--finalize-only", "--output-dir", str(tmp_path), "--config", str(config)])
