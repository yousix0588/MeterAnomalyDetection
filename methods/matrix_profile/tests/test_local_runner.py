from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/scripts"))
import run_matrix_profile_local as local
import numpy as np
import pandas as pd
import pytest
from anomalies.multiscale import scoring
from anomalies.multiscale.detector import MultiscaleAnomalyDetector
from anomalies.multiscale.models import ChannelContext, ChannelStatus
from anomalies.multiscale.preprocessing import prepare_channel


def synthetic_frame(days):
    rng = np.random.default_rng(17)
    frames = []
    for day in days:
        index = pd.date_range(pd.Timestamp(day, tz="Australia/Sydney"), periods=288, freq="5min")
        power = 5 + 2*np.sin(np.arange(288)*2*np.pi/288) + rng.normal(0, .15, 288)
        frames.append(pd.DataFrame(dict(timestamp=index, pRealKw=power, iRMSMax=power*8)))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("category", ["General Power", "Solar Generation"])
def test_fixed_cache_is_equivalent_and_target_independent(monkeypatch, category):
    config = local.DetectionConfig(detection_start=date(2026, 9, 1), detection_end=date(2026, 9, 6))
    days = [date(2026, 7, 1)+timedelta(days=i) for i in range(31)]
    days += [date(2026, 9, i) for i in (1, 2, 6)]
    context = ChannelContext("DD_TEST", 0, "Test", 1, category)
    prepared = prepare_channel(synthetic_frame(days), config, category).data
    assert prepared is not None
    original = scoring.calibrate_leave_one_day_out
    calls = []
    def counted(arrays, window):
        calls.append((tuple(arrays), window))
        return original(arrays, window)
    monkeypatch.setattr(scoring, "calibrate_leave_one_day_out", counted)
    cache = {}
    for day in prepared.target_5m:
        expected = scoring.score_day(prepared, day, context, config)
        count = len(calls)
        actual = scoring.score_day(prepared, day, context, config, cache)
        if day == date(2026, 9, 2):
            assert len(calls) == count  # second weekday does not recalibrate
        np.testing.assert_array_equal(actual.short_percentiles, expected.short_percentiles)
        np.testing.assert_array_equal(actual.medium_percentiles, expected.medium_percentiles)
        pd.testing.assert_series_equal(actual.magnitude_percentiles, expected.magnitude_percentiles)
        assert actual.daily_score == expected.daily_score
        assert actual.nearest_neighbors == expected.nearest_neighbors
    snapshot = {key: values.copy() for key, values in cache.items()}
    prepared.target_30m[date(2026, 9, 1)]["pRealKw"] += 100
    scoring.score_day(prepared, date(2026, 9, 1), context, config, cache)
    for key in cache:
        np.testing.assert_array_equal(cache[key], snapshot[key])
    assert len(cache) == (3 if category == "Solar Generation" else 6)


def test_near_ties_do_not_double_count_ecdf():
    history = {date(2026, 7, i+1): pd.Series([value]) for i, value in enumerate([0., 2.])}
    result = scoring.same_slot_percentiles(history, pd.Series([2.+1e-10]))
    assert result.iloc[0] == 50  # both baseline deviations ~= 1, not 150%
    assert scoring.same_slot_percentiles(history, pd.Series([10.])).iloc[0] == 100


def test_partial_target_day_is_not_labelled_normal():
    config = local.DetectionConfig(detection_start=date(2026, 9, 1), detection_end=date(2026, 9, 2))
    days = [date(2026, 7, 1)+timedelta(days=i) for i in range(31)] + [date(2026, 9, 1)]
    result = MultiscaleAnomalyDetector(config).detect_channel(
        synthetic_frame(days), ChannelContext("DD_TEST", 0, "Test", 1, "General Power"))
    assert result.status in {ChannelStatus.DETECTED, ChannelStatus.NO_EVENT}
    assert result.day_statuses[1].status == ChannelStatus.MISSING_TARGET
    assert len(result.interval_scores) == 48


def test_canonical_missing_grid_retains_disclosure():
    config = local.DetectionConfig(detection_start=date(2026, 9, 1), detection_end=date(2026, 9, 1))
    status = pd.DataFrame([dict(date="2026-09-01", status="missing_target")])
    native = local.complete_grid(pd.DataFrame(columns=local.INTERVAL_SCORE_COLUMNS), "DD_TEST_0", status, config)
    canonical = local.ensemble.validate(local._upsample_interval_frame(native), config.timezone)
    assert len(native) == 48 and len(canonical) == 96
    assert canonical.max_score.isna().all()
    assert not canonical.available.any() and not canonical.is_anomaly.any()
    assert canonical.availability_reason.eq("missing_target").all()
    assert canonical.data_status.eq("upsampled_from_30m").all()
    assert canonical.source_resolution_minutes.eq(30).all()
    wide = local.ensemble.build_wide(canonical, pd.Timestamp("2026-09-01", tz=config.timezone),
                                      pd.Timestamp("2026-09-02", tz=config.timezone))
    for model in local.ensemble.ALLOWED_MODELS:
        assert not wide[f"{model}_available"].any()


def test_skipped_channel_still_records_missing_dates():
    config = local.DetectionConfig(detection_start=date(2026, 9, 1), detection_end=date(2026, 9, 2))
    days = [date(2026, 7, 1)+timedelta(days=i) for i in range(31)] + [date(2026, 9, 1)]
    frame = synthetic_frame(days)
    frame.pRealKw = 0.
    result = MultiscaleAnomalyDetector(config).detect_channel(frame, ChannelContext("DD_TEST", 0, "Test", 1, "General Power"))
    assert result.status == ChannelStatus.LOW_ACTIVITY
    assert [item.status for item in result.day_statuses] == [ChannelStatus.LOW_ACTIVITY, ChannelStatus.MISSING_TARGET]


def test_input_preparation_selects_only_baseline_and_target(tmp_path):
    source, history, destination = [tmp_path/name for name in ("raw", "history", "merged")]
    source.mkdir()
    history.mkdir()
    def raw(day):
        frame = pd.DataFrame({column: [1.] for column in local.COLUMNS})
        frame["timestamp"] = [f"{day} 00:00:00+10:00"]
        return frame
    pd.concat([raw("2026-07-01"), raw("2026-08-01")]).to_csv(history/"DD_TEST_0.csv", index=False)
    for folder, day in [("01_09_2026", "2026-09-01"), ("01_10_2026", "2026-10-01")]:
        (source/folder).mkdir()
        raw(day).to_csv(source/folder/"DD_TEST_0.csv", index=False)
    config = local.DetectionConfig(detection_start=date(2026, 9, 1), detection_end=date(2026, 9, 30))
    with local.ArchiveReader(str(source)) as reader:
        indexed = local.index_target(reader, date(2026, 9, 1), date(2026, 10, 1))
        local.prepare_inputs(reader, indexed, ["DD_TEST_0", "DD_ABSENT_0"], history, destination, config)
    result = pd.read_csv(destination/"DD_TEST_0.csv")
    assert list(result.columns) == local.COLUMNS
    assert result.timestamp.str[:10].tolist() == ["2026-07-01", "2026-09-01"]
    manifest = pd.read_csv(destination/"_manifest.csv").set_index("meter_channel")
    assert manifest.loc["DD_ABSENT_0", "target_days_missing"] == 30
    assert manifest.loc["DD_TEST_0", "target_days_present"] == 1


def test_duplicate_and_bad_schema_rejected():
    frame = pd.DataFrame({column: [1., 1.] for column in local.COLUMNS})
    frame.timestamp = ["2026-09-01 00:00:00+10:00"]*2
    with pytest.raises(ValueError, match="Duplicate"):
        local.normalize(frame, "Australia/Sydney")
    with pytest.raises(ValueError, match="12-column"):
        local.normalize(frame.drop(columns="pRealKw"), "Australia/Sydney")


def test_refined_duplicate_events_keep_both_evidence_records():
    from anomalies.multiscale.aggregate import _coalesce_refined_events
    import json
    event = dict(event_id="same", device_id="TEST", channel_idx=0, start_time="2026-09-01T00:00:00+10:00",
                 end_time="2026-09-01T01:00:00+10:00", event_type="state_change", confidence="medium",
                 short_percentile=99.9, medium_percentile=90., magnitude_percentile=100.,
                 dominant_scale="short", evidence=json.dumps(dict(coarse_start="first")))
    other = dict(event, short_percentile=99.8, evidence=json.dumps(dict(coarse_start="second")))
    result = _coalesce_refined_events(pd.DataFrame([event, other]))
    assert len(result) == 1
    assert result.short_percentile.iloc[0] == 99.9
    assert len(json.loads(result.evidence.iloc[0])["refined_candidate_records"]) == 2
    other["end_time"] = "2026-09-01T02:00:00+10:00"
    with pytest.raises(ValueError, match="Conflicting"):
        _coalesce_refined_events(pd.DataFrame([event, other]))
