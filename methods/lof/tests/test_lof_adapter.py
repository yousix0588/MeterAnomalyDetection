import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"shared/scripts"))
sys.path.insert(0, str(ROOT/"shared/data_tools"))
sys.path.insert(0, str(ROOT/"methods/lof/src"))
import run_lof_local as runner
from meter_input import COLUMNS, normalize
from lof_batch import (
    CALIBRATION_VERSION, CONTAMINATION, FEATURES, N_NEIGHBORS, aggregate_15min, events_from_scores,
    plot_native_events, score_channel,
)


def raw(count=300, start="2026-09-01"):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(count, len(COLUMNS)-1)), columns=COLUMNS[1:])
    frame.insert(0, "timestamp", pd.date_range(start, periods=count, freq="5min", tz="Australia/Sydney"))
    if count > 20:
        frame.loc[20, FEATURES] = 100.
    return frame


def test_scores_and_labels_equal_original_lof_computation():
    frame = raw()
    frame.loc[2, "iRMSMax"] = np.nan
    points, details = score_channel(frame, "DD_TEST_0")
    valid = frame[FEATURES].notna().all(axis=1)
    X_scaled = StandardScaler().fit_transform(frame.loc[valid, FEATURES])
    model = LocalOutlierFactor(n_neighbors=20, contamination=.005)
    labels = model.fit_predict(X_scaled)
    np.testing.assert_array_equal(points.loc[valid, "lof_label"].to_numpy(dtype=int), labels)
    np.testing.assert_allclose(points.loc[valid, "lof_score"], -model.negative_outlier_factor_, rtol=1e-12, atol=1e-12)
    assert details["threshold"] == pytest.approx(-model.offset_, rel=1e-12)
    assert pd.isna(points.loc[2, "is_anomaly"])
    assert pd.isna(points.loc[2, "lof_score"])
    assert not points.loc[2, "available"]
    assert points.loc[2, "data_status"] == "missing_features"


def test_original_model_parameters_without_backup_dependency():
    assert FEATURES == ["powerFactor", "pRealKw", "pReactiveKw", "vRMSMin", "vRMSMax", "iRMSMin", "iRMSMax"]
    assert N_NEIGHBORS == 20
    assert CONTAMINATION == .005


@pytest.mark.parametrize("count", [0, 20])
def test_insufficient_data_does_not_turn_into_normal(count):
    points, details = score_channel(raw(count=count), "DD_SHORT_0")
    assert details["status"] == "insufficient_channel_data"
    assert not points.available.any()
    assert points.is_anomaly.isna().all()
    start = pd.Timestamp("2026-09-01", tz="Australia/Sydney")
    bins = aggregate_15min(points, start, start+pd.Timedelta(days=1), series_id="DD_SHORT_0")
    assert len(bins) == 96
    assert bins.max_score.isna().all() and bins.is_anomaly.isna().all()
    assert not bins.available.any()


def test_15min_max_mean_std_and_native_label_are_not_rethresholded():
    points, _ = score_channel(raw(count=30), "DD_TEST_0")
    points.loc[:2, "lof_score"] = [2., 4., 9.]
    points.loc[:2, "is_anomaly"] = [False, True, False]
    start = points.timestamp.iloc[0]
    bins = aggregate_15min(points, start, start+pd.Timedelta(hours=3))
    row = bins.iloc[0]
    assert row.max_score == row.raw_score == 9.
    assert row.mean_score == 5.
    assert row.score_std == np.std([2., 4., 9.], ddof=0)
    assert row.point_anomaly_count == 1 and row.is_anomaly
    assert row.coverage_ratio == 1. and row.classification_complete
    assert bins.max_percentile.isna().all() and bins.mean_percentile.isna().all()
    assert not bins.ensemble_eligible.any()
    assert bins.calibration_version.eq(CALIBRATION_VERSION).all()
    assert pd.isna(bins.iloc[-1].is_anomaly)


def test_partial_missing_bins_and_gap_do_not_bridge_events():
    points, _ = score_channel(raw(count=30), "DD_TEST_0")
    points["is_anomaly"] = pd.Series(False, index=points.index, dtype="boolean")
    points.loc[[0, 2, 6, 7, 8], "is_anomaly"] = True
    points.loc[1, "available"] = False
    points.loc[1, "is_anomaly"] = pd.NA
    native = events_from_scores(points, 5)
    assert native.duration_minutes.tolist() == [5, 5, 15]
    bins = aggregate_15min(points, points.timestamp.iloc[0], points.timestamp.iloc[0]+pd.Timedelta(hours=3))
    assert bins.iloc[0].coverage_ratio == 2/3
    assert bins.iloc[0].data_status == "partial_coverage"
    assert not bins.iloc[0].classification_complete
    aligned = events_from_scores(bins, 15)
    assert aligned.duration_minutes.tolist() == [15, 15]
    assert aligned.event_time_resolution_minutes.eq(15).all()
    assert aligned.event_semantics.str.contains("not_continuous").all()


def test_existing_frozen_ensemble_rejects_lof_sidecar():
    import build_ensemble_features as ensemble
    points, _ = score_channel(raw(count=30), "DD_TEST_0")
    start = points.timestamp.iloc[0]
    bins = aggregate_15min(points, start, start+pd.Timedelta(hours=3))
    assert ensemble.REQUIRED_COLUMNS.issubset(bins.columns)
    with pytest.raises(ValueError, match="Unknown model"):
        ensemble.validate(bins, "Australia/Sydney")


def fixture_run(tmp_path, monkeypatch, *, with_target=True):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path/"configs").mkdir()
    (tmp_path/"shared_input").mkdir()
    for name in ("methods/lof/src/lof_batch.py", "shared/scripts/run_lof_local.py", "shared/data_tools/meter_input.py"):
        destination = tmp_path/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    config = json.loads((ROOT/"configs/lof.json").read_text())
    config.update(dataset_profile="configs/dataset.json", input_directory="shared_input", output_directory="runs/lof/test")
    (tmp_path/"configs/lof.json").write_text(json.dumps(config))
    profile = dict(dataset_id="test", time=dict(timezone="Australia/Sydney", input_interval_minutes=5,
        ensemble_interval_minutes=15, test_start_inclusive="2026-09-01", test_end_exclusive="2026-09-02"),
        data=dict(shared_input_directory="shared_input"))
    (tmp_path/"configs/dataset.json").write_text(json.dumps(profile))
    history = raw(count=30, start="2026-07-01")
    context = raw(count=30, start="2026-08-31")
    target = raw(count=288) if with_target else raw(count=0)
    combined = pd.concat([history, context, target], ignore_index=True)
    path = tmp_path/"shared_input/DD_TEST_0.csv"
    combined.to_csv(path, index=False)
    return path, target


def test_run_uses_target_only_and_preserves_inputs_and_outputs(tmp_path, monkeypatch):
    path, target = fixture_run(tmp_path, monkeypatch)
    assert not (tmp_path/"legacy").exists()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    output = runner.run("configs/lof.json")
    actual = pd.read_csv(output/"channels/DD_TEST_0/point_scores_5min.csv")
    expected, _ = score_channel(normalize(target, "Australia/Sydney"), "DD_TEST_0")
    np.testing.assert_allclose(actual.lof_score, expected.lof_score, rtol=1e-12)
    assert len(actual) == 288
    assert actual.timestamp.str.startswith("2026-09-01").all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    meta = json.loads((output/"run_metadata.json").read_text())
    assert meta["status"] == "complete" and not meta["historical_baseline_used"] and not meta["context_used"]
    assert not meta["ensemble_eligible"] and meta["scored_channels"] == 1
    assert meta["implementation_source"] == "methods/lof/src/lof_batch.py"
    assert meta["backup_required"] is False
    assert "original_source" not in meta and "original_source_sha256" not in meta
    assert not (tmp_path/"legacy").exists()
    output_hash = hashlib.sha256((output/"ALL_SCORES_15MIN.csv").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="nonempty"):
        runner.run("configs/lof.json")
    assert hashlib.sha256((output/"ALL_SCORES_15MIN.csv").read_bytes()).hexdigest() == output_hash


def test_empty_target_exports_unknown_grid(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch, with_target=False)
    output = runner.run("configs/lof.json")
    bins = pd.read_csv(output/"ALL_SCORES_15MIN.csv")
    assert len(bins) == 96 and bins.is_anomaly.isna().all()
    assert not bins.available.any()
    assert pd.read_csv(output/"ALL_EVENTS_5MIN.csv").empty


def test_config_changes_and_outside_paths_are_rejected(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="inside"):
        runner.resolve(tmp_path.parent/"outside")
    with pytest.raises(ValueError, match="runs/lof"):
        runner.run("configs/lof.json", output_dir="runs/matrix_profile/test")
    config_path = tmp_path/"configs/lof.json"
    config = json.loads(config_path.read_text()); config["contamination"] = .01
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="original LOF"):
        runner.run("configs/lof.json")


def test_duplicate_timestamp_is_rejected_without_silent_averaging(tmp_path, monkeypatch):
    path, target = fixture_run(tmp_path, monkeypatch)
    pd.concat([target, target.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate input timestamp"):
        runner.run("configs/lof.json")
    meta = json.loads((tmp_path/"runs/lof/test/run_metadata.json").read_text())
    assert meta["status"] == "failed"


def test_channel_selection_is_explicit(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.discover_inputs("shared_input", ["DD_TEST_0", "DD_TEST_0"])
    with pytest.raises(FileNotFoundError, match="missing"):
        runner.discover_inputs("shared_input", ["unknown"])
    with pytest.raises(ValueError, match="positive"):
        runner.discover_inputs("shared_input", max_channels=0)


def test_optional_plotting_handles_zero_events(tmp_path):
    points, _ = score_channel(raw(count=20), "DD_TEST_0")
    events = events_from_scores(points, 5)
    start = pd.Timestamp("2026-09-01", tz="Australia/Sydney")
    plot_native_events(events, tmp_path/"plots", start, start+pd.Timedelta(days=1))
    assert len(list((tmp_path/"plots").glob("*.png"))) == 5
