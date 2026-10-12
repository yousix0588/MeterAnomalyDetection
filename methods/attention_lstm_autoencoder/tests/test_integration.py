"""Adapter tests do not require PyTorch, trained weights or backup folders."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"shared/scripts"))
from methods.attention_lstm_autoencoder import project,inference
from methods.attention_lstm_autoencoder.main import main


def training_config():
    return yaml.safe_load((ROOT/"configs/attention_lstm_autoencoder.yaml").read_text(encoding="utf-8"))


def package(tmp_path,category="1"):
    folder=tmp_path/"models"/category
    (folder/"calibration").mkdir(parents=True)
    config=training_config()
    project.write_json(folder/"config.json",config)
    scaler=dict(features=config["data"]["features"],center=[0.]*11,scale=[1.]*11)
    project.write_json(folder/"scaler.json",scaler)
    # Deliberately not actual model weights. Pure tests inject a NumPy scoring
    # stub and verify I/O only; real neural execution has separate optional tests.
    (folder/"best_model.pt").write_bytes(b"synthetic-package-integrity-fixture")
    values=np.asarray([0.,1.,2.,4.],dtype=np.float32)
    np.save(folder/"calibration/max_sorted.npy",values)
    np.save(folder/"calibration/mean_sorted.npy",values)
    metadata=dict(binding="attention-fixture-v1",source="held_out_calibration",split_config=config["splits"],
        threshold=1.5,score_block_size=3,frequency="5min",score_definition="last_block_point_reconstruction_mse_max",
        total_scores=4,sample_scores=4,approximate=False,model_sha256=project.digest(folder/"best_model.pt"),
        scaler_sha256=project.digest(folder/"scaler.json"),max_sample_sha256=project.digest(folder/"calibration/max_sorted.npy"),
        mean_sample_sha256=project.digest(folder/"calibration/mean_sorted.npy"))
    project.write_json(folder/"metadata.json",metadata)
    return folder


def raw(day,count=288):
    frame=pd.DataFrame({feature:np.ones(count) for feature in project.FEATURES})
    frame.insert(0,"timestamp",pd.date_range(day,periods=count,freq="5min",tz="Australia/Sydney"))
    return frame


def fixture(tmp_path,monkeypatch,*,context=True,weights=True,unsupported=False):
    monkeypatch.setattr(project,"ROOT",tmp_path)
    (tmp_path/"configs").mkdir(); (tmp_path/"input").mkdir()
    for name in ("methods/attention_lstm_autoencoder/project.py","methods/attention_lstm_autoencoder/inference.py","methods/attention_lstm_autoencoder/src/model.py"):
        dest=tmp_path/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/name).read_bytes())
    profile=dict(dataset_id="september_fixture",time=dict(timezone="Australia/Sydney",input_interval_minutes=5,
        ensemble_interval_minutes=15,train_end_exclusive="2026-08-01",test_start_inclusive="2026-09-01",test_end_exclusive="2026-09-02"),
        data=dict(shared_input_directory="input"))
    project.write_json(tmp_path/"configs/dataset.json",profile)
    repo={"DD_TEST":dict(device_id="DD_TEST",channels=[dict(id="DD_TEST_C1",category_id=1,category_name="Grid")])}
    project.write_json(tmp_path/"configs/repository.json",repo)
    config=dict(version=1,model=project.MODEL,mode="frozen_inference",dataset_profile="configs/dataset.json",
        meter_repository="configs/repository.json",model_packages={"1":"models/1"},
        output_directory="runs/attention_lstm_autoencoder/results/test",batch_size=8,device="cpu",ensemble_eligible=False)
    project.write_json(tmp_path/"configs/inference.json",config)
    target=raw("2026-09-01"); target.loc[1,"pRealKw"]=25.
    frames=[raw("2026-07-01"),target]
    if context: frames.append(raw("2026-08-31"))
    path=tmp_path/"input/DD_TEST_0.csv"
    pd.concat(frames,ignore_index=True).to_csv(path,index=False)
    if unsupported: target.to_csv(tmp_path/"input/AA_UNKNOWN_0.csv",index=False)
    folder=package(tmp_path) if weights else tmp_path/"models/1"
    return path,folder


def numpy_scoring_stub(package,runtime):
    def score(prepared):
        result={key:[] for key in ("max","mean","std")}
        for index in np.flatnonzero(prepared["valid"]):
            start=prepared["starts"][index]
            tail=(prepared["values"][start+285:start+288]-package["center"])/package["scale"]
            error=np.mean(tail**2,axis=1)
            result["max"].append(error.max()); result["mean"].append(error.mean()); result["std"].append(error.std(ddof=0))
        return {key:np.asarray(value) for key,value in result.items()}
    return score


def test_project_paths_and_training_defaults():
    config=project.load_training_config("configs/attention_lstm_autoencoder.yaml")
    assert Path(config["data"]["root"])==ROOT/"data/processed/meter_csvs"
    assert config["data"]["window_size"]==288 and config["data"]["stride"]==3
    assert config["splits"]["train_end"]=="2026-08-01"
    assert config["splits"]["calibration_ratio"]==.15
    with pytest.raises(ValueError,match="separate output"):
        project.output_path("runs/lstm_autoencoder/old")


def test_network_contains_attention_and_native_model_identity_is_distinct():
    tree=ast.parse((ROOT/"methods/attention_lstm_autoencoder/src/model.py").read_text())
    classes={node.name for node in tree.body if isinstance(node,ast.ClassDef)}
    assert {"TemporalAttention","LSTMAutoencoder"}.issubset(classes)
    native=ast.parse((ROOT/"methods/attention_lstm_autoencoder/src/pipeline.py").read_text())
    assert any(isinstance(node,ast.Constant) and node.value==project.MODEL for node in ast.walk(native))


def test_category_mapping_uses_numeric_rank_not_suffix_guess(tmp_path,monkeypatch):
    monkeypatch.setattr(project,"ROOT",tmp_path)
    path=tmp_path/"repo.json"
    project.write_json(path,{"D":dict(device_id="D",channels=[dict(id="D_C6",category_id=4,category_name="AC"),dict(id="D_C1",category_id=1,category_name="Grid")])})
    mapping=project.category_mapping(path)
    assert mapping["D_0"]["channel_id"]=="D_C1"
    assert mapping["D_1"]["channel_id"]=="D_C6"
    assert "D_5" not in mapping


def test_preflight_missing_packages_does_not_create_results(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,weights=False)
    report=inference.make_plan("configs/inference.json")[0]
    assert not report["trained_packages_ready"]
    assert "best_model.pt" in report["category_packages"]["1"]["reason"]
    with pytest.raises(FileNotFoundError,match="not ready"):
        inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    assert not (tmp_path/"runs").exists()


@pytest.mark.parametrize("change",["cutoff","feature_order","checksum","ecdf","approximation"])
def test_incompatible_or_tampered_package_is_rejected(tmp_path,monkeypatch,change):
    monkeypatch.setattr(project,"ROOT",tmp_path); folder=package(tmp_path)
    if change=="cutoff":
        config=project.read_json(folder/"config.json"); config["splits"]["train_end"]="2026-09-01"; project.write_json(folder/"config.json",config)
    elif change=="feature_order":
        scaler=project.read_json(folder/"scaler.json"); scaler["features"]=list(reversed(scaler["features"])); project.write_json(folder/"scaler.json",scaler)
    elif change=="checksum":
        (folder/"best_model.pt").write_bytes(b"changed")
    elif change=="ecdf":
        np.save(folder/"calibration/max_sorted.npy",np.asarray([2.,1.,3.,4.],np.float32))
        metadata=project.read_json(folder/"metadata.json"); metadata["max_sample_sha256"]=project.digest(folder/"calibration/max_sorted.npy"); project.write_json(folder/"metadata.json",metadata)
    else:
        metadata=project.read_json(folder/"metadata.json"); metadata["approximate"]=True; project.write_json(folder/"metadata.json",metadata)
    with pytest.raises(ValueError):
        inference.load_package(folder,pd.Timestamp("2026-08-01",tz="Australia/Sydney"))


def test_new_period_inference_uses_real_context_and_preserves_all_sources(tmp_path,monkeypatch):
    source,folder=fixture(tmp_path,monkeypatch)
    before=project.digest(source)
    packet=inference.load_package(folder,pd.Timestamp("2026-08-01",tz="Australia/Sydney"))
    frozen=packet["hashes"].copy()
    output=inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    scores=pd.read_csv(output/"ALL_SCORES_15MIN.csv",dtype={"is_anomaly":"boolean"})
    assert len(scores)==96 and scores.available.all()
    assert scores.model.eq(project.MODEL).all()
    assert scores.iloc[0].max_score>scores.iloc[0].mean_score>1.
    assert scores.iloc[0].is_anomaly and not scores.iloc[1].is_anomaly
    assert scores.coverage_ratio.eq(1.).all()
    assert scores.max_percentile.between(0,1).all()
    events=pd.read_csv(output/"ALL_EVENTS.csv")
    assert len(events)==1 and events.iloc[0].duration_minutes==15
    assert project.digest(source)==before
    assert all(project.digest(folder/name)==sha for name,sha in frozen.items())
    metadata=project.read_json(output/"run_metadata.json")
    assert metadata["status"]=="complete" and metadata["fitting_performed"] is False and metadata["calibration_updated"] is False
    with pytest.raises(FileExistsError,match="nonempty"):
        inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    assert not (tmp_path/"legacy").exists()


def test_absent_context_is_unknown_not_normal(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,context=False)
    output=inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    scores=pd.read_csv(output/"ALL_SCORES_15MIN.csv",dtype={"is_anomaly":"boolean"})
    assert not scores.available.iloc[:95].any()
    assert scores.is_anomaly.iloc[:95].isna().all()
    assert scores.max_score.iloc[:95].isna().all()
    assert scores.data_status.iloc[:95].eq("insufficient_history").all()
    assert scores.available.iloc[95]


def test_interpolation_keeps_original_coverage(tmp_path,monkeypatch):
    source,folder=fixture(tmp_path,monkeypatch)
    frame=pd.read_csv(source); frame.loc[frame.timestamp.eq("2026-09-01 00:05:00+10:00"),"pRealKw"]=np.nan; frame.to_csv(source,index=False)
    output=inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    scores=pd.read_csv(output/"ALL_SCORES_15MIN.csv")
    assert scores.iloc[0].available and scores.iloc[0].coverage_ratio==pytest.approx(2/3)
    assert scores.iloc[0].data_status=="interpolated"
    values=inference.short_gap_fill(np.array([[np.nan],[1.],[np.nan],[3.],[np.nan]]),1)
    assert values[2,0]==2. and np.isnan(values[0,0]) and np.isnan(values[-1,0])


def test_unknown_category_keeps_identical_csv_schema_and_missing_flags(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,unsupported=True)
    output=inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    scores=pd.read_csv(output/"ALL_SCORES_15MIN.csv",dtype={"is_anomaly":"boolean"})
    unknown=scores.loc[scores.series_id.eq("AA_UNKNOWN_0")]
    assert len(scores)==192 and len(unknown)==96
    assert unknown.max_percentile.isna().all() and unknown.is_anomaly.isna().all() and not unknown.available.any()
    assert unknown.coverage_ratio.isna().all()
    assert scores.loc[scores.series_id.eq("DD_TEST_0"),"available"].all()


def test_duplicate_timestamp_and_output_outside_scope_are_rejected(tmp_path,monkeypatch):
    source,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="separate output"):
        inference.run_inference("configs/inference.json",output_dir="runs/lstm_autoencoder/test",_scorer_factory=numpy_scoring_stub)
    frame=pd.read_csv(source); pd.concat([frame,frame.iloc[:1]]).to_csv(source,index=False)
    with pytest.raises(ValueError,match="Duplicate input timestamp"):
        inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    assert project.read_json(tmp_path/"runs/attention_lstm_autoencoder/results/test/run_metadata.json")["status"]=="failed"


def test_original_ensemble_registry_is_unchanged(tmp_path,monkeypatch):
    import build_ensemble_features as ensemble
    fixture(tmp_path,monkeypatch)
    output=inference.run_inference("configs/inference.json",_scorer_factory=numpy_scoring_stub)
    scores=pd.read_csv(output/"ALL_SCORES_15MIN.csv")
    assert ensemble.REQUIRED_COLUMNS.issubset(scores.columns)
    with pytest.raises(ValueError,match="Unknown model"):
        ensemble.validate(scores,"Australia/Sydney")


def test_validation_cli_has_no_model_or_backup_dependency(tmp_path,monkeypatch,capsys):
    fixture(tmp_path,monkeypatch,weights=False)
    assert main(["validate","--config","configs/inference.json"])==2
    report=json.loads(capsys.readouterr().out)
    assert report["automatic_training"] is False and "channels" not in report
    assert not (tmp_path/"runs").exists()
