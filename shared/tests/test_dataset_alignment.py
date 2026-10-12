import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"shared/scripts"))
sys.path.insert(0,str(ROOT/"shared/data_tools"))
import align_dataset_workflow as workflow
import build_ensemble_features as ensemble
from meter_input import COLUMNS, normalize


def profile():
    value=json.loads((ROOT/"configs/datasets/september_2026.json").read_text())
    value["time"]["test_end_exclusive"]="2026-09-02"
    return value


def scores(model="matrix_profile",key="DD_TEST_0"):
    index=pd.date_range("2026-09-01",periods=96,freq="15min",tz="Australia/Sydney")
    values=np.linspace(.1,.99,96)
    frame=pd.DataFrame(dict(model=model,series_id=key,interval_start=index,
        interval_end=index+pd.Timedelta(minutes=15),raw_score=values,max_score=values,
        mean_score=values,score_std=0.,max_percentile=values,mean_percentile=values,
        threshold=.995,is_anomaly=False,coverage_ratio=1.,source_resolution_minutes=30,
        aggregation_method="repeat_from_native_30m",data_status="upsampled_from_30m",
        calibration_version="fixed_2026-07-01_2026-07-31_v2",available=True))
    return ensemble.validate(frame,"Australia/Sydney")


def event():
    return dict(event_id="event-test",device_id="DD_TEST",channel_idx=0,
        start_time="2026-09-01T00:15:00+10:00",end_time="2026-09-01T01:15:00+10:00",
        duration_minutes=60,event_type="shape_change",confidence="high",
        review_queue="priority_alert",priority_segments=json.dumps([dict(
            start_time="2026-09-01T00:15:00+10:00",end_time="2026-09-01T00:30:00+10:00")]))


def test_profile_keeps_train_test_and_context_disjoint(tmp_path):
    path=tmp_path/"profile.json"; value=profile(); path.write_text(json.dumps(value))
    actual,periods=workflow.load_profile(path)
    assert periods["train_end_exclusive"]==pd.Timestamp("2026-08-01",tz="Australia/Sydney")
    assert periods["context_start_inclusive"]==pd.Timestamp("2026-08-31",tz="Australia/Sydney")
    bad=copy.deepcopy(value); bad["time"]["context_start_inclusive"]="2026-07-31"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="disjoint"): workflow.load_profile(path)


def test_profile_cannot_change_established_windows(tmp_path):
    path=tmp_path/"profile.json"; value=profile()
    value["window_policy"]["neural_stride_points"]=1; path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="neural window"): workflow.load_profile(path)


def test_subsecond_input_is_not_silently_on_grid():
    frame=pd.DataFrame({column:[1.] for column in COLUMNS})
    frame.timestamp=["2026-09-01T00:00:00.100+10:00"]
    with pytest.raises(ValueError,match="5-minute grid"): normalize(frame,"Australia/Sydney")


def test_actual_priority_half_is_not_copied_from_30_minutes():
    data=scores(); events=pd.DataFrame([event()])
    grid=workflow.review_grid(events,data,data.interval_start.iloc[0],data.interval_end.iloc[-1],"Australia/Sydney")
    assert grid.priority_alert_overlap.iloc[:4].tolist()==[False,True,False,False]
    assert grid.priority_core_minutes.iloc[:4].tolist()==[0.,15.,0.,0.]
    assert grid.review_state.iloc[2]=="source_event_noncore_context"


def test_missing_model_score_preserves_unknown_not_normal():
    data=scores(); data.loc[1,"available"]=False
    grid=workflow.review_grid(pd.DataFrame([event()]),data,data.interval_start.iloc[0],data.interval_end.iloc[-1],"Australia/Sydney")
    assert pd.isna(grid.priority_alert_overlap.iloc[1])
    assert pd.isna(grid.priority_core_minutes.iloc[1])
    assert grid.review_state.iloc[1]=="unavailable_model_score"
    assert not grid.triage_available.iloc[1]


def test_no_source_event_is_not_a_normal_label():
    data=scores(); events=pd.DataFrame([event()]).iloc[:0]
    grid=workflow.review_grid(events,data,data.interval_start.iloc[0],data.interval_end.iloc[-1],"Australia/Sydney")
    assert grid.review_state.eq("no_source_event").all()
    assert not grid.priority_alert_overlap.any()
    assert grid.triage_available.all()


def test_queues_can_overlap_on_a_fifteen_minute_bin():
    data=scores(); events=pd.DataFrame([event(),{**event(),"event_id":"quality",
         "review_queue":"data_quality","start_time":"2026-09-01T00:20:00+10:00",
         "end_time":"2026-09-01T00:25:00+10:00","priority_segments":"[]"}])
    grid=workflow.review_grid(events,data,data.interval_start.iloc[0],data.interval_end.iloc[-1],"Australia/Sydney")
    assert grid.priority_alert_overlap.iloc[1] and grid.data_quality_overlap.iloc[1]


def test_nullable_csv_flag_import_does_not_turn_false_into_true(tmp_path):
    data=scores(); data.loc[1,"available"]=False
    frame=workflow.review_grid(pd.DataFrame([event()]),data,data.interval_start.iloc[0],data.interval_end.iloc[-1],"Australia/Sydney")
    path=tmp_path/"flags.csv"; frame.to_csv(path,index=False)
    restored=workflow.read_review_flags(path)
    assert restored.priority_alert_overlap.iloc[0]==False
    assert pd.isna(restored.priority_alert_overlap.iloc[1])
    assert str(restored.priority_alert_overlap.dtype)=="boolean"
    assert sum(len(chunk) for chunk in workflow.read_review_flags(path,chunksize=10))==len(frame)


def test_other_model_training_cutoff_cannot_move_to_test_month():
    periods={"train_end_exclusive":pd.Timestamp("2026-08-01",tz="Australia/Sydney")}
    part=pd.DataFrame(dict(calibration_version=["train_pre_2026-08-01_v1"]))
    workflow.validate_saved_calibration(part,"rpca",periods)
    part.calibration_version="train_pre_2026-09-01_v1"
    with pytest.raises(ValueError,match="training cutoff"):
        workflow.validate_saved_calibration(part,"rpca",periods)


def test_stream_duplicate_keys_and_wrong_period_are_rejected(tmp_path):
    path=tmp_path/"scores.csv"; data=scores()
    pd.concat([data,data.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match="Duplicate"):
        list(workflow.score_series(path,"matrix_profile",data.interval_start.iloc[0],data.interval_end.iloc[-1],"Australia/Sydney",chunksize=20))
    data.to_csv(path,index=False)
    with pytest.raises(ValueError,match="no target-period"):
        list(workflow.score_series(path,"matrix_profile",pd.Timestamp("2026-10-01",tz="Australia/Sydney"),pd.Timestamp("2026-11-01",tz="Australia/Sydney"),"Australia/Sydney"))


def test_directory_discovery_reuses_existing_contract(tmp_path):
    path=tmp_path/"a_scores_15min.csv"; path.touch()
    assert ensemble.discover_score_files(tmp_path)==[path]
    preferred=tmp_path/"ALL_SCORES_15MIN.csv"; preferred.touch()
    assert ensemble.discover_score_files(tmp_path)==[preferred]


def test_queue_descriptions_never_claim_four_model_consensus():
    for queue in workflow.QUEUES:
        assert "four models" not in workflow.review_selection_reason(queue).lower()
    with pytest.raises(KeyError): workflow.review_selection_reason("unknown")


def test_plot_flag_label_preserves_missing_model_count():
    import plot_manual_review_set as plot
    assert plot.flag_label(dict(flagged_model_count=1,available_model_count=1))=="1/1 available models flagged (4 configured)"
    assert plot.flag_label(dict(flagged_model_count=4,available_model_count=4))=="4/4 flagged at anchor"


def make_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,"ROOT",tmp_path)
    value=profile(); value["data"]=dict(raw_target_source="raw",original_history_directory="history",
        prepared_source_directory="prepared",shared_input_directory="shared_input",injected_directory="injected")
    value["models"]["matrix_profile"]=dict(score_input="run/ALL_SCORES_15MIN.csv",run_directory="run")
    value["review"]=dict(triage_directory="triage",policy_config="policy.json")
    for name in ("prepared","run","raw/31_08_2026","triage"):
        (tmp_path/name).mkdir(parents=True)
    def raw(day):
        index=pd.date_range(day,periods=288,freq="5min",tz="Australia/Sydney")
        frame=pd.DataFrame({column:np.arange(288)+number+.25 for number,column in enumerate(COLUMNS)})
        frame.timestamp=index
        return frame
    pd.concat([raw("2026-07-01"),raw("2026-09-01")]).to_csv(tmp_path/"prepared/DD_TEST_0.csv",index=False)
    pd.DataFrame([dict(meter_channel="DD_TEST_0",n_rows=576)]).to_csv(tmp_path/"prepared/_manifest.csv",index=False)
    raw("2026-08-31").to_csv(tmp_path/"raw/31_08_2026/DD_TEST_0.csv",index=False)
    (tmp_path/"prepared/input_signature.json").write_text(json.dumps(dict(signature="old")))
    metadata=dict(status="complete",baseline_start="2026-07-01",baseline_end="2026-07-31",detection_start="2026-09-01",
                  detection_end_exclusive="2026-09-02",timezone="Australia/Sydney",history_mode="fixed",signature="old")
    (tmp_path/"run/run_metadata.json").write_text(json.dumps(metadata))
    data=scores(); data.loc[0,"available"]=False; data.to_csv(tmp_path/"run/ALL_SCORES_15MIN.csv",index=False)
    row=event(); original={key:item for key,item in row.items() if key not in ("review_queue","priority_segments")}
    pd.DataFrame([original]).to_csv(tmp_path/"run/ALL_EVENTS.csv",index=False)
    pd.DataFrame([row]).to_csv(tmp_path/"triage/events_all_classified.csv",index=False)
    pd.DataFrame([dict(review_unit_id="unit",review_queue="priority_alert",start_time=row["start_time"],end_time=row["end_time"],
                      source_event_count=1,event_ids='["event-test"]',series_ids='["DD_TEST_0"]')]).to_csv(tmp_path/"triage/review_units.csv",index=False)
    pd.DataFrame([dict(event_id=row["event_id"],series_id="DD_TEST_0",review_unit_id="unit",review_queue="priority_alert")]).to_csv(tmp_path/"triage/event_memberships.csv",index=False)
    (tmp_path/"policy.json").write_text("{}")
    review=dict(status="complete",baseline_start="2026-07-01",baseline_end="2026-07-31",detection_start="2026-09-01",detection_end_exclusive="2026-09-02",
                policy_config_sha256=workflow.digest(tmp_path/"policy.json"),queue_counts=[],original_production_sha256={
                    "run/ALL_EVENTS.csv":workflow.digest(tmp_path/"run/ALL_EVENTS.csv"),
                    "run/ALL_SCORES_15MIN.csv":workflow.digest(tmp_path/"run/ALL_SCORES_15MIN.csv")})
    (tmp_path/"triage/triage_summary.json").write_text(json.dumps(review))
    path=tmp_path/"profile.json"; path.write_text(json.dumps(value))
    return workflow.load_profile(path)


def test_end_to_end_preserves_scores_and_context_roles(tmp_path,monkeypatch):
    value,periods=make_fixture(tmp_path,monkeypatch)
    before=workflow.digest(tmp_path/"run/ALL_SCORES_15MIN.csv")
    workflow.verify_model_period(value,periods)
    dataset=workflow.prepare_dataset(value,periods)
    raw=pd.read_csv(tmp_path/"shared_input/DD_TEST_0.csv")
    assert len(raw)==864 and raw.timestamp.str[:10].value_counts().to_dict()=={"2026-07-01":288,"2026-08-31":288,"2026-09-01":288}
    assert list(raw.columns)==COLUMNS
    assert pd.read_csv(tmp_path/"shared_input/_manifest.csv").context_only_rows.iloc[0]==288
    assert workflow.prepare_dataset(value,periods)["dataset_signature"]==dataset["dataset_signature"]
    meta=workflow.export_dataset(value,periods,dataset,tmp_path/"aligned")
    assert meta["score_rows"]==96 and meta["source_formal_events"]==1
    exported=pd.read_csv(tmp_path/"aligned/all_models_scores_15min.csv")
    pd.testing.assert_frame_equal(exported,pd.read_csv(tmp_path/"run/ALL_SCORES_15MIN.csv"))
    assert not exported.interval_start.str.startswith("2026-08").any()
    assert workflow.digest(tmp_path/"run/ALL_SCORES_15MIN.csv")==before
    wide=pd.read_csv(tmp_path/"aligned/ensemble_features_15min.csv")
    assert not wide.lstm_autoencoder_available.any() and wide.lstm_autoencoder_max_score.isna().all()
    assert wide.matrix_profile_priority_alert_overlap.iloc[1]==True
    formal=pd.read_csv(tmp_path/"aligned/review/model_events.csv")
    assert ensemble.build_events(scores()).empty
    assert formal.event_id.tolist()==["event-test"]
    assert {"model","series_id","start","end","max_raw_score","mean_raw_score","interval_count"}.issubset(formal.columns)
    with pytest.raises(FileExistsError): workflow.export_dataset(value,periods,dataset,tmp_path/"aligned")


def test_shared_input_tampering_and_injection_source_are_rejected(tmp_path,monkeypatch):
    value,periods=make_fixture(tmp_path,monkeypatch)
    workflow.prepare_dataset(value,periods)
    with (tmp_path/"shared_input/DD_TEST_0.csv").open("a") as handle: handle.write("changed")
    with pytest.raises(ValueError,match="Shared inputs changed"): workflow.prepare_dataset(value,periods)
    bad=copy.deepcopy(value); bad["data"]["raw_target_source"]="injected/raw"
    with pytest.raises(ValueError,match="Injected measurements"): workflow.prepare_dataset(bad,periods)
