"""Frozen Attention AE inference for a new dataset, without fitting or calibration."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import importlib.util
import pickle
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from . import project

sys.path.insert(0,str(project.ROOT/"shared/data_tools"))
from meter_input import normalize

REQUIRED_PACKAGE_FILES = ("best_model.pt", "config.json", "scaler.json", "metadata.json",
                          "calibration/max_sorted.npy", "calibration/mean_sorted.npy")


def local_boundary(value, tz):
    value=pd.Timestamp(value)
    return value.tz_localize(tz) if value.tz is None else value.tz_convert(tz)


def load_package(directory, cutoff):
    directory=project.resolve(directory)
    missing=[name for name in REQUIRED_PACKAGE_FILES if not (directory/name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete trained package {project.relative(directory)}: {', '.join(missing)}")
    for name in REQUIRED_PACKAGE_FILES:
        project.resolve(directory/name)
    config=project.read_json(directory/"config.json")
    scaler=project.read_json(directory/"scaler.json")
    metadata=project.read_json(directory/"metadata.json")
    project.check_window(config["data"])
    if (not 0<=float(config["data"]["max_missing_ratio"])<=1
            or int(config["data"]["max_interpolation_gap"])<0):
        raise ValueError("Invalid trained missing-data policy")
    tz=config["splits"]["timezone"]
    if local_boundary(config["splits"]["train_end"],tz) != cutoff:
        raise ValueError("Trained package cutoff differs from the frozen dataset reference")
    if metadata.get("source") != "held_out_calibration":
        raise ValueError("Attention package must contain frozen held-out calibration metadata")
    if metadata.get("split_config") != config["splits"]:
        raise ValueError("Package training configuration is not bound to metadata")
    if not metadata.get("binding") or metadata.get("score_block_size") != 3 or pd.to_timedelta(metadata["frequency"]) != pd.Timedelta(minutes=5):
        raise ValueError("Package scoring definition is incompatible")
    if metadata.get("score_definition") != "last_block_point_reconstruction_mse_max":
        raise ValueError("Unexpected trained score definition")
    features=config["data"]["features"]
    if scaler.get("features") != features:
        raise ValueError("Frozen scaler feature order differs from the model configuration")
    center=np.asarray(scaler["center"],dtype=np.float32)
    scale=np.asarray(scaler["scale"],dtype=np.float32)
    if center.shape != (len(features),) or scale.shape != center.shape or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError("Invalid frozen scaler")
    if not np.isfinite(metadata["threshold"]) or metadata["threshold"] < 0:
        raise ValueError("Invalid frozen reconstruction threshold")
    hashes={name:project.digest(directory/name) for name in REQUIRED_PACKAGE_FILES}
    expected={"best_model.pt":"model_sha256","scaler.json":"scaler_sha256",
              "calibration/max_sorted.npy":"max_sample_sha256","calibration/mean_sorted.npy":"mean_sample_sha256"}
    for name, field in expected.items():
        if hashes[name] != metadata.get(field):
            raise ValueError(f"Trained artifact checksum mismatch: {name}")
    maximum=np.load(directory/"calibration/max_sorted.npy",allow_pickle=False)
    mean=np.load(directory/"calibration/mean_sorted.npy",allow_pickle=False)
    for values in (maximum,mean):
        if values.ndim != 1 or len(values)<2 or not np.isfinite(values).all() or np.any(values<0) or np.any(np.diff(values)<0):
            raise ValueError("Frozen ECDF arrays must be finite, sorted, nonnegative and nonempty")
    if len(maximum) != len(mean) or len(maximum) != metadata.get("sample_scores"):
        raise ValueError("Frozen ECDF sample count differs from metadata")
    if metadata.get("total_scores",0) < len(maximum):
        raise ValueError("Invalid calibration population/sample count")
    if not isinstance(metadata.get("approximate"),bool) or metadata["approximate"]!=(metadata["total_scores"]>len(maximum)):
        raise ValueError("Calibration approximation flag differs from sample counts")
    return dict(directory=directory,config=config,scaler=scaler,metadata=metadata,
                maximum=maximum,mean=mean,center=center,scale=scale,hashes=hashes)


def make_plan(config_path, *, series_ids=None, max_channels=None):
    config=project.read_json(project.resolve(config_path))
    if (config.get("version") != 1 or config.get("model") != project.MODEL
            or config.get("mode") != "frozen_inference" or config.get("ensemble_eligible") is not False):
        raise ValueError("Use the independent frozen Attention inference configuration")
    if int(config["batch_size"])<1 or config["device"] not in ("auto","cpu","cuda"):
        raise ValueError("Invalid inference runtime settings")
    project.output_path(config["output_directory"])
    profile=project.read_json(project.resolve(config["dataset_profile"]))
    time=profile["time"]; tz=time["timezone"]
    if (time["input_interval_minutes"],time["ensemble_interval_minutes"]) != (5,15):
        raise ValueError("Dataset must retain five-minute input and fifteen-minute output")
    start=local_boundary(time["test_start_inclusive"],tz)
    end=local_boundary(time["test_end_exclusive"],tz)
    cutoff=local_boundary(time["train_end_exclusive"],tz)
    if cutoff>start or start>=end or any(stamp!=stamp.floor("15min") for stamp in (start,end,cutoff)):
        raise ValueError("Invalid frozen reference/target boundaries")
    directory=project.resolve(profile["data"]["shared_input_directory"])
    paths=sorted(path for path in directory.glob("*.csv") if not path.name.startswith("_"))
    if series_ids:
        if len(series_ids)!=len(set(series_ids)):
            raise ValueError("Duplicate selected series IDs")
        missing=set(series_ids)-{path.stem for path in paths}
        if missing:
            raise FileNotFoundError(f"Requested shared inputs missing: {sorted(missing)}")
        paths=[path for path in paths if path.stem in series_ids]
    if max_channels is not None:
        if max_channels<1:
            raise ValueError("max_channels must be positive")
        paths=paths[:max_channels]
    if not paths:
        raise FileNotFoundError("No shared per-channel input CSV found")
    for path in paths:
        project.resolve(path)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",path.stem):
            raise ValueError("Unsafe shared channel filename")
    mapping=project.category_mapping(config["meter_repository"])
    selected_categories={str(mapping[p.stem]["category_id"]) for p in paths if p.stem in mapping}
    packages={}; reports={}
    for category,directory in config["model_packages"].items():
        if category not in selected_categories:
            continue
        try:
            packages[category]=load_package(directory,cutoff)
            reports[category]=dict(status="ready",directory=project.relative(directory))
        except (OSError,ValueError,KeyError) as exc:
            reports[category]=dict(status="unavailable",directory=project.relative(directory),reason=str(exc))
    channels=[]
    for path in paths:
        info=mapping.get(path.stem,{})
        category=str(info.get("category_id",""))
        status=(reports.get(category,{}).get("status","unavailable") if category in config["model_packages"]
                else "unsupported_category")
        channels.append(dict(series_id=path.stem,**info,status=status))
    ready=all(row["status"] in ("ready","unsupported_category") for row in channels)
    report=dict(model=project.MODEL,dataset_id=profile["dataset_id"],
                train_end_exclusive=cutoff.isoformat(),test_start=start.isoformat(),test_end_exclusive=end.isoformat(),
                selected_channels=len(paths),category_packages=reports,channels=channels,
                trained_packages_ready=ready and bool(packages),torch_installed=importlib.util.find_spec("torch") is not None,
                output_directory=config["output_directory"],automatic_training=False,ensemble_eligible=False)
    return report,config,profile,start,end,paths,mapping,packages


def short_gap_fill(array,gap):
    """Same bounded interior linear interpolation as the supplied training code."""
    result=array.copy()
    if gap==0:
        return result
    for feature in range(array.shape[1]):
        x=array[:,feature]; missing=~np.isfinite(x)
        edges=np.diff(np.r_[False,missing,False].astype(np.int8))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if b-a<=gap and a>0 and b<len(x):
                result[a:b,feature]=np.linspace(x[a-1],x[b],b-a+2)[1:-1]
    return result


def prepare_windows(frame,package,start,end):
    data=package["config"]["data"]
    context_start=start-pd.Timedelta(minutes=(288-3)*5)
    index=pd.date_range(context_start,end,freq="5min",inclusive="left")
    selected=frame.loc[frame.timestamp.ge(context_start)&frame.timestamp.lt(end)].set_index("timestamp")
    values=selected.reindex(index)[data["features"]].to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Input magnitudes exceed the trained float32 representation")
    observed=np.isfinite(values).all(axis=1)
    filled=short_gap_fill(values,int(data["max_interpolation_gap"]))
    complete=np.isfinite(filled).all(axis=1)
    intervals=pd.date_range(start,end,freq="15min",inclusive="left")
    starts=np.arange(len(intervals),dtype=np.int64)*3
    obs=np.r_[0,np.cumsum(observed,dtype=np.int64)]
    full=np.r_[0,np.cumsum(complete,dtype=np.int64)]
    counts=obs[starts+288]-obs[starts+285]
    valid=((full[starts+288]-full[starts])==288)&((288-(obs[starts+288]-obs[starts]))<=float(data["max_missing_ratio"])*288+1e-9)
    first=selected.index.min() if len(selected) else end
    missing_history=index[starts]<first
    return dict(values=filled,starts=starts,valid=valid,counts=counts,intervals=intervals,
                missing_history=np.asarray(missing_history),context_start=context_start)


def torch_scorer(package,runtime):
    """Load only a safe inference payload; no optimizer or pickle fallback."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch in an isolated Attention environment; no automatic installation/training") from exc
    from .src.model import LSTMAutoencoder, reconstruction_point_errors
    try:
        checkpoint=torch.load(package["directory"]/"best_model.pt",map_location="cpu",weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("Safe weight loading rejected this payload; request an exported inference package with tensor weights and plain configuration") from exc
    if not isinstance(checkpoint,dict):
        raise ValueError("Expected the supplied pipeline's best-model inference payload")
    state=checkpoint.get("model_state",{})
    if not any(key.startswith("attention.") for key in state):
        raise ValueError("Original LSTM weights cannot be used with the Attention architecture")
    trained=checkpoint.get("config",{})
    scoring_keys=("features","frequency","window_size","stride","score_block_size","max_missing_ratio","max_interpolation_gap")
    if (trained.get("model")!=package["config"]["model"]
            or trained.get("splits")!=package["config"]["splits"]
            or any(trained.get("data",{}).get(key)!=package["config"]["data"].get(key) for key in scoring_keys)):
        raise ValueError("Checkpoint architecture/features differ from the supplied training configuration")
    requested=runtime["device"]
    if requested=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device=torch.device("cuda" if requested=="auto" and torch.cuda.is_available() else "cpu" if requested=="auto" else requested)
    model=LSTMAutoencoder(len(package["config"]["data"]["features"]),**package["config"]["model"]).to(device)
    model.load_state_dict(state,strict=True); model.eval()
    def score(prepared):
        indices=np.flatnonzero(prepared["valid"]); maxima=[]; means=[]; deviations=[]
        with torch.no_grad():
            for offset in range(0,len(indices),int(runtime["batch_size"])):
                batch=indices[offset:offset+int(runtime["batch_size"])]
                windows=np.stack([(prepared["values"][prepared["starts"][i]:prepared["starts"][i]+288]-package["center"])/package["scale"] for i in batch])
                tensor=torch.from_numpy(windows).to(device)
                error=reconstruction_point_errors(tensor,model(tensor))[:,-3:].float()
                maxima.append(error.max(dim=1).values.cpu().numpy()); means.append(error.mean(dim=1).cpu().numpy())
                deviations.append(error.std(dim=1,unbiased=False).cpu().numpy())
        return {key:np.concatenate(parts) if parts else np.empty(0,np.float32)
                for key,parts in (("max",maxima),("mean",means),("std",deviations))}
    return score


def score_frame(key,prepared,package,scorer,start,end,info):
    grid=pd.date_range(start,end,freq="15min",inclusive="left")
    frame=pd.DataFrame(dict(model=project.MODEL,series_id=key,interval_start=grid,interval_end=grid+pd.Timedelta(minutes=15)))
    frame["window_start"]=grid-pd.Timedelta(minutes=285*5)
    frame["window_end"]=frame.interval_end
    for field in ("raw_score","max_score","mean_score","score_std","max_percentile","mean_percentile"):
        frame[field]=np.nan
    frame["available"]=False
    frame["is_anomaly"]=pd.Series(pd.NA,index=frame.index,dtype="boolean")
    frame["valid_point_count"]=pd.Series(pd.NA,index=frame.index,dtype="Int64")
    frame["expected_point_count"]=3; frame["coverage_ratio"]=np.nan
    frame["source_resolution_minutes"]=5
    frame["aggregation_method"]="last_15min_of_causal_24h_window"
    frame["threshold"]=np.nan; frame["data_status"]="unsupported_category"
    frame["calibration_version"]="unavailable_no_category_model"; frame["model_version"]=""
    frame["calibration_source"]=None; frame["calibration_approximate"]=None; frame["training_end_exclusive"]=None
    frame["ensemble_eligible"]=False
    for field in ("category_id","category_name","channel_id"):
        frame[field]=info.get(field)
    if package is None:
        return frame
    metadata=package["metadata"]; valid=prepared["valid"]
    scores=scorer(prepared)
    for name,column in (("max","max_score"),("mean","mean_score"),("std","score_std")):
        values=np.asarray(scores[name])
        if len(values)!=int(valid.sum()) or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError("Invalid model reconstruction score block")
        frame.loc[valid,column]=values
    frame["raw_score"]=frame.max_score
    for name,reference in (("max",package["maximum"]),("mean",package["mean"])):
        frame.loc[valid,name+"_percentile"]=np.searchsorted(reference,frame.loc[valid,name+"_score"],side="right")/len(reference)
    frame["threshold"]=float(metadata["threshold"])
    frame["available"]=valid
    frame.loc[valid,"is_anomaly"]=frame.loc[valid,"max_score"]>metadata["threshold"]
    frame["valid_point_count"]=prepared["counts"]
    frame["coverage_ratio"]=prepared["counts"]/3
    frame["data_status"]=np.where(valid,np.where(prepared["counts"]==3,"observed","interpolated"),np.where(prepared["missing_history"],"insufficient_history","missing_data"))
    frame["calibration_version"]=metadata["binding"]; frame["model_version"]=metadata["binding"]
    frame["calibration_source"]=metadata["source"]; frame["calibration_approximate"]=bool(metadata["approximate"])
    frame["training_end_exclusive"]=package["config"]["splits"]["train_end"]
    return frame


def event_frame(frame,dataset_id):
    columns=["model","event_id","series_id","start","end","duration_minutes","max_raw_score","mean_raw_score","max_score_percentile","threshold","interval_count","calibration_version","model_version"]
    selected=frame.loc[frame.available&frame.is_anomaly.fillna(False)].copy()
    rows=[]
    if len(selected):
        selected["_event"]=(selected.interval_start.diff().ne(pd.Timedelta(minutes=15))).cumsum()
        for _,group in selected.groupby("_event"):
            first=group.iloc[0]; start=first.interval_start; end=group.interval_end.iloc[-1]
            identity=f"{project.MODEL}|{dataset_id}|{first.model_version}|{first.series_id}|{start.isoformat()}|{end.isoformat()}"
            rows.append(dict(model=project.MODEL,event_id=sha1(identity.encode()).hexdigest()[:16],series_id=first.series_id,
                start=start,end=end,duration_minutes=int((end-start).total_seconds()/60),max_raw_score=group.max_score.max(),
                mean_raw_score=group.mean_score.mean(),max_score_percentile=group.max_percentile.max(),threshold=first.threshold,
                interval_count=len(group),calibration_version=first.calibration_version,model_version=first.model_version))
    return pd.DataFrame(rows,columns=columns)


def run_inference(config_path,*,output_dir=None,series_ids=None,max_channels=None,_scorer_factory=None):
    report,config,profile,start,end,paths,mapping,packages=make_plan(config_path,series_ids=series_ids,max_channels=max_channels)
    if not report["trained_packages_ready"]:
        reasons=[str(key)+": "+value.get("reason",value["status"]) for key,value in report["category_packages"].items() if value["status"]!="ready"]
        raise FileNotFoundError("Trained Attention packages are not ready: "+"; ".join(reasons))
    # Test injection is internal only; the CLI always uses safe PyTorch loading.
    output=project.output_path(output_dir or config["output_directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Attention inference output is nonempty; choose a new directory")
    input_directory=project.resolve(profile["data"]["shared_input_directory"])
    for protected in [input_directory,*[package["directory"] for package in packages.values()]]:
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError("Inference output must be separate from inputs and trained packages")
    factory=_scorer_factory or torch_scorer
    scorers={category:factory(package,config) for category,package in packages.items()}
    output.mkdir(parents=True,exist_ok=True)
    (output/"channels").mkdir()
    metadata=dict(model=project.MODEL,status="running",dataset_id=profile["dataset_id"],created_at_utc=datetime.now(timezone.utc).isoformat(),
        time=profile["time"],window_minutes=1440,stride_minutes=15,scoring_tail_points=3,
        fitting_performed=False,calibration_updated=False,ensemble_eligible=False,
        context_role="observed immediately preceding timestamps only; excluded from fitting/calibration",
        interpolation="bounded internal linear gaps as trained; offline, not strict real-time",
        selected_series=[path.stem for path in paths],input_sha256={},
        model_packages={key:dict(directory=project.relative(value["directory"]),sha256=value["hashes"],binding=value["metadata"]["binding"]) for key,value in packages.items()},
        code_sha256={project.relative(path):project.digest(path) for path in (project.ROOT/"methods"/project.MODEL/"inference.py",project.ROOT/"methods"/project.MODEL/"project.py",project.ROOT/"methods"/project.MODEL/"src/model.py")})
    target=output/"run_metadata.json"; project.write_json(target,metadata)
    project.write_json(output/"inference_config.json",config); project.write_json(output/"dataset_profile.json",profile)
    summaries=[]
    try:
        for position,path in enumerate(paths,1):
            info=mapping.get(path.stem,{})
            category=str(info.get("category_id","")); package=packages.get(category)
            prepared=None
            if package is not None:
                before=project.digest(path)
                raw=normalize(pd.read_csv(path),profile["time"]["timezone"])
                if before!=project.digest(path):
                    raise ValueError("Shared input changed during reading")
                prepared=prepare_windows(raw,package,start,end)
                metadata["input_sha256"][project.relative(path)]=before
            frame=score_frame(path.stem,prepared,package,scorers.get(category),start,end,info)
            events=event_frame(frame,profile["dataset_id"])
            frame.to_csv(output/"channels"/(path.stem+"_scores_15min.csv"),index=False)
            frame.to_csv(output/"ALL_SCORES_15MIN.csv",mode="a",header=not (output/"ALL_SCORES_15MIN.csv").exists(),index=False)
            events.to_csv(output/"ALL_EVENTS.csv",mode="a",header=not (output/"ALL_EVENTS.csv").exists(),index=False)
            summaries.append(dict(model=project.MODEL,series_id=path.stem,category_id=info.get("category_id"),
                intervals=len(frame),available_intervals=int(frame.available.sum()),anomalies=int(frame.is_anomaly.fillna(False).sum()),
                events=len(events),status="scored" if package is not None else "unsupported_category"))
            print(f"Attention AE {position}/{len(paths)} {path.stem}: {int(frame.available.sum())}/{len(frame)} available",flush=True)
        pd.DataFrame(summaries).to_csv(output/"run_summary.csv",index=False)
        pd.DataFrame(report["channels"]).to_csv(output/"channel_status.csv",index=False)
        for package in packages.values():
            if any(project.digest(package["directory"]/name)!=sha for name,sha in package["hashes"].items()):
                raise ValueError("Frozen model package changed during inference")
        metadata["status"]="complete"
        metadata["output_sha256"]={project.relative(path):project.digest(path) for path in sorted(output.rglob("*.csv"))}
        project.write_json(target,metadata)
        return output
    except Exception as exc:
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        project.write_json(target,metadata)
        raise
