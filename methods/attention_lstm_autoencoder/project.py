"""Project-relative configuration and input views; no model fitting here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL = "attention_lstm_autoencoder"
FEATURES = ["powerFactor", "pRealKw", "pRealPositiveKw", "pRealNegativeKw",
            "pReactiveKw", "pReactivePositiveKw", "pReactiveNegativeKw",
            "vRMSMin", "vRMSMax", "iRMSMin", "iRMSMax"]


def resolve(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Paths must stay inside group14reshape")
    return path


def relative(value):
    return resolve(value).relative_to(ROOT.resolve()).as_posix()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def output_path(value):
    path = resolve(value)
    parent = (ROOT/"runs"/MODEL).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f"Use a separate output under runs/{MODEL}")
    return path


def load_training_config(value):
    config = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if config.get("method") != MODEL:
        raise ValueError("Use the independent Attention LSTM configuration")
    check_window(config["data"])
    for section, key in (("data","root"),("data","cache_directory"),("data","input_view_directory"),("output","directory")):
        config[section][key] = str(resolve(config[section][key]))
    output_path(config["output"]["directory"])
    for key in ("cache_directory", "input_view_directory"):
        if not Path(config["data"][key]).is_relative_to((ROOT/"data/processed"/MODEL).resolve()):
            raise ValueError("Attention training cache/views must stay in their separate processed-data directory")
    return config


def check_window(data):
    if (pd.to_timedelta(data["frequency"]) != pd.Timedelta(minutes=5)
            or (data["window_size"], data["stride"], data["score_block_size"]) != (288,3,3)):
        raise ValueError("The project contract is 5-minute input, 24-hour context and 15-minute scoring")
    features = data["features"]
    if not features or len(features) != len(set(features)) or not set(features).issubset(FEATURES):
        raise ValueError("Features must be an ordered unique subset of the eleven shared power features")


def training_input_view(config):
    """Exclude _manifest CSV without changing the imported preprocessing algorithm."""
    if os.name != "posix":
        raise RuntimeError("Imported training/replay uses POSIX locks/signals; use Spartan/Linux/WSL")
    source = resolve(config["data"]["root"])
    view = resolve(config["data"]["input_view_directory"])
    if source == view or source in view.parents or view in source.parents:
        raise ValueError("Input view must be separate from original input")
    paths = sorted(path for path in source.glob("*.csv") if not path.name.startswith("_"))
    if not paths:
        raise FileNotFoundError("No shared per-channel input CSV found")
    inventory = []
    for path in paths:
        resolve(path)
        columns = list(pd.read_csv(path,nrows=0).columns)
        if columns != ["timestamp", *FEATURES]:
            raise ValueError(f"Expected the shared twelve-column schema: {path.name}")
        inventory.append(dict(name=path.name,path=str(path.resolve()),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
    marker = view/"_view_metadata.json"
    if view.exists() and any(view.iterdir()):
        if not marker.exists() or read_json(marker).get("inventory") != inventory:
            raise ValueError("Training input changed; use new input-view/cache/output paths")
        if {p.name for p in view.iterdir()} != {p.name for p in paths} | {marker.name}:
            raise ValueError("Unexpected file in Attention training input view")
        for path in paths:
            if not (view/path.name).is_symlink() or (view/path.name).resolve() != path.resolve():
                raise ValueError("Training input view link changed")
    else:
        view.mkdir(parents=True,exist_ok=True)
        for path in paths:
            (view/path.name).symlink_to(path.resolve())
        write_json(marker,dict(inventory=inventory,source=relative(source),original_data_modified=False))
    native = json.loads(json.dumps(config))
    native["data"]["root"] = str(view)
    return native


def category_mapping(repository):
    """Use the teammate's numeric C-order rule, not a guessed C(n+1) mapping."""
    repository = read_json(resolve(repository))
    result = {}
    seen_channels = set()
    for device_key, device in repository.items():
        device_id = str(device.get("device_id",device_key))
        channels = device.get("channels",[])
        ranked = []
        for channel in channels:
            prefix = device_id+"_C"
            channel_id = str(channel["id"])
            suffix = channel_id.removeprefix(prefix)
            if not channel_id.startswith(prefix) or not suffix.isdigit() or channel_id in seen_channels:
                raise ValueError("Invalid or duplicate registered channel")
            seen_channels.add(channel_id)
            category = channel["category_id"]
            if not isinstance(category,int) or isinstance(category,bool):
                raise ValueError("Category IDs must be integers")
            ranked.append((int(suffix),channel))
        if len({number for number,_ in ranked})!=len(ranked):
            raise ValueError("Duplicate numeric channel suffix")
        for index, (_,channel) in enumerate(sorted(ranked,key=lambda item:item[0])):
            value=dict(category_id=channel["category_id"],category_name=channel["category_name"],channel_id=str(channel["id"]))
            key=f"{device_id}_{index}"
            if key in result:
                raise ValueError("Duplicate CSV channel mapping")
            result[key]=value
            result[str(channel["id"])]=value
    return result
