from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    MeterWindowDataset,
    chronological_date_split,
    clean_series,
    fit_global_scaler,
    load_series,
)
from .model import LSTMAutoencoder, reconstruction_errors, reconstruction_point_errors
from .utils import save_json, select_device, set_seed

LOG = logging.getLogger(__name__)


def prepare_data(config: dict[str, Any]):
    d = config["data"]
    raw = load_series(
        d["root"], d["extensions"], d["timestamp_column"], d["features"], d.get("meter_limit")
    )
    train: dict[str, pd.DataFrame] = {}
    val: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    split = config["splits"]
    score_block_size = d.get("score_block_size", d["stride"])
    if score_block_size != d["stride"]:
        raise ValueError("score_block_size must equal stride for gap-free 15-minute outputs")
    for key, frame in raw.items():
        train_raw, val_raw, test_raw = chronological_date_split(
            frame,
            split["train_end"],
            split["test_start"],
            split["test_end"],
            split["timezone"],
            split["validation_ratio"],
        )
        train[key] = clean_series(train_raw, d["frequency"], d["max_interpolation_gap"])[0]
        val[key] = clean_series(val_raw, d["frequency"], d["max_interpolation_gap"])[0]
        # Test scoring is causal: the pre-August tail is context only, while each
        # window is responsible solely for its final 15-minute block.
        baseline_raw = pd.concat([train_raw, val_raw]).sort_index()
        baseline_clean = clean_series(
            baseline_raw, d["frequency"], d["max_interpolation_gap"]
        )[0]
        test_clean = clean_series(test_raw, d["frequency"], d["max_interpolation_gap"])[0]
        context_rows = d["window_size"] - score_block_size
        scoring = pd.concat([baseline_clean.tail(context_rows), test_clean]).sort_index()
        full_index = pd.date_range(
            scoring.index.min(), scoring.index.max(), freq=d["frequency"], tz=scoring.index.tz
        )
        test[key] = scoring.reindex(full_index)
    scaler = fit_global_scaler(list(train.values()), d.get("scaler_max_rows", 1_000_000))
    args = (scaler, d["window_size"], d["stride"], d["max_missing_ratio"])
    return (
        MeterWindowDataset.from_frames(train, *args),
        MeterWindowDataset.from_frames(val, *args),
        MeterWindowDataset.from_frames(test, *args),
        scaler,
    )


@torch.no_grad()
def score_dataset(model, loader, device) -> np.ndarray:
    model.eval()
    scores = []
    for batch in loader:
        batch = batch.to(device)
        scores.append(reconstruction_errors(batch, model(batch)).cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def score_interval_blocks(model, loader, device, block_size: int) -> dict[str, np.ndarray]:
    """Score only the final block of each causal context window."""
    model.eval()
    maxima, means, deviations = [], [], []
    for batch in loader:
        batch = batch.to(device)
        point_errors = reconstruction_point_errors(batch, model(batch))[:, -block_size:]
        maxima.append(point_errors.max(dim=1).values.cpu().numpy())
        means.append(point_errors.mean(dim=1).cpu().numpy())
        deviations.append(point_errors.std(dim=1, unbiased=False).cpu().numpy())
    return {
        "max": np.concatenate(maxima),
        "mean": np.concatenate(means),
        "std": np.concatenate(deviations),
    }


def frozen_ecdf(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(baseline, dtype=float)[np.isfinite(baseline)])
    if reference.size == 0:
        raise ValueError("Training score distribution is empty")
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    result[finite] = np.searchsorted(reference, np.asarray(values)[finite], side="right") / reference.size
    return result


def train(config: dict[str, Any]) -> Path:
    set_seed(config["seed"])
    d = config["data"]
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, scaler = prepare_data(config)
    t = config["training"]
    device = select_device(t["device"])
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"])
    model = LSTMAutoencoder(input_size=len(config["data"]["features"]), **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    criterion = nn.MSELoss()
    history = []
    best_loss = float("inf")
    bad_epochs = 0
    checkpoint = output / "best_model.pt"
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            optimizer.step()
            losses.append(loss.item())
        val_scores = score_dataset(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(val_scores.mean())}
        history.append(row)
        LOG.info("epoch=%d train=%.6f val=%.6f", epoch, row["train_loss"], row["val_loss"])
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": config}, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    block_size = d.get("score_block_size", d["stride"])
    calibration_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"]
    )
    calibration = score_interval_blocks(model, calibration_loader, device, block_size)
    threshold = float(np.quantile(calibration["max"], t["threshold_quantile"]))
    np.savez_compressed(
        output / "calibration_scores.npz",
        max_scores=calibration["max"],
        mean_scores=calibration["mean"],
    )
    joblib.dump(scaler, output / "scaler.joblib")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    calibration_version = f"train_pre_{config['splits']['train_end']}_v1"
    save_json({"threshold": threshold, "device": str(device), "train_windows": len(train_ds),
               "val_windows": len(val_ds), "test_windows": len(test_ds),
               "calibration_source": "train_only", "calibration_file": "calibration_scores.npz",
               "score_definition": "last_15min_point_reconstruction_error_max",
               "calibration_version": calibration_version},
              output / "metadata.json")
    plot_history(history, output / "training_history.png")
    predict(config, checkpoint, output / "scaler.joblib", threshold, dataset=test_ds,
            calibration_values=calibration)
    return output


def predict(config, checkpoint_path, scaler_path, threshold, dataset=None, calibration_values=None) -> Path:
    d, t = config["data"], config["training"]
    output = Path(config["output"]["directory"])
    device = select_device(t["device"])
    if dataset is None:
        raw = load_series(d["root"], d["extensions"], d["timestamp_column"], d["features"], d.get("meter_limit"))
        frames = {}
        split = config["splits"]
        block_size = d.get("score_block_size", d["stride"])
        for key, frame in raw.items():
            train_raw, val_raw, test_raw = chronological_date_split(
                frame, split["train_end"], split["test_start"], split["test_end"],
                split["timezone"], split["validation_ratio"],
            )
            baseline = clean_series(
                pd.concat([train_raw, val_raw]).sort_index(), d["frequency"],
                d["max_interpolation_gap"],
            )[0]
            test = clean_series(test_raw, d["frequency"], d["max_interpolation_gap"])[0]
            context_rows = d["window_size"] - block_size
            scoring = pd.concat([baseline.tail(context_rows), test]).sort_index()
            full_index = pd.date_range(
                scoring.index.min(), scoring.index.max(), freq=d["frequency"], tz=scoring.index.tz
            )
            frames[key] = scoring.reindex(full_index)
        dataset = MeterWindowDataset.from_frames(frames, joblib.load(scaler_path), d["window_size"],
                                                  d["stride"], d["max_missing_ratio"])
    model = LSTMAutoencoder(input_size=len(d["features"]), **config["model"]).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    loader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"])
    block_size = d.get("score_block_size", d["stride"])
    scores = score_interval_blocks(model, loader, device, block_size)
    if calibration_values is None:
        calibration_path = output / "calibration_scores.npz"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Frozen training calibration missing: {calibration_path}")
        stored = np.load(calibration_path)
        calibration_values = {"max": stored["max_scores"], "mean": stored["mean_scores"]}
    max_percentiles = frozen_ecdf(scores["max"], calibration_values["max"])
    mean_percentiles = frozen_ecdf(scores["mean"], calibration_values["mean"])
    step = pd.to_timedelta(d["frequency"])
    block_duration = step * block_size
    rows = []
    for index, meta in enumerate(dataset.metadata):
        interval_end = meta.end + step
        interval_start = interval_end - block_duration
        rows.append({
            "model": "lstm_autoencoder", "series_id": meta.series_id,
            "interval_start": interval_start, "interval_end": interval_end,
            "window_start": meta.start, "window_end": interval_end,
            "raw_score": float(scores["max"][index]),
            "anomaly_score": float(scores["max"][index]),
            "max_score": float(scores["max"][index]),
            "mean_score": float(scores["mean"][index]),
            "score_std": float(scores["std"][index]),
            "max_percentile": float(max_percentiles[index]),
            "mean_percentile": float(mean_percentiles[index]),
            "threshold": threshold, "is_anomaly": bool(scores["max"][index] > threshold),
            "valid_point_count": block_size, "expected_point_count": block_size,
            "coverage_ratio": 1.0, "source_resolution_minutes": 5,
            "aggregation_method": "last_15min_of_causal_24h_window",
            "data_status": "native_5m",
            "calibration_version": f"train_pre_{config['splits']['train_end']}_v1",
            "available": True,
        })
    result = pd.DataFrame(rows).sort_values(["series_id", "window_start"])
    target = output / "anomaly_windows.csv"
    result.to_csv(target, index=False)
    result.to_csv(output / "scores_15min.csv", index=False)
    result.groupby("series_id", as_index=False).agg(
        windows=("is_anomaly", "size"), anomalies=("is_anomaly", "sum"),
        max_score=("anomaly_score", "max"), mean_score=("anomaly_score", "mean")
    ).sort_values("max_score", ascending=False).to_csv(output / "meter_summary.csv", index=False)
    plot_scores(result, output / "anomaly_scores.png")
    return target


def plot_history(history, path):
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["epoch"], frame["train_loss"], label="train")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation")
    ax.set(xlabel="Epoch", ylabel="MSE", title="Training history")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_scores(result, path):
    top = result.groupby("series_id")["anomaly_score"].max().nlargest(8).index
    view = result[result["series_id"].isin(top)]
    fig, axes = plt.subplots(len(top), 1, figsize=(12, max(3, 2 * len(top))), squeeze=False)
    for ax, series_id in zip(axes[:, 0], top):
        part = view[view["series_id"] == series_id]
        ax.plot(part["window_end"], part["anomaly_score"], linewidth=.8)
        ax.axhline(part["threshold"].iloc[0], color="red", linestyle="--")
        ax.set_title(series_id, loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
