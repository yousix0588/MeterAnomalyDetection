#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJECTED_DIR = PROJECT_ROOT / "data" / "injected_meters"
ENSEMBLE_CSV = PROJECT_ROOT / "runs" / "ensemble" / "injected_ensemble" / "all_models_scores_15min.csv"

# Load GT
gt_map = {}
for p in sorted(INJECTED_DIR.glob("*.csv")):
    if p.name.startswith("_"):
        continue
    sid = p.stem
    df = pd.read_csv(p)
    if "pRealKw" in df.columns and "pRealKw_original" in df.columns:
        diff = (pd.to_numeric(df["pRealKw"], errors="coerce") - pd.to_numeric(df["pRealKw_original"], errors="coerce")).abs()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        gt_map[sid] = df.set_index("timestamp")["is_inj"] = (diff > 1e-4)

# Load all scores
df = pd.read_csv(ENSEMBLE_CSV)
df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True)
df["is_anomaly"] = df["is_anomaly"].astype(str).str.lower().isin(["true", "1"])

# Pivot
pivot_anom = df.pivot(index=["series_id", "interval_start", "interval_end"], columns="model", values="is_anomaly").fillna(False)

# Match GT
gt_flags = []
for sid, start, end in pivot_anom.index:
    p = INJECTED_DIR / f"{sid}.csv"
    gt = False
    if p.exists():
        cdf = pd.read_csv(p)
        if "pRealKw_original" in cdf.columns:
            cdf["ts"] = pd.to_datetime(cdf["timestamp"], utc=True)
            sub = cdf[(cdf["ts"] >= start) & (cdf["ts"] < end)]
            diff = (pd.to_numeric(sub["pRealKw"], errors="coerce") - pd.to_numeric(sub["pRealKw_original"], errors="coerce")).abs()
            gt = bool((diff > 1e-4).any())
    gt_flags.append(gt)

pivot_anom["ground_truth"] = gt_flags

# Evaluate Voting Strategies
models = [c for c in pivot_anom.columns if c != "ground_truth"]
pivot_anom["vote_any_1"] = pivot_anom[models].any(axis=1)
pivot_anom["vote_at_least_2"] = pivot_anom[models].sum(axis=1) >= 2
pivot_anom["vote_all_3"] = pivot_anom[models].sum(axis=1) == len(models)

print("\n--- ENSEMBLE VOTING STRATEGIES ---")
for col in ["vote_any_1", "vote_at_least_2", "vote_all_3"]:
    tp = (pivot_anom[col] & pivot_anom["ground_truth"]).sum()
    fp = (pivot_anom[col] & ~pivot_anom["ground_truth"]).sum()
    fn = (~pivot_anom[col] & pivot_anom["ground_truth"]).sum()
    tn = (~pivot_anom[col] & ~pivot_anom["ground_truth"]).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"Strategy: {col:15s} | TP: {tp:5d} | FP: {fp:5d} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
