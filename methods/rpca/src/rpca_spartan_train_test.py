#!/usr/bin/env python3
"""
RPCA-based robust subspace anomaly detection for Spartan.

Designed for:
  /data/projects/punim1257/Group14/meter_csvs/<meter_channel>.csv

Protocol:
- One meter-channel per Slurm task.
- Six electrical variables.
- Historical baseline: dates before --test-start.
- Test period: [--test-start, --test-end).
- Fit RobustScaler on training data only.
- Fit Robust PCA (Principal Component Pursuit) on historical daily profiles.
- Extract a robust low-rank subspace from the RPCA low-rank component.
- Project July daily profiles into the learned normal subspace.
- Reconstruction residual = anomaly signal.
- Thresholds are the 99th percentile of TRAINING residual scores.
- Outputs a July pRealKw + anomaly-score chart and score CSV.

This is intentionally a train/test version of the earlier batch RPCA prototype:
it avoids using future test data to define "normal".
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "pRealKw",
    "pReactiveKw",
    "vRMSMin",
    "vRMSMax",
    "iRMSMin",
    "iRMSMax",
]
POINTS_PER_DAY = 288


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svt(x, tau):
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    shrunk = np.maximum(s - tau, 0.0)
    rank = int(np.sum(shrunk > 0))
    if rank == 0:
        return np.zeros_like(x), 0
    return (u[:, :rank] * shrunk[:rank]) @ vt[:rank, :], rank


def robust_pca(x, lam=None, tol=1e-6, max_iter=250):
    """Principal Component Pursuit: X = L + S."""
    m, n = x.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_two = np.linalg.norm(x, 2)
    norm_inf = np.max(np.abs(x)) / lam
    y = x / (max(norm_two, norm_inf) + 1e-12)

    norm_fro = np.linalg.norm(x, "fro")
    mu = 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    l = np.zeros_like(x)
    s = np.zeros_like(x)

    for it in range(max_iter):
        l, rank = svt(x - s + y / mu, 1.0 / mu)
        s = soft_threshold(x - l + y / mu, lam / mu)
        residual = x - l - s
        relerr = np.linalg.norm(residual, "fro") / (norm_fro + 1e-12)
        y += mu * residual
        mu = min(mu * rho, mu_bar)
        if relerr < tol:
            break

    return l, s, {
        "iterations": it + 1,
        "rank_L": rank,
        "relative_error": float(relerr),
    }


def load_daily_tensor(csv_path):
    """
    Convert a merged meter-channel CSV into daily 288x6 profiles.

    Timestamps are converted to Australia/Melbourne.
    The DST fall-back day may contain duplicate local clock times; duplicate
    5-minute wall-clock slots are averaged so that a day returns to 288 slots.
    Days with missing slots after this step are skipped.
    """
    usecols = ["timestamp", *FEATURES]
    df = pd.read_csv(csv_path, usecols=usecols)

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    df["timestamp"] = ts[ts.notna()].dt.tz_convert("Australia/Melbourne")

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=FEATURES)
    df["date"] = df["timestamp"].dt.date
    df["slot"] = (
        df["timestamp"].dt.hour * 12
        + df["timestamp"].dt.minute // 5
    ).astype(int)

    # Mean duplicate local slots on DST fall-back day.
    grouped = (
        df.groupby(["date", "slot"], as_index=False)[FEATURES]
        .mean()
    )

    day_arrays = []
    day_dates = []

    for date, g in grouped.groupby("date"):
        g = g.set_index("slot").reindex(range(POINTS_PER_DAY))
        if g[FEATURES].isna().any().any():
            continue
        day_arrays.append(g[FEATURES].to_numpy(float))
        day_dates.append(pd.Timestamp(date))

    if not day_arrays:
        raise ValueError("No complete 288-point days were found.")

    return pd.DatetimeIndex(day_dates), np.stack(day_arrays)


def choose_subspace(l_train, energy=0.95):
    """
    Obtain the row-space basis of the robust low-rank training matrix.
    """
    _, singular_values, vt = np.linalg.svd(l_train, full_matrices=False)
    power = singular_values ** 2
    total = power.sum()

    if total <= 0:
        raise ValueError("RPCA low-rank component has zero energy.")

    cumulative = np.cumsum(power) / total
    rank = int(np.searchsorted(cumulative, energy) + 1)
    rank = min(rank, vt.shape[0])

    return vt[:rank], rank


def reconstruct(x_centered, basis):
    return (x_centered @ basis.T) @ basis


def rms_point_score(residual_flat):
    """
    residual_flat: n_days x (288*6)
    Returns:
      point_score: n_days x 288
      daily_score: n_days
    """
    r3 = residual_flat.reshape(
        residual_flat.shape[0],
        POINTS_PER_DAY,
        len(FEATURES),
    )
    point = np.sqrt(np.mean(r3 ** 2, axis=2))
    daily = np.sqrt(np.mean(r3 ** 2, axis=(1, 2)))
    return point, daily


def make_timeline_plot(
    out_png,
    test_dates,
    test_raw,
    test_point_score,
    point_threshold,
    title,
):
    p_idx = FEATURES.index("pRealKw")

    timestamps = pd.DatetimeIndex([
        day + pd.Timedelta(minutes=5 * i)
        for day in test_dates
        for i in range(POINTS_PER_DAY)
    ])

    power = test_raw[:, :, p_idx].reshape(-1)
    score = test_point_score.reshape(-1)
    anomaly = score > point_threshold

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )

    axes[0].plot(timestamps, power, linewidth=0.9, label="pRealKw")
    if anomaly.any():
        axes[0].scatter(
            timestamps[anomaly],
            power[anomaly],
            marker="x",
            s=12,
            label="Candidate anomaly",
        )
    axes[0].set_ylabel("pRealKw (kW)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(
        timestamps,
        score,
        linewidth=0.9,
        label="RPCA residual anomaly score",
    )
    axes[1].axhline(
        point_threshold,
        linestyle="--",
        linewidth=1.2,
        label="Training 99th percentile threshold",
    )
    axes[1].set_ylabel("Anomaly score")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=0.2)

    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=35)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--test-start", default="2026-07-01")
    ap.add_argument("--test-end", default="2026-08-01")
    ap.add_argument("--power-cutoff", type=float, default=0.5)
    ap.add_argument("--threshold-quantile", type=float, default=0.99)
    ap.add_argument("--subspace-energy", type=float, default=0.95)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meter_channel = csv_path.stem

    dates, raw = load_daily_tensor(csv_path)

    train_mask = dates < pd.Timestamp(args.test_start)
    test_mask = (
        (dates >= pd.Timestamp(args.test_start))
        & (dates < pd.Timestamp(args.test_end))
    )

    if train_mask.sum() < 30:
        raise ValueError(
            f"{meter_channel}: only {train_mask.sum()} complete training days."
        )
    if test_mask.sum() == 0:
        raise ValueError(
            f"{meter_channel}: no complete test days in requested period."
        )

    train_raw = raw[train_mask]
    test_raw = raw[test_mask]
    train_dates = dates[train_mask]
    test_dates = dates[test_mask]

    p_idx = FEATURES.index("pRealKw")
    p95 = float(
        np.quantile(np.abs(train_raw[:, :, p_idx]), 0.95)
    )
    if p95 < args.power_cutoff:
        pd.DataFrame([{
            "meter_channel": meter_channel,
            "status": "skipped_low_activity",
            "p95_abs_pRealKw_train": p95,
        }]).to_csv(
            outdir / f"{meter_channel}_summary.csv",
            index=False,
        )
        return

    # Robust scaling is fitted ONLY on training data.
    scaler = RobustScaler()
    scaler.fit(train_raw.reshape(-1, len(FEATURES)))

    train_scaled = scaler.transform(
        train_raw.reshape(-1, len(FEATURES))
    ).reshape(train_raw.shape)

    test_scaled = scaler.transform(
        test_raw.reshape(-1, len(FEATURES))
    ).reshape(test_raw.shape)

    x_train = train_scaled.reshape(train_scaled.shape[0], -1)
    x_test = test_scaled.reshape(test_scaled.shape[0], -1)

    # Robust center across historical daily profiles.
    center = np.median(x_train, axis=0)
    x_train_c = x_train - center
    x_test_c = x_test - center

    # RPCA learns robust normal low-rank structure from history.
    l_train, s_train, info = robust_pca(x_train_c)

    basis, subspace_rank = choose_subspace(
        l_train,
        energy=args.subspace_energy,
    )

    # Reconstruction residuals in the learned robust subspace.
    train_recon_c = reconstruct(x_train_c, basis)
    test_recon_c = reconstruct(x_test_c, basis)

    train_resid = x_train_c - train_recon_c
    test_resid = x_test_c - test_recon_c

    train_point, train_daily = rms_point_score(train_resid)
    test_point, test_daily = rms_point_score(test_resid)

    point_threshold = float(
        np.quantile(train_point.ravel(), args.threshold_quantile)
    )
    daily_threshold = float(
        np.quantile(train_daily, args.threshold_quantile)
    )

    # Daily scores.
    daily_df = pd.DataFrame({
        "date": test_dates.date,
        "daily_anomaly_score": test_daily,
        "is_daily_anomaly": test_daily > daily_threshold,
    })
    daily_df.to_csv(
        outdir / f"{meter_channel}_daily_scores.csv",
        index=False,
    )

    # 5-minute test scores.
    timestamps = [
        day + pd.Timedelta(minutes=5 * i)
        for day in test_dates
        for i in range(POINTS_PER_DAY)
    ]
    point_df = pd.DataFrame({
        "timestamp": timestamps,
        "anomaly_score": test_point.reshape(-1),
    })
    point_df["is_anomaly"] = (
        point_df["anomaly_score"] > point_threshold
    )
    point_df.to_csv(
        outdir / f"{meter_channel}_point_scores.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "meter_channel": meter_channel,
        "status": "ok",
        "n_train_days": int(train_mask.sum()),
        "n_test_days": int(test_mask.sum()),
        "p95_abs_pRealKw_train": p95,
        "rpca_rank_L": info["rank_L"],
        "subspace_rank": subspace_rank,
        "rpca_iterations": info["iterations"],
        "point_threshold": point_threshold,
        "daily_threshold": daily_threshold,
        "n_anomalous_points": int(
            np.sum(test_point > point_threshold)
        ),
        "n_anomalous_days": int(
            np.sum(test_daily > daily_threshold)
        ),
        "max_daily_score": float(np.max(test_daily)),
        "max_daily_score_date": str(
            test_dates[int(np.argmax(test_daily))].date()
        ),
    }])
    summary.to_csv(
        outdir / f"{meter_channel}_summary.csv",
        index=False,
    )

    make_timeline_plot(
        outdir / f"{meter_channel}_rpca_july.png",
        test_dates,
        test_raw,
        test_point,
        point_threshold,
        title=(
            f"RPCA-based anomaly detection — {meter_channel}\n"
            f"Historical baseline before {args.test_start}; "
            f"test period {args.test_start} to {args.test_end}"
        ),
    )


if __name__ == "__main__":
    main()
