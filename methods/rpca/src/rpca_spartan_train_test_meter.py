#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from rpca_spartan_train_test import (
    FEATURES,
    POINTS_PER_DAY,
    load_daily_tensor,
    robust_pca,
    choose_subspace,
    reconstruct,
)


def load_meter_tensor(meter_id, manifest_path, meter_dir):
    """
    Load all channels belonging to one physical meter.

    Each channel is first converted to:
        n_days x 288 x 6

    Only dates that are complete for ALL included channels are retained.

    Final meter tensor:
        n_days x 288 x (6 * n_channels)
    """
    manifest = pd.read_csv(manifest_path)
    channel_col = manifest.columns[0]

    channels = sorted(
        ch
        for ch in manifest[channel_col].astype(str)
        if ch.startswith(meter_id + "_")
    )

    if not channels:
        raise ValueError(
            f"{meter_id}: no channels found in {manifest_path}"
        )

    loaded = []
    common_dates = None

    for ch in channels:
        csv_path = Path(meter_dir) / f"{ch}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        dates, raw = load_daily_tensor(csv_path)

        loaded.append((ch, dates, raw))

        this_dates = set(dates)

        if common_dates is None:
            common_dates = this_dates
        else:
            common_dates &= this_dates

    common_dates = pd.DatetimeIndex(
        sorted(common_dates)
    )

    if len(common_dates) == 0:
        raise ValueError(
            f"{meter_id}: no common complete days across channels."
        )

    channel_arrays = []

    for ch, dates, raw in loaded:
        lookup = {
            pd.Timestamp(d): i
            for i, d in enumerate(dates)
        }

        idx = [
            lookup[pd.Timestamp(d)]
            for d in common_dates
        ]

        channel_arrays.append(raw[idx])

    # n_days x 288 x (6 * n_channels)
    meter_raw = np.concatenate(
        channel_arrays,
        axis=2,
    )

    return common_dates, meter_raw, channels


def rms_meter_scores(residual_flat, feature_dim):
    """
    residual_flat:
        n_days x (288 * feature_dim)

    Returns:
        point score: n_days x 288
        daily score: n_days

    At each 5-minute point, residuals are combined across
    all channels and all six electrical features using RMS.
    """
    r3 = residual_flat.reshape(
        residual_flat.shape[0],
        POINTS_PER_DAY,
        feature_dim,
    )

    point = np.sqrt(
        np.mean(r3 ** 2, axis=2)
    )

    daily = np.sqrt(
        np.mean(r3 ** 2, axis=(1, 2))
    )

    return point, daily


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--meter-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--meter-dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument(
        "--test-start",
        default="2026-07-01",
    )
    ap.add_argument(
        "--test-end",
        default="2026-08-01",
    )
    ap.add_argument(
        "--power-cutoff",
        type=float,
        default=0.5,
    )
    ap.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.995,
    )
    ap.add_argument(
        "--subspace-energy",
        type=float,
        default=0.95,
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dates, raw, channels = load_meter_tensor(
        args.meter_id,
        args.manifest,
        args.meter_dir,
    )

    n_channels = len(channels)
    feature_dim = n_channels * len(FEATURES)

    train_mask = (
        dates < pd.Timestamp(args.test_start)
    )

    test_mask = (
        (dates >= pd.Timestamp(args.test_start))
        & (dates < pd.Timestamp(args.test_end))
    )

    if train_mask.sum() < 30:
        raise ValueError(
            f"{args.meter_id}: only "
            f"{train_mask.sum()} complete training days."
        )

    if test_mask.sum() == 0:
        raise ValueError(
            f"{args.meter_id}: no complete test days."
        )

    train_raw = raw[train_mask]
    test_raw = raw[test_mask]

    test_dates = dates[test_mask]

    # --------------------------------------------------
    # Meter-level activity check
    #
    # For each channel, calculate training p95 |pRealKw|.
    # If at least one channel is active enough, retain meter.
    # --------------------------------------------------

    p_idx = FEATURES.index("pRealKw")

    power_indices = [
        i * len(FEATURES) + p_idx
        for i in range(n_channels)
    ]

    channel_p95 = [
        float(
            np.quantile(
                np.abs(
                    train_raw[:, :, idx]
                ),
                0.95,
            )
        )
        for idx in power_indices
    ]

    meter_activity_p95 = max(channel_p95)

    if meter_activity_p95 < args.power_cutoff:
        pd.DataFrame([{
            "meter_id": args.meter_id,
            "status": "skipped_low_activity",
            "n_channels": n_channels,
            "channels": ";".join(channels),
            "p95_abs_pRealKw_train": meter_activity_p95,
        }]).to_csv(
            outdir /
            f"{args.meter_id}_summary.csv",
            index=False,
        )

        return

    # --------------------------------------------------
    # Robust scaling — training data only
    # --------------------------------------------------

    scaler = RobustScaler()

    scaler.fit(
        train_raw.reshape(
            -1,
            feature_dim,
        )
    )

    train_scaled = scaler.transform(
        train_raw.reshape(
            -1,
            feature_dim,
        )
    ).reshape(train_raw.shape)

    test_scaled = scaler.transform(
        test_raw.reshape(
            -1,
            feature_dim,
        )
    ).reshape(test_raw.shape)

    # --------------------------------------------------
    # Flatten each day
    #
    # channel-level:
    # 288 x 6
    #
    # meter-level:
    # 288 x (6 * number of channels)
    # --------------------------------------------------

    x_train = train_scaled.reshape(
        train_scaled.shape[0],
        -1,
    )

    x_test = test_scaled.reshape(
        test_scaled.shape[0],
        -1,
    )

    # Robust centre estimated from historical days.
    center = np.median(
        x_train,
        axis=0,
    )

    x_train_c = x_train - center
    x_test_c = x_test - center

    # --------------------------------------------------
    # Robust PCA on historical meter profiles
    # --------------------------------------------------

    l_train, s_train, info = robust_pca(
        x_train_c
    )

    basis, subspace_rank = choose_subspace(
        l_train,
        energy=args.subspace_energy,
    )

    train_recon_c = reconstruct(
        x_train_c,
        basis,
    )

    test_recon_c = reconstruct(
        x_test_c,
        basis,
    )

    train_resid = (
        x_train_c - train_recon_c
    )

    test_resid = (
        x_test_c - test_recon_c
    )

    train_point, train_daily = (
        rms_meter_scores(
            train_resid,
            feature_dim,
        )
    )

    test_point, test_daily = (
        rms_meter_scores(
            test_resid,
            feature_dim,
        )
    )

    # Threshold is calculated independently
    # for each physical meter using TRAINING scores.
    point_threshold = float(
        np.quantile(
            train_point.ravel(),
            args.threshold_quantile,
        )
    )

    daily_threshold = float(
        np.quantile(
            train_daily,
            args.threshold_quantile,
        )
    )

    # --------------------------------------------------
    # Daily output
    # --------------------------------------------------

    daily_df = pd.DataFrame({
        "date": test_dates.date,
        "daily_anomaly_score": test_daily,
        "is_daily_anomaly":
            test_daily > daily_threshold,
    })

    daily_df.to_csv(
        outdir /
        f"{args.meter_id}_daily_scores.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5-minute point output
    # --------------------------------------------------

    timestamps = [
        day + pd.Timedelta(
            minutes=5 * i
        )
        for day in test_dates
        for i in range(POINTS_PER_DAY)
    ]

    point_df = pd.DataFrame({
        "timestamp": timestamps,
        "anomaly_score":
            test_point.reshape(-1),
    })

    point_df["is_anomaly"] = (
        point_df["anomaly_score"]
        > point_threshold
    )

    point_df.to_csv(
        outdir /
        f"{args.meter_id}_point_scores.csv",
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = pd.DataFrame([{
        "meter_id": args.meter_id,
        "status": "ok",
        "n_channels": n_channels,
        "channels": ";".join(channels),
        "n_train_days":
            int(train_mask.sum()),
        "n_test_days":
            int(test_mask.sum()),
        "p95_abs_pRealKw_train":
            meter_activity_p95,
        "rpca_rank_L":
            info["rank_L"],
        "subspace_rank":
            subspace_rank,
        "rpca_iterations":
            info["iterations"],
        "point_threshold":
            point_threshold,
        "daily_threshold":
            daily_threshold,
        "n_anomalous_points":
            int(
                np.sum(
                    test_point
                    > point_threshold
                )
            ),
        "n_anomalous_days":
            int(
                np.sum(
                    test_daily
                    > daily_threshold
                )
            ),
        "max_daily_score":
            float(
                np.max(test_daily)
            ),
        "max_daily_score_date":
            str(
                test_dates[
                    int(
                        np.argmax(
                            test_daily
                        )
                    )
                ].date()
            ),
    }])

    summary.to_csv(
        outdir /
        f"{args.meter_id}_summary.csv",
        index=False,
    )

    print(
        f"{args.meter_id}: "
        f"{n_channels} channels, "
        f"{train_mask.sum()} train days, "
        f"{test_mask.sum()} test days, "
        f"threshold={point_threshold:.6f}"
    )


if __name__ == "__main__":
    main()