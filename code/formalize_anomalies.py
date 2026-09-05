"""
Standardize the four models' anomaly CSVs into one common schema, so the
downstream ensemble vote doesn't need to know each model's quirks.

Common output schema (one row per detected event, long format):
    model_name, meter_id, start, end, duration_minutes,
    peak_measure, severity_pct

peak_measure: each model's own "how extreme was this event" number, in
whatever units that model natively uses (VAE/LSTM: score/threshold ratio;
LOF: raw LOF score, no per-meter threshold available; ALL_EVENTS:
magnitude_percentile, already 0-100).

severity_pct: peak_measure converted to a 0-100 PERCENTILE RANK within
that model's own set of detections. This is the key normalization step --
raw peak_measure values are NOT comparable across models (different
scales, different units), but "how extreme is this relative to everything
else this same model flagged" is comparable, and is what the ensemble
vote actually uses as each model's confidence/weight.

Models
------
- LOF   : july_lof_anomaly_events.csv
- VAE   : ALL_METERS_anomalies.csv
- EVENTS: ALL_EVENTS.csv (filtered to the July test window; device_id +
          channel_idx combined into meter_id)
- RPCA  : RPCA_anomalies_final.csv (same shape as VAE; timestamps here are
          tz-naive in the source file, so they're explicitly localized to
          Australia/Melbourne to stay comparable with the other three)
- LSTM  : ltsm_anomaly.csv, event-level with real window_start/window_end,
          anomaly_score, and threshold (UTC timestamps, converted to
          Australia/Melbourne). IMPORTANT: this file only covers 11
          series_id values -- far fewer meters than the other four models.
          A meter absent from this file was never evaluated by LSTM, NOT
          confirmed normal by it -- don't read "no LSTM vote" as "LSTM
          agrees this meter is fine."

Usage
-----
    python formalize_anomalies.py \
        --lof july_lof_anomaly_events.csv \
        --vae ALL_METERS_anomalies.csv \
        --events ALL_EVENTS.csv \
        --rpca RPCA_anomalies_final.csv \
        --lstm ltsm_anomaly.csv \
        --test_start 2026-07-01 --test_end 2026-08-01 \
        --out standardized_all_models.csv

    # any subset of --lof/--vae/--events/--rpca/--lstm can be omitted; the
    # script just standardizes whichever inputs you provide
"""

import argparse

import pandas as pd


def _add_severity_pct(df: pd.DataFrame) -> pd.DataFrame:
    """percentile rank of peak_measure within this model's own detections"""
    df["severity_pct"] = df["peak_measure"].rank(pct=True) * 100
    return df


def standardize_lof(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start", "end"])
    out = pd.DataFrame({
        "model_name": "LOF",
        "meter_id": df["meter_id"],
        "start": df["start"],
        "end": df["end"],
        "duration_minutes": df["duration_minutes"],
        "peak_measure": df["max_lof_score"],   # no per-meter threshold available for LOF
    })
    return _add_severity_pct(out)


def standardize_vae(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start", "end"])
    out = pd.DataFrame({
        "model_name": "VAE",
        "meter_id": df["meter_id"],
        "start": df["start"],
        "end": df["end"],
        "duration_minutes": df["duration_minutes"],
        "peak_measure": df["max_score"] / df["threshold"],   # ratio, comparable across this model's own meters
    })
    return _add_severity_pct(out)


def standardize_rpca(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # timestamps here are naive (no UTC offset), unlike the other three models --
    # localize explicitly to Australia/Melbourne so this joins/compares cleanly
    # against the tz-aware start/end from LOF, VAE, and ALL_EVENTS
    df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.tz_localize("Australia/Melbourne")
    df["end"] = pd.to_datetime(df["end"], errors="coerce").dt.tz_localize("Australia/Melbourne")
    df = df.dropna(subset=["start", "end"])
    out = pd.DataFrame({
        "model_name": "RPCA",
        "meter_id": df["meter_id"],
        "start": df["start"],
        "end": df["end"],
        "duration_minutes": df["duration_minutes"],
        "peak_measure": df["max_score"] / df["threshold"],   # same shape/ratio as VAE
    })
    return _add_severity_pct(out)


def standardize_events(path: str, test_start: str = None, test_end: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # parse to UTC first (unambiguous regardless of mixed +10:00/+11:00 offsets
    # across the AEST/AEDT boundary), then convert back to local time
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne")
    df = df.dropna(subset=["start_time", "end_time"])
    df["meter_id"] = df["device_id"].astype(str) + "_" + df["channel_idx"].astype(str)
    df["duration_minutes"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60

    # ALL_EVENTS spans the full history, not just the test window -- filter to
    # match the other models' scope (skip filtering if bounds aren't given)
    if test_start is not None:
        df = df[df["start_time"] >= pd.Timestamp(test_start).tz_localize(df["start_time"].dt.tz)]
    if test_end is not None:
        df = df[df["start_time"] < pd.Timestamp(test_end).tz_localize(df["start_time"].dt.tz)]

    out = pd.DataFrame({
        "model_name": "ALL_EVENTS",
        "meter_id": df["meter_id"],
        "start": df["start_time"],
        "end": df["end_time"],
        "duration_minutes": df["duration_minutes"],
        "peak_measure": df["magnitude_percentile"],   # already 0-100; ranked again below for consistency
    })
    return _add_severity_pct(out)


def standardize_lstm(path: str, test_start: str = None, test_end: str = None) -> pd.DataFrame:
    """
    Event-level LSTM output (ltsm_anomaly.csv): series_id, window_start,
    window_end, anomaly_score, threshold, is_anomaly. Same shape/logic as
    VAE (ratio of score to threshold), so it slots in as a full day-level
    voter alongside LOF/VAE/ALL_EVENTS/RPCA.

    Two things handled explicitly:
    - Only is_anomaly == True rows are kept (defensive -- the file provided
      is already 100% True, but this guards against a future export that
      includes non-anomalous windows too).
    - Timestamps arrive in UTC; converted to Australia/Melbourne so day
      boundaries align with the other models' local-calendar-day votes.
    """
    df = pd.read_csv(path)
    df = df[df["is_anomaly"] == True]  # noqa: E712 -- explicit filter, defensive
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne")
    df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce", utc=True).dt.tz_convert("Australia/Melbourne")
    df = df.dropna(subset=["window_start", "window_end"])

    if test_start is not None:
        df = df[df["window_start"] >= pd.Timestamp(test_start).tz_localize(df["window_start"].dt.tz)]
    if test_end is not None:
        df = df[df["window_start"] < pd.Timestamp(test_end).tz_localize(df["window_start"].dt.tz)]

    out = pd.DataFrame({
        "model_name": "LSTM",
        "meter_id": df["series_id"],
        "start": df["window_start"],
        "end": df["window_end"],
        "duration_minutes": (df["window_end"] - df["window_start"]).dt.total_seconds() / 60,
        "peak_measure": df["anomaly_score"] / df["threshold"],
    })
    return _add_severity_pct(out)


def standardize_lstm_summary(path: str) -> pd.DataFrame:
    """
    The LSTM output actually available today (ltsm_summary.csv) is a
    per-METER summary -- series_id, windows, anomalies, max_score,
    mean_score -- with no start/end timestamps at all, unlike the other
    four models. It CANNOT be day-level voted the same way: there's no
    information about WHICH days within the test period were anomalous,
    only that some fraction of that meter's windows were.

    Rather than fabricate day-level events (e.g. spanning the whole test
    period, which would falsely inflate day-by-day agreement with the
    other models on every single day), this returns a separate meter-level
    table: meter_id, lstm_anomaly_rate, lstm_max_score, lstm_mean_score,
    lstm_severity_pct. This is merged into the per-METER ranking
    (rank_meters.py) as an additional corroboration column, not into the
    per-meter-DAY ensemble vote (ensemble_vote.py) -- the two files
    operate at genuinely different granularities and shouldn't be
    conflated.
    """
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "meter_id": df["series_id"],
        "lstm_n_windows": df["windows"],
        "lstm_n_anomalies": df["anomalies"],
        "lstm_anomaly_rate": df["anomalies"] / df["windows"],
        "lstm_max_score": df["max_score"],
        "lstm_mean_score": df["mean_score"],
    })
    # percentile rank within LSTM's own meters, same normalization principle
    # used for severity_pct elsewhere -- comparable across LSTM's own output,
    # not directly comparable in scale to the other models' severity_pct
    out["lstm_severity_pct"] = out["lstm_anomaly_rate"].rank(pct=True) * 100
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lof", default=None, help="Path to july_lof_anomaly_events.csv")
    parser.add_argument("--vae", default=None, help="Path to ALL_METERS_anomalies.csv")
    parser.add_argument("--events", default=None, help="Path to ALL_EVENTS.csv")
    parser.add_argument("--rpca", default=None, help="Path to RPCA_anomalies_final.csv")
    parser.add_argument("--lstm", default=None, help="Path to a per-EVENT LSTM csv, once available (see standardize_lstm)")
    parser.add_argument("--lstm_summary", default=None, help="Path to ltsm_summary.csv (per-meter summary, no timestamps)")
    parser.add_argument("--test_start", default="2026-07-01", help="Filter ALL_EVENTS to on/after this date")
    parser.add_argument("--test_end", default="2026-08-01", help="Filter ALL_EVENTS to before this date")
    parser.add_argument("--out", default="standardized_all_models.csv")
    args = parser.parse_args()

    frames = []
    if args.lof:
        frames.append(standardize_lof(args.lof))
        print(f"LOF: {len(frames[-1])} events standardized")
    if args.vae:
        frames.append(standardize_vae(args.vae))
        print(f"VAE: {len(frames[-1])} events standardized")
    if args.events:
        frames.append(standardize_events(args.events, args.test_start, args.test_end))
        print(f"ALL_EVENTS (filtered to test window): {len(frames[-1])} events standardized")
    if args.rpca:
        frames.append(standardize_rpca(args.rpca))
        print(f"RPCA: {len(frames[-1])} events standardized")
    if args.lstm:
        frames.append(standardize_lstm(args.lstm, args.test_start, args.test_end))
        print(f"LSTM (event-level): {len(frames[-1])} events standardized")

    lstm_summary_df = None
    if args.lstm_summary:
        lstm_summary_df = standardize_lstm_summary(args.lstm_summary)
        summary_out = args.out.rsplit(".csv", 1)[0] + "_lstm_meter_summary.csv"
        lstm_summary_df.to_csv(summary_out, index=False)
        print(f"LSTM (meter-level summary, NOT part of the day-level vote): "
              f"{len(lstm_summary_df)} meters -> {summary_out}")

    if not frames:
        raise SystemExit("No input files provided -- pass at least one of --lof/--vae/--events/--lstm")

    combined = pd.concat(frames, ignore_index=True).sort_values(["meter_id", "start"])
    combined.to_csv(args.out, index=False)
    print(f"\nCombined {len(combined)} events across {combined['model_name'].nunique()} model(s) -> {args.out}")
    print(combined["model_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
