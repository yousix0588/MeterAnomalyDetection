"""
Roll up ranked_ensemble_anomalies.csv (one row per meter-day) into ONE row
per meter, so you can see which meters have the most anomaly activity
overall -- not just which single meter-day ranked highest.

Sort order
----------
Primary: n_anomaly_days (how many distinct days this meter was flagged by
at least one model) -- descending, so the most frequently-flagged meters
lead.
Tiebreak: max_weighted_severity (that meter's single most severe
meter-day) -- descending, so among meters with the same anomaly count,
the more severe one comes first, per your request.

Most severe anomaly period ("agreed window")
------------------------------------------------
Each meter's peak day is chosen by vote_count FIRST (how many models
agreed that day), with weighted_severity used only as the tiebreaker when
two days for the same meter have equal vote_count. A day 3 models agree
on beats a day only 1 model flags very strongly -- consensus across
independent models is treated as stronger evidence than any single
model's intensity.

For that peak day, the reported start/end is the AGREED overlapping time
window across the specific models that voted that day -- not just one
model's own event span. Each contributing model's strongest event that
day is clipped to the day's boundaries, then intersected (latest start,
earliest end) across all of them. If the models agree on the day but
their actual events don't share a common time-of-day window (e.g. one
flagged the morning, another the evening), there's no genuine agreed
window -- agreed_start/agreed_end fall back to the full day span and
has_time_overlap is False, so that distinction is never hidden.

Usage
-----
    python rank_meters.py --input ranked_ensemble_anomalies.csv \
        --events standardized_all_models.csv --out ranked_meters.csv
"""

import argparse

import pandas as pd


def find_agreed_window(std_df: pd.DataFrame, meter_id: str, peak_day, contributing_models: str) -> dict:
    """
    For the peak day, compute the AGREED overlapping time window across the
    specific models that voted that day (not just one model's own event
    span). For each contributing model, take its strongest event overlapping
    that day, clip it to the day's boundaries, then intersect (max of
    starts, min of ends) across all contributing models.

    If the intersection is empty (same day, but the models' actual events
    don't share a common time-of-day window -- e.g. one flagged the morning,
    another the evening), there's no genuine agreed WINDOW, only agreed DAY.
    In that case agreed_start/agreed_end fall back to the full day span and
    has_time_overlap is False, so this distinction is never hidden.
    """
    if not isinstance(contributing_models, str) or not contributing_models:
        return {"agreed_start": None, "agreed_end": None, "agreed_duration_minutes": None, "has_time_overlap": None}

    tz = std_df["start"].dt.tz
    day_start = pd.Timestamp(peak_day).tz_localize(tz)
    day_end = day_start + pd.Timedelta(days=1)

    meter_events = std_df[std_df["meter_id"] == meter_id]
    starts, ends = [], []
    for model in contributing_models.split(","):
        model_events = meter_events[meter_events["model_name"] == model]
        overlapping = model_events[(model_events["start"] < day_end) & (model_events["end"] > day_start)]
        if overlapping.empty:
            continue
        best = overlapping.loc[overlapping["severity_pct"].idxmax()]
        starts.append(max(best["start"], day_start))
        ends.append(min(best["end"], day_end))

    if not starts:
        return {"agreed_start": None, "agreed_end": None, "agreed_duration_minutes": None, "has_time_overlap": None}

    agreed_start, agreed_end = max(starts), min(ends)
    has_overlap = agreed_start < agreed_end
    if not has_overlap:
        agreed_start, agreed_end = day_start, day_end  # fall back to the full day span

    return {"agreed_start": agreed_start, "agreed_end": agreed_end,
            "agreed_duration_minutes": (agreed_end - agreed_start).total_seconds() / 60,
            "has_time_overlap": has_overlap}


def rank_meters(ensemble_path: str, events_path: str) -> pd.DataFrame:
    df = pd.read_csv(ensemble_path, parse_dates=["day"])
    df["day"] = df["day"].dt.date
    std_df = pd.read_csv(events_path, parse_dates=["start", "end"])

    n_models = len([c for c in df.columns if c.endswith("_severity") and c != "weighted_severity"])
    majority_threshold = n_models // 2 + 1

    agg = df.groupby("meter_id").agg(
        n_anomaly_days=("day", "count"),
        n_majority_days=("vote_count", lambda v: (v >= majority_threshold).sum()),
        n_unanimous_days=("vote_count", lambda v: (v == n_models).sum()),
        max_vote_count=("vote_count", "max"),
        max_weighted_severity=("weighted_severity", "max"),
        mean_weighted_severity=("weighted_severity", "mean"),
    ).reset_index()

    # peak day per meter -- prioritize vote_count first (most models agreeing),
    # weighted_severity only as the tiebreaker when vote_count is equal
    peak_days = (df.sort_values(["meter_id", "vote_count", "weighted_severity"], ascending=[True, False, False])
                   .groupby("meter_id", as_index=False).first())
    peak_days = peak_days[["meter_id", "day", "vote_count", "weighted_severity", "contributing_models"]]
    peak_days = peak_days.rename(columns={"day": "most_severe_day", "vote_count": "most_severe_day_vote_count",
                                           "weighted_severity": "most_severe_day_weighted_severity",
                                           "contributing_models": "most_severe_day_models"})

    agg = agg.merge(peak_days, on="meter_id", how="left")

    # trace back to the actual AGREED overlapping window across contributing models
    detail_rows = [find_agreed_window(std_df, r.meter_id, r.most_severe_day, r.most_severe_day_models)
                    for r in agg.itertuples()]
    agg = pd.concat([agg, pd.DataFrame(detail_rows)], axis=1)

    agg = agg.sort_values(
        ["n_anomaly_days", "max_weighted_severity"], ascending=[False, False]
    ).reset_index(drop=True)
    agg.insert(0, "meter_rank", agg.index + 1)

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ranked_ensemble_anomalies.csv (from ensemble_vote.py)")
    parser.add_argument("--events", required=True, help="Path to standardized_all_models.csv (from formalize_anomalies.py)")
    parser.add_argument("--lstm_summary", default=None,
                         help="Path to the *_lstm_meter_summary.csv from formalize_anomalies.py (optional). "
                              "Merged in as extra context columns only -- LSTM has no day-level timestamps, "
                              "so it does NOT contribute to n_anomaly_days/vote_count/max_weighted_severity, "
                              "which all come purely from the day-level vote among the other models.")
    parser.add_argument("--out", default="ranked_meters.csv")
    args = parser.parse_args()

    ranked = rank_meters(args.input, args.events)

    if args.lstm_summary:
        lstm_df = pd.read_csv(args.lstm_summary)
        ranked = ranked.merge(lstm_df, on="meter_id", how="left")
        # meters LSTM never scored at all (not in its 503) get NaN here, not 0 --
        # 0 would wrongly claim "LSTM checked and found nothing"

    ranked.to_csv(args.out, index=False)

    print(f"Ranked {len(ranked)} meters -> {args.out}")
    print(f"\nTop 15 meters by anomaly-day count (ties broken by severity):")
    cols = ["meter_rank", "meter_id", "n_anomaly_days", "most_severe_day_vote_count",
            "max_weighted_severity", "agreed_start", "agreed_end", "has_time_overlap", "most_severe_day_models"]
    if args.lstm_summary:
        cols += ["lstm_anomaly_rate", "lstm_severity_pct"]
    print(ranked[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

