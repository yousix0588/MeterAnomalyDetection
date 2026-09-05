"""
Ensemble majority vote across the standardized outputs of multiple models
(from formalize_anomalies.py).

Why meter-DAY granularity
---------------------------
Each model defines "an event" differently (LOF: 5-minute points, VAE/LSTM:
multi-hour windows, ALL_EVENTS: variable-duration change-point events).
There's no clean way to say two differently-shaped intervals are "the same
event" in general, so the vote is taken at meter-day granularity: an event
"votes" for every calendar day it overlaps. This is coarser than any
individual model's native resolution, but it's the common denominator that
makes a fair cross-model vote possible.

Vote logic
----------
For each (meter_id, day):
    - vote_count      = how many distinct models flagged this meter-day
    - weighted_severity = mean of each contributing model's severity_pct
                          (0-100 percentile rank WITHIN that model's own
                          detections -- see formalize_anomalies.py)
    - <MODEL>_severity  = that model's severity_pct for this meter-day, or
                          blank if it didn't flag it (wide columns, one per
                          model actually present in the input)
    - contributing_models = comma-separated list of which models voted

Final ranking is by vote_count first (more independent models agreeing is
the strongest signal), then weighted_severity as the tiebreaker within the
same vote count.

Usage
-----
    python ensemble_vote.py --input standardized_all_models.csv --out ranked_ensemble_anomalies.csv

    # require at least 3 of however many models are present to count as
    # "majority" in the summary printout (does not affect the output rows,
    # every meter-day with >=1 vote is still written out and ranked)
    python ensemble_vote.py --input standardized_all_models.csv --out ranked_ensemble_anomalies.csv --majority_threshold 3
"""

import argparse

import pandas as pd


def expand_to_days(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_name, meter_id, day, severity_pct) that an event overlaps."""
    rows = []
    for r in df.itertuples(index=False):
        start_day = r.start.normalize()
        end_day = r.end.normalize()
        for day in pd.date_range(start_day, end_day, freq="D"):
            rows.append({"model_name": r.model_name, "meter_id": r.meter_id,
                         "day": day.date(), "severity_pct": r.severity_pct})
    return pd.DataFrame(rows)


def ensemble_vote(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["start", "end"])
    daily = expand_to_days(df)

    # if a model flagged the same meter-day more than once (e.g. two separate
    # events both touching that day), keep its strongest severity for that day
    daily = daily.groupby(["model_name", "meter_id", "day"], as_index=False)["severity_pct"].max()

    # wide pivot: one column per model's severity for that meter-day
    wide = daily.pivot_table(index=["meter_id", "day"], columns="model_name",
                              values="severity_pct", aggfunc="max")
    model_cols = list(wide.columns)
    wide.columns = [f"{c}_severity" for c in wide.columns]
    wide = wide.reset_index()

    severity_cols = [f"{c}_severity" for c in model_cols]
    wide["vote_count"] = wide[severity_cols].notna().sum(axis=1)
    wide["weighted_severity"] = wide[severity_cols].mean(axis=1, skipna=True)
    wide["contributing_models"] = wide[severity_cols].apply(
        lambda row: ",".join(m for m, c in zip(model_cols, severity_cols) if pd.notna(row[c])), axis=1)

    wide = wide.sort_values(["vote_count", "weighted_severity"], ascending=[False, False]).reset_index(drop=True)
    wide.insert(0, "ensemble_rank", wide.index + 1)

    front_cols = ["ensemble_rank", "meter_id", "day", "vote_count", "weighted_severity", "contributing_models"]
    wide = wide[front_cols + severity_cols]
    return wide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to standardized_all_models.csv (from formalize_anomalies.py)")
    parser.add_argument("--out", default="ranked_ensemble_anomalies.csv")
    parser.add_argument("--majority_threshold", type=int, default=None,
                         help="For the summary printout only: how many votes counts as 'majority' "
                              "(default: more than half of the models present in the input)")
    args = parser.parse_args()

    ranked = ensemble_vote(args.input)
    ranked.to_csv(args.out, index=False)

    n_models = len([c for c in ranked.columns if c.endswith("_severity") and c != "weighted_severity"])
    majority_threshold = args.majority_threshold or (n_models // 2 + 1)

    print(f"Ranked {len(ranked)} meter-day anomaly candidates across {n_models} model(s) -> {args.out}")
    print(f"\nVote count distribution:")
    print(ranked["vote_count"].value_counts().sort_index(ascending=False).to_string())
    n_majority = (ranked["vote_count"] >= majority_threshold).sum()
    print(f"\n{n_majority} meter-days reach majority ({majority_threshold}+ of {n_models} models agreeing)")
    print(f"\nTop 10 by ensemble rank:")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
