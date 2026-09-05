"""
Rank every detected anomaly by severity and flag which ones are likely
noise rather than genuine events.

Why not just sort by max_score
--------------------------------
max_score isn't comparable across meters -- every meter has its own
threshold on its own score scale (see ALL_METERS_anomalies.csv's
`threshold` column, which varies ~0.29 to ~4.07 across meters). A
max_score of 5 could be a huge outlier for a quiet meter and unremarkable
for a noisy one. So severity is built from the RATIO of each anomaly's
score to that meter's own threshold, not the raw score.

Severity score
--------------
Three ingredients, each converted to a 0-100 percentile rank across the
whole dataset (percentile ranking makes the composite robust to the
extreme outliers in this data -- a few anomalies have max_score/threshold
ratios in the millions, and raw averaging would let those dominate
everything):
  1. peak intensity   = max_score / threshold      (how extreme at its worst)
  2. sustained intensity = mean_score / threshold   (how extreme on average
                                                       over the whole period)
  3. duration_minutes  (how long it lasted)

severity_score = mean of the three percentile ranks (0-100). Both high
magnitude AND long duration push this up, per "long duration and high
value is more severity" -- an anomaly that's only extreme on one axis
scores in the middle, not the top.

The noise line
---------------
Anomalies are flagged likely_noise = True when they're BOTH weak in peak
intensity (ratio_max < 2.0 -- i.e. barely crossed their own meter's
threshold) AND short-lived (duration_minutes < 500, roughly the dataset's
own median duration). Either signal alone being strong keeps it out of
the noise bucket; both being weak is what marks it. In this dataset,
~88% of detections sit in the 1.5-3x threshold range (a tight cluster
right around the decision boundary), so this dual-weak-signal rule is
deliberately conservative about calling something noise -- it only
flags detections that are unremarkable on both axes at once.

extreme_outlier flags ratio_max > 100 -- three rows in this dataset have
ratios in the thousands-to-millions range, which is far beyond what a
"severe but real" physical anomaly usually looks like and more likely
indicates a data-quality issue (corrupted readings, a sensor dropout,
etc.) for that period rather than a typical severe fault. These are kept
at the top of the ranking (something is clearly very wrong) but flagged
separately so you sanity-check the underlying data before treating them
as a normal severe anomaly.

Usage
-----
    python rank_anomalies.py --input ALL_METERS_anomalies.csv --output ALL_METERS_anomalies_ranked.csv
"""

import argparse

import pandas as pd


def rank_anomalies(input_path: str, ratio_noise_cutoff: float = 2.0,
                    duration_noise_cutoff: float = 500.0, extreme_ratio_cutoff: float = 100.0) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df["ratio_max"] = df["max_score"] / df["threshold"]
    df["ratio_mean"] = df["mean_score"] / df["threshold"]

    # percentile rank (0-100) across the whole dataset -- robust to the
    # extreme-outlier rows since rank position doesn't care how far out they are
    df["pct_ratio_max"] = df["ratio_max"].rank(pct=True) * 100
    df["pct_ratio_mean"] = df["ratio_mean"].rank(pct=True) * 100
    df["pct_duration"] = df["duration_minutes"].rank(pct=True) * 100

    df["severity_score"] = df[["pct_ratio_max", "pct_ratio_mean", "pct_duration"]].mean(axis=1)

    df["likely_noise"] = (df["ratio_max"] < ratio_noise_cutoff) & (df["duration_minutes"] < duration_noise_cutoff)
    df["extreme_outlier"] = df["ratio_max"] > extreme_ratio_cutoff

    df = df.sort_values("severity_score", ascending=False).reset_index(drop=True)
    df.insert(0, "severity_rank", df.index + 1)

    # tidy column order: identity/timing, then severity info, then raw source columns
    front_cols = ["severity_rank", "meter_id", "start", "end", "duration_minutes",
                  "max_score", "mean_score", "threshold", "ratio_max", "ratio_mean",
                  "severity_score", "likely_noise", "extreme_outlier"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ALL_METERS_anomalies.csv")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_ranked.csv)")
    parser.add_argument("--ratio_noise_cutoff", type=float, default=2.0,
                         help="Below this max_score/threshold ratio counts as 'weak' for the noise rule")
    parser.add_argument("--duration_noise_cutoff", type=float, default=500.0,
                         help="Below this duration (minutes) counts as 'weak' for the noise rule")
    parser.add_argument("--extreme_ratio_cutoff", type=float, default=100.0,
                         help="Above this max_score/threshold ratio gets flagged extreme_outlier")
    args = parser.parse_args()

    out_path = args.output or args.input.rsplit(".csv", 1)[0] + "_ranked.csv"
    ranked = rank_anomalies(args.input, args.ratio_noise_cutoff, args.duration_noise_cutoff, args.extreme_ratio_cutoff)
    ranked.to_csv(out_path, index=False)

    n_noise = ranked["likely_noise"].sum()
    n_extreme = ranked["extreme_outlier"].sum()
    print(f"Ranked {len(ranked)} anomalies -> {out_path}")
    print(f"  {n_noise} ({n_noise/len(ranked):.0%}) flagged likely_noise")
    print(f"  {n_extreme} flagged extreme_outlier (check data quality for these)")
    print(f"\nTop 10 most severe:")
    print(ranked[["severity_rank", "meter_id", "start", "duration_minutes", "ratio_max",
                  "severity_score", "extreme_outlier"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
