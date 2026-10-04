#!/usr/bin/env python3
"""Draw aligned, print-friendly figures for every manual-review case."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    ("lstm_autoencoder", "LSTM AE", "#4C78A8"),
    ("lstm_vae", "LSTM-VAE", "#F58518"),
    ("matrix_profile", "MP (30m repeat)", "#54A24B"),
    ("rpca", "RPCA", "#B279A2"),
)
CATEGORY_NAMES = {
    "four_model_consensus": "Four models flagged",
    "three_model_consensus": "Three models flagged",
    "matrix_profile_only": "Only Matrix Profile flagged",
    "long_event": "Sustained model flag",
}
MODEL_NAMES = {model: label for model, label, _ in MODELS}


def category_name(case: pd.Series) -> str:
    if case["category"] == "single_model_only":
        return f"Only {MODEL_NAMES[str(case['source_model'])]} flagged"
    if case["category"] == "long_event":
        return f"Sustained {MODEL_NAMES[str(case['source_model'])]} flag"
    return CATEGORY_NAMES.get(str(case["category"]), str(case["category"]))


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def truth(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].astype(str).str.lower().eq("true").to_numpy(dtype=bool)


def figure_for_case(case: pd.Series, timeline: pd.DataFrame,
                    raw: pd.DataFrame, destination: Path) -> None:
    case_id = str(case["case_id"])
    interval = timeline.loc[timeline["case_id"].eq(case_id)].sort_values("minutes_from_anchor")
    points = raw.loc[raw["case_id"].eq(case_id)].sort_values("minutes_from_anchor")
    if len(interval) != 192 or len(points) != 576:
        raise ValueError(f"Expected complete 48-hour context for {case_id}")
    x15 = numeric(interval, "minutes_from_anchor") / 60.0
    x5 = numeric(points, "minutes_from_anchor") / 60.0
    anchor = str(case["anchor_start"])
    category = category_name(case)
    event_hours = pd.to_numeric(case.get("event_duration_minutes"), errors="coerce") / 60.0

    fig, axes = plt.subplots(
        4, 1, figsize=(13.2, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.35, 1.85, 1.1], "hspace": 0.12},
    )
    fig.patch.set_facecolor("white")
    flagged_count = int(case["flagged_model_count"])
    fig.suptitle(f"{case_id}  |  {case['series_id']}  |  {category}  |  {flagged_count}/4 flagged at anchor",
                 y=0.985,
                 fontsize=15, fontweight="medium")
    subtitle = f"Anchor: {anchor}   |   24 h before / 24 h after"
    if pd.notna(event_hours):
        subtitle += f"   |   derived run: {event_hours:.1f} h"
        if event_hours > 24:
            subtitle += " (extends beyond view)"
    fig.text(0.5, 0.948, subtitle, ha="center", va="top", fontsize=10, color="#444444")

    power = axes[0]
    power.plot(x5, numeric(points, "pRealKw"), color="#343A40", linewidth=0.85,
               alpha=0.78, label="Raw pRealKw (5 min)")
    power.plot(x15, numeric(interval, "pRealKw_mean"), color="#4C78A8",
               linewidth=1.55, alpha=0.95, drawstyle="steps-post",
               label="Mean pRealKw (15 min)")
    power.set_ylabel("Real power (kW)")
    power.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=False)

    current = axes[1]
    current.plot(x5, numeric(points, "iRMSMax"), color="#697E8A", linewidth=0.85,
                 alpha=0.85, label="Raw iRMSMax (5 min)")
    current.set_ylabel("Max RMS current (A)")
    current.legend(loc="upper left", fontsize=8.5, frameon=False)

    scores = axes[2]
    for model, label, color in MODELS:
        values = numeric(interval, f"{model}_max_percentile")
        scores.plot(x15, values, color=color, linewidth=1.45,
                    drawstyle="steps-post" if model == "matrix_profile" else "default",
                    label=label)
        at_anchor = np.flatnonzero(np.isclose(x15, 0.0))
        if len(at_anchor) and np.isfinite(values[at_anchor[0]]):
            scores.scatter([0], [values[at_anchor[0]]], color=color, s=28, zorder=4)
    scores.set_ylim(-0.035, 1.055)
    scores.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    scores.set_ylabel("Max percentile (0–1)")
    scores.legend(loc="upper left", ncol=4, fontsize=8.2, frameon=False)

    flags = axes[3]
    for index, (model, label, color) in enumerate(MODELS):
        y = 3 - index
        available = truth(interval, f"{model}_available")
        marked = truth(interval, f"{model}_is_anomaly") & available
        missing = ~available
        flags.hlines(y, -24, 24, color="#E3E7EA", linewidth=4, zorder=1)
        if missing.any():
            flags.scatter(x15[missing], np.full(missing.sum(), y), color="#9AA1A7",
                          marker="x", s=13, linewidths=0.7, zorder=2)
        if marked.any():
            flags.scatter(x15[marked], np.full(marked.sum(), y), color=color,
                          marker="s", s=19, linewidths=0, zorder=3)
    flags.set_ylim(-0.65, 3.65)
    flags.set_yticks([3, 2, 1, 0], [label for _, label, _ in MODELS])
    flags.set_ylabel("Flagged intervals")
    flags.set_xlabel("Hours relative to selected 15-minute interval")

    for axis in axes:
        axis.axvline(0, color="#D1495B", linewidth=1.2, linestyle="--", zorder=5)
        axis.set_xlim(-24, 24)
        axis.grid(axis="x", color="#E8EBED", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8.5)
    flags.set_xticks(np.arange(-24, 25, 6))
    fig.text(
        0.5, 0.015,
        "Dashed line = selected interval. Empty score = unavailable; MP repeats each native 30-minute score. Flags are candidates, not verified faults.",
        ha="center", fontsize=8.4, color="#555555",
    )
    fig.subplots_adjust(top=0.91, bottom=0.085, left=0.11, right=0.985)
    fig.savefig(destination, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_index(cases: pd.DataFrame, output: Path) -> None:
    cards = []
    for _, case in cases.iterrows():
        case_id = str(case["case_id"])
        category = category_name(case)
        series = html.escape(str(case["series_id"]))
        anchor = html.escape(str(case["anchor_start"]))
        flagged_count = int(case["flagged_model_count"])
        cards.append(
            f'<article><a href="{case_id}.png"><img src="{case_id}.png" alt="{case_id} time-series chart" loading="lazy"></a>'
            f'<p><strong>{case_id}</strong> · {html.escape(category)} · {flagged_count}/4 flagged at anchor · {series}'
            f'<br><small>{anchor} · Pending manual review</small></p></article>'
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>August manual-review case charts</title>
<style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0 auto;max-width:1440px;padding:20px;color:#20252a;background:#fff}
h1{font-size:22px;margin:0 0 8px}p.note{color:#555;margin:0 0 20px;line-height:1.5}
section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
article{min-width:0;border-bottom:1px solid #ddd;padding-bottom:12px}article img{width:100%;height:auto;display:block}
article p{margin:8px 0 0;line-height:1.5;font-size:14px}small{color:#666}
@media(max-width:800px){section{grid-template-columns:1fr}}
</style></head><body>
<h1>August manual-review case charts</h1>
<p class="note">Each chart shows 24 hours before and after the selected interval. The top panels show raw power and current; the third panel shows each model's maximum percentile; the bottom panel shows model flags by interval. The red dashed line marks the selected interval. Every case is pending manual review: a model flag is not a verified fault. Click a chart to open the full-size image. Matrix Profile repeats each native 30-minute score on two 15-minute intervals.</p>
<section>""" + "\n".join(cards) + "</section></body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path,
                        default=ROOT / "runs/ensemble/manual_review_august_v3")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    output = (args.output_dir or review / "plots").resolve()
    inputs = [review / name for name in (
        "review_cases.csv", "review_timeline_15min.csv", "review_raw_5min.csv"
    )]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output contains files; choose a new --output-dir: {output}")

    cases = pd.read_csv(inputs[0], dtype={"case_id": str, "series_id": str})
    timeline = pd.read_csv(inputs[1], dtype={"case_id": str, "series_id": str})
    raw = pd.read_csv(inputs[2], dtype={"case_id": str, "series_id": str})
    if cases["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs")
    output.mkdir(parents=True, exist_ok=True)
    for _, case in cases.iterrows():
        figure_for_case(case, timeline, raw, output / f"{case['case_id']}.png")
    make_index(cases, output)
    metadata = {"case_count": len(cases), "figures": [f"{item}.png" for item in cases.case_id]}
    (output / "plot_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "figures": len(cases)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
