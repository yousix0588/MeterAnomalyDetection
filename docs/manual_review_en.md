# August anomaly-candidate manual review

Run these commands from the `group14reshape` project directory:

```bash
python shared/scripts/build_manual_review_set.py --output-dir runs/ensemble/manual_review_august_v3
python shared/scripts/plot_manual_review_set.py --review-dir runs/ensemble/manual_review_august_v3
```

The commands create a separate English-language v3 review set. Existing v1 and v2 files are not overwritten. Both scripts stop if their output directory already contains files; choose a new output directory for another run.

## Case selection

The default selection contains 41 cases: six intervals flagged by all four models, six flagged by three models, six flagged only by Matrix Profile, up to two sustained events per model (eight in this run), and five intervals each flagged only by LSTM AE, LSTM-VAE, or RPCA. At each single-model anchor, all four model results are available, so an unflagged model is not confused with a missing result. The first 26 case IDs match v1/v2; the 15 added cases start at MR027.

Selection is deterministic, favors different devices, avoids choosing intervals less than 12 hours apart on the same series, and requires the full 24-hour context on each side to fall within August. The single-model quotas are intentionally balanced for review. They do not represent the models' alert frequencies in the full dataset, and the 41 cases cannot be used directly to estimate overall accuracy.

## Files

- `review_cases.csv`: one row per case, with anchor-time model flags and editable `review_status`, `review_label`, and `review_notes` columns. All cases initially have `review_status=unreviewed`. Suggested labels are `confirmed_anomaly`, `scheduled_operation`, `data_quality_issue`, `normal`, and `uncertain`.
- `review_timeline_15min.csv`: aligned 15-minute model scores and flags for each case, plus statistics calculated from the three original 5-minute measurements in each interval. Missing model results remain blank, not zero.
- `review_raw_5min.csv`: the original 5-minute measurements in the same 48-hour view, including all source measurement columns.
- `review_manifest.json`: input paths, selection rules, candidate counts, and output row counts.
- `plots/index.html`: a gallery linking to all 41 charts. Each chart shows raw power, raw current, model score percentiles, and interval flags. The red dashed line marks the selected 15-minute interval.

Timestamps retain the source `+10:00` offset and intervals are start-inclusive, end-exclusive. A sustained-event case is anchored at the start of a derived sequence of model flags; the 48-hour chart may not show the whole event. Matrix Profile has a native 30-minute resolution, so two adjacent 15-minute bins may repeat one score. A percentile near 1 is a rank against the calibration baseline, not a fault probability. Model flags and derived events are review candidates, not verified fault labels.

For each case, inspect the raw measurements, data continuity, recurring operating patterns, and other models before entering a human label. Once labeled, group results by `source_model`, `category`, and `review_label` before considering threshold or ensemble-rule changes.
