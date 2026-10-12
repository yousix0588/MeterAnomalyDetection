# Batch LOF integration

This directory adapts the imported per-channel LOF detector without changing
its seven features, `StandardScaler`, `n_neighbors=20`, `contamination=0.005`
or `fit_predict` computation. It does not implement novelty detection, rolling
windows, pretrained inference or a frozen July model.

- `src/lof_batch.py`: native scoring, 15-minute statistics, event views and optional plots.
- `tests/test_lof_adapter.py`: algorithm equivalence, missing-data, period and output-protection tests.
- `requirements.txt`: dependencies for an independently installed LOF environment.
- `../../configs/lof.json`: model settings and shared dataset profile reference.
- `../../shared/scripts/run_lof_local.py`: project-relative executable entry point.
- `../../runs/lof/<run_name>/`: generated native and aligned results, excluded from Git.

The imported tree is archived under `legacy/lof_import/MeterAnomalyDetection-main/`,
with all original file contents unchanged, including its historical VAE,
RPCA, Matrix Profile and merge scripts. Those duplicate
algorithms are not copied over the current four models. The new entry point
replaces the original script's broken feature-list syntax, hardcoded August
paths and plotting tail; do not execute that historical script directly.

The entire `legacy/` directory is an ignored local backup only. Runtime and
tests never inspect, read or import it, and both work without that directory.
Algorithm preservation is checked against an independent reference computation
and explicit feature/parameter assertions instead of archived source files.
New runs record the current implementation path and code hashes, with
`backup_required=false`. Existing run metadata remains a historical snapshot
and is not rewritten. Keep `methods/lof/`, which contains the current
executable implementation; neither the old top-level `LOF/` nor its backup is
required to run or test the model.

## Input and fit scope

Reuse the shared, per-channel, twelve-column five-minute CSVs. Only rows inside
the dataset profile's `[test_start_inclusive, test_end_exclusive)` interval
are fitted and scored. July history and August 31 context are not fitted. A
feature-incomplete observed row is retained as unavailable; a channel with
fewer than 21 feature-complete target rows is not fitted. Missing timestamps
are not interpolated or fabricated. Shared normalization rejects duplicates,
wrong columns, off-grid timestamps and infinite values.

This is **retrospective batch outlier detection**. Both standardization and
the neighbourhood model use the whole target batch. It must not be described
as a frozen training model, an online detector, or leakage-free train/test
evaluation. Changing the batch dates changes the fit and can change labels.
`contamination=0.005` controls the model's batch threshold, not a measured
fault prevalence, accuracy or guaranteed event count.

## Run from any working directory

From the project root on Windows:

```powershell
# First verify one full-month channel. Do not split the month into fit chunks.
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_lof_local.py --max-channels 1 --output-dir runs/lof/september_2026_smoke

# Explicitly launch all channels when ready; this was not run during integration.
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_lof_local.py --output-dir runs/lof/september_2026_batch

# Optional plots use newly generated native events lasting at least 15 minutes.
# Choose another run directory rather than overwriting an earlier run.
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_lof_local.py --series-id DD03710131709_2 --output-dir runs/lof/september_2026_selected --plots
```

The existing local Matrix Profile environment has the LOF dependencies. Using
it does not change or install packages. For another machine, create an isolated
environment and install `methods/lof/requirements.txt`; then invoke the same
entry point using that Python executable. On Spartan, run within a suitable
allocated compute job, not a full-data job on the login node. No existing
Slurm job or environment is edited by this adapter.

Override `--dataset-profile`, `--input-dir`, `--output-dir` or repeat
`--series-id` as needed. Relative paths always resolve from `group14reshape`.
Paths outside the repository and output directories outside `runs/lof` are
rejected. Nonempty output directories are never overwritten; failed runs are
marked in metadata and require a new directory for another attempt.

## Output contract

```text
runs/lof/<run_name>/
  channels/<series_id>/
    point_scores_5min.csv          All observed target rows, including unavailable rows
    scores_15min.csv               Full target-period grid
    events_5min.csv                Consecutive native flagged-point runs
  ALL_ANOMALIES_5MIN.csv            Original flagged points with extra provenance fields
  ALL_SCORES_15MIN.csv              All channels' aligned statistics
  ALL_EVENTS_5MIN.csv               Native flagged-point runs, including single-point spikes
  ALL_EVENTS_NATIVE_MIN15MIN.csv    Subset lasting at least 15 consecutive native minutes
  ALL_EVENTS_15MIN.csv              Occupied 15-minute-bin runs for time-key comparison
  run_summary.csv                  Coverage, flags, counts and native thresholds per channel
  run_metadata.json                Batch fit scope, input/code hashes, versions and fit details
  lof_config.json                  Configuration snapshot
  dataset_profile.json             Shared dataset profile snapshot
  plots/                           Five optional English plots
```

Point scores keep `lof_score=-negative_outlier_factor_`, the original
`lof_label` (`-1` flagged, `1` not flagged) and threshold `-offset_`. The
15-minute `raw_score` is `max_score`; `mean_score` and population `score_std`
describe the available native points in that bin. A bin is flagged if any
available native point is flagged. Native labels are not recalculated from a
new aggregated threshold. `valid_point_count/3` gives `coverage_ratio` and
`classification_complete` means all three native points were scored. A false
flag in a partially covered bin refers only to the available points.

An unavailable bin has missing scores and a nullable `is_anomaly`, not a
normal label or zero score. Import nullable flags as Pandas `boolean` and
native labels as `Int64` when reading CSV.

Common keys remain `model=lof`, `series_id`, `interval_start`, `interval_end`
and event keys `model`, `event_id`, `series_id`, `start`, `end`. Native and
aligned event identities differ deliberately. A single flagged five-minute
point occupies a fifteen-minute bin, but **does not establish fifteen minutes
of continuous abnormal behaviour**. Use native events for actual flagged
durations, and aligned events only for bin-level comparison.

`max_percentile` and `mean_percentile` remain empty. Every output declares
`calibration_version=lof_batch_not_frozen_v1` and `ensemble_eligible=False`.
Do not populate the frozen dataset profile's `models` with this sidecar,
rename it to another model, rank target scores into fake training percentiles,
or use the imported `merge/formalize_anomalies.py` target-ranking routine as
frozen calibration. The existing four-model validator deliberately rejects
these outputs. Join time keys for separate comparisons without changing the
original score table or adding LOF into voting/score fusion.
