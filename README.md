# Group 14 Power Anomaly Detection

This project contains four independent anomaly-detection methods for electrical meter data:

1. LSTM Autoencoder
2. LSTM Variational Autoencoder (LSTM-VAE)
3. Robust Principal Component Analysis (RPCA)
4. Multi-scale Matrix Profile

The current repository structure separates source code, cluster jobs, configuration, input data, and generated results. The four methods still retain their original internal implementations and output formats. A shared model interface and ensemble layer can be added in a later phase.

## Aligned time policy

The project uses half-open date boundaries so that midnight is never assigned to both training and testing:

- Training and validation: timestamps before `2026-08-01 00:00:00` in `Australia/Sydney` (the final training date is 31 July).
- Testing: timestamps from `2026-08-01 00:00:00` inclusive to `2026-09-01 00:00:00` exclusive (all of August).
- LSTM Autoencoder and LSTM-VAE: 24-hour windows (`288` points at five-minute sampling) with a 15-minute stride (`3` points). Each accepted test window must lie completely inside August.
- RPCA: one natural-day profile of `288` five-minute points; it does not use a sliding stride.
- Matrix Profile: data is internally aggregated to 30-minute intervals and uses 2-hour, 6-hour, and 24-hour windows (`4`, `12`, and `48` points) with a 30-minute stride.

The authoritative shared boundaries are recorded in `configs/common.yaml`. The Matrix Profile command-line interface uses an inclusive detection end date, so its equivalent range is 1 August through 31 August.

## Canonical model outputs

Each method retains its native artifacts and additionally produces a dense `scores_15min.csv` view for ensemble modelling. The canonical schema is defined by `configs/output_contract.yaml` and includes raw, maximum, mean, and standard-deviation scores, frozen train-only ECDF percentiles, coverage, source resolution, and data-status fields.

- The LSTM Autoencoder and LSTM-VAE use the previous 24 hours as causal context but score only the final three five-minute samples. They do not average a window score across the whole day.
- RPCA aggregates each group of three native five-minute residual scores and keeps both the maximum and mean.
- Matrix Profile retains `scores_30min.csv` as its native output. Its `scores_15min.csv` compatibility view repeats each 30-minute value and explicitly sets `data_status=upsampled_from_30m` and `source_resolution_minutes=30`.
- ECDF calibration artifacts are fitted only from timestamps before 1 August and are not updated from test-period scores.

Run `shared/scripts/build_ensemble_features.py` with one or more `--input` arguments to validate canonical files and create `all_models_scores_15min.csv`, the wide `ensemble_features_15min.csv`, `all_models_events.csv`, and `run_summary.csv`. Missing model outputs remain unavailable rather than being filled with zero.

## Directory structure

```text
group14reshape/
|-- configs/
|-- data/
|   |-- raw/
|   |-- processed/
|   |   `-- meter_csvs/
|   `-- manifests/
|-- docs/
|-- legacy/
|-- methods/
|   |-- lstm_autoencoder/
|   |-- lstm_vae/
|   |-- rpca/
|   `-- matrix_profile/
|-- runs/
|   |-- lstm_autoencoder/
|   |-- lstm_vae/
|   |-- rpca/
|   `-- matrix_profile/
`-- shared/
    |-- data_tools/
    `-- scripts/
```

## Top-level directories

### `configs/`

Stores YAML configuration files in one central location. `common.yaml` defines the shared time boundaries and window policy. The LSTM Autoencoder YAML is executable, while the other model YAML files document the values currently passed through their CLI and Slurm scripts. See `configs/README.md` for details.

### `data/`

Contains input data and dataset indexes. Generated model results must not be stored here.

- `data/raw/`: original archives or extracted daily meter files. These files are not modified by the models and are ignored by Git.
- `data/processed/`: data produced by preprocessing and intended to be consumed by the detection methods.
- `data/processed/meter_csvs/`: one chronologically ordered CSV per meter-channel, plus the generated `_manifest.csv` when available. This is the intended common input location for the four methods.
- `data/manifests/`: small tracked files that enumerate datasets, meter IDs, channels, or cluster-array work items. The existing RPCA July and August channel lists are stored here.

### `docs/`

Reserved for project-level documentation, experiment descriptions, architecture notes, result interpretation, and ensemble-design documents. Method-specific operating instructions should remain with the corresponding method.

### `legacy/`

Reserved for old scripts or files that must be retained for reference but are no longer part of the active workflow. Files placed here should not be imported by active model code.

### `methods/`

Contains the implementation of each anomaly-detection method. Each method owns its source code and its model-specific Spartan jobs. Generated outputs do not belong in this directory.

### `runs/`

Contains generated experiment artifacts such as anomaly CSV files, plots, logs, checkpoints, scalers, and training histories. Results are separated by method so that similarly named files from different algorithms do not overwrite one another. This entire directory is ignored by Git.

### `shared/`

Contains reusable project utilities that are not owned by a single detection method.

- `shared/data_tools/`: shared ingestion and preprocessing programs. `build_meter_csvs.py` combines daily files from directories or archives into per-meter-channel CSV files.
- `shared/scripts/`: reserved for future project-wide orchestration, validation, or aggregation scripts. Model-specific scripts should not be placed here.

## Method directories

### `methods/lstm_autoencoder/`

Contains the deterministic LSTM Autoencoder. The model encodes each time-series window into a latent representation, reconstructs the input, and uses reconstruction error as its anomaly score.

- `main.py`: command-line entry point for training and prediction.
- `src/`: data loading, window generation, LSTM Autoencoder architecture, training pipeline, scoring, plotting, and utility functions.

This method currently expects its configuration at `configs/lstm_autoencoder.yaml`. No Slurm files were included with the imported LSTM Autoencoder code, so it does not yet have a `jobs/` directory.

### `methods/lstm_vae/`

Contains the LSTM Variational Autoencoder detector. It learns a probabilistic latent representation and detects anomalies from reconstruction-based window scores.

- `src/`: LSTM-VAE model, single-series pipeline, multi-meter train/test runner, and array-result aggregation code.
- `jobs/`: Spartan Slurm scripts for building processed meter CSVs, running the VAE job array, and aggregating array outputs.

### `methods/rpca/`

Contains the Robust PCA detector. It learns a robust low-rank representation of historical daily electrical profiles and detects anomalous residuals in the test period.

- `src/`: RPCA training and scoring, manifest preparation, result aggregation, and event-export programs.
- `jobs/`: Spartan Slurm scripts for test runs, July and August job arrays, result aggregation, and event exports.

### `methods/matrix_profile/`

Contains the multi-scale Matrix Profile detector. It evaluates short, medium, and daily temporal patterns and includes event construction, data-quality checks, rolling history, shard aggregation, and evidence plotting.

- `src/`: the Matrix Profile Python package, detection entry point, shard aggregation, plotting code, and meter metadata repository.
- `jobs/`: Spartan environment setup, deployment, preflight, array execution, aggregation, plotting, and submission scripts, plus the Matrix Profile dependency list.

## Run directories

### `runs/lstm_autoencoder/`

Stores artifacts produced by the LSTM Autoencoder.

- `outputs/`: original standard output directory.
- `outputs_all/`: original full-run output directory.
- `outputs_smoke/`: small smoke-test outputs used to confirm that the pipeline runs.

### `runs/lstm_vae/`

Stores historical LSTM-VAE experiments without merging or deleting their original results.

- `august/`: primary August experiment outputs, formerly `out_aug`.
- `august_legacy/`: an additional retained August result set, formerly `vae_out_aug`.
- `july/`: retained July VAE outputs, formerly `vae_out_july`.
- `july_secondary/`: the additional July VAE result set formerly stored under `results/vae_july`.

### `runs/rpca/`

Stores RPCA experiments and cluster logs.

- `july/`: July RPCA results.
- `august/`: August RPCA results.
- `test/`: small RPCA test-run outputs.
- `logs/`: Slurm standard-output and error logs for RPCA jobs.

### `runs/matrix_profile/`

Stores Matrix Profile runtime artifacts.

- `results/`: preflight, shard, aggregate, event, summary, and selected-plot outputs.
- `logs/`: Slurm standard-output and error logs.

## Path conventions

- Source code: `methods/<method>/src`
- Method-specific Slurm files: `methods/<method>/jobs`
- Raw input: `data/raw`
- Processed meter input: `data/processed/meter_csvs`
- Dataset and channel lists: `data/manifests`
- Generated outputs: `runs/<method>`
- Shared preprocessing: `shared/data_tools`

The existing Spartan scripts retain their original project roots, either `/data/projects/punim1257/Group14` or `/data/gpfs/projects/punim1257/Group14`, while all child paths follow the structure documented above.
