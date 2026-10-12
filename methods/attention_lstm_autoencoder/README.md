# Independent Attention LSTM Autoencoder

This method retains the teammate's LSTM encoder, temporal attention, GELU
latent projections and LSTM decoder. It is a separate model version, not a
replacement for `methods/lstm_autoencoder/`. Original AE weights cannot be
loaded into the Attention network. Matching pretrained Attention weights can
be reused without retraining.

## Directory roles

- `src/`: imported disk-sharded training, model, calibration, resume and frozen-test replay implementation. Only the output model name is changed to `attention_lstm_autoencoder`; the network and training/scoring calculations are retained.
- `project.py`: repository-relative paths, training input view, window contract and category mapping.
- `inference.py`: separate frozen-model inference on a new dataset, using observed pre-test context without training or recalibration.
- `main.py`: `validate`, `infer`, `prepare`, `train`, `predict`, `export` and `train-by-category` commands.
- `train_by_category.py`: imported category grouping runner; each category receives a separate input view, cache and output directory.
- `jobs/`: an explicit frozen-inference Slurm template; no jobs are submitted by integration.
- `tests/`: dependency-light adapter tests and optional real PyTorch network checks.
- `requirements.txt`: dependencies for an isolated Attention environment; existing environments are not modified.
- `../../configs/attention_lstm_autoencoder.yaml`: candidate settings for a NEW August training run, not the teammate's verified production configuration.
- `../../configs/attention_lstm_autoencoder_inference.json`: shared September dataset and category-package locations.
- `../../shared/scripts/run_attention_lstm_autoencoder.py`: project-level command alias.

All integration paths resolve from `group14reshape`, regardless of shell cwd.
Backup folders are never imported or read by runtime/tests. Existing models,
August/September results and the default four-model fusion are unchanged.
The uploaded `lstmaecodes/` originals and `codes.zip` are retained unchanged in
`legacy/attention_lstm_import/`, which is ignored by Git and not required by any
command or test. The active implementation is entirely in this method directory.

## Two separate workflows

### Imported training and replay

The native pipeline still partitions pre-test history into training,
validation and held-out calibration. It retains bounded disk shards, mmap
windows, atomic checkpoints and resume checks. Training fits a RobustScaler
only on training rows; calibration freezes an exact or explicitly approximate
sample ECDF. This differs from the old AE's training-score ECDF; do not silently
relabel held-out calibration as `train_only`.

Native `prepare/train/predict` require Linux/Spartan/WSL because the imported
locks and signal handling are POSIX-specific. These commands retain
`train_end == test_start` and do not borrow pre-test rows as test context.
`predict` replays only the original frozen prepared test dataset. It is **not**
the new-period inference command. Resume is for runs started by this integrated
version; do not force an older checkpoint through changed code/path identities.

The adapter creates a symlink-only input view, excluding `_manifest.csv` and
checking the shared twelve-column schema. It does not move or rewrite original
data. Native preprocessing then retains the teammate's SQLite resampling,
duplicate resolution and bounded offline interpolation. Cache and input views
are isolated under `data/processed/attention_lstm_autoencoder/`.

```bash
# Run from group14reshape in an activated, compatible Linux environment.
# These are explicit NEW-run commands, never launched automatically.
python shared/scripts/run_attention_lstm_autoencoder.py prepare
python shared/scripts/run_attention_lstm_autoencoder.py train
python shared/scripts/run_attention_lstm_autoencoder.py train --resume
python shared/scripts/run_attention_lstm_autoencoder.py predict
python shared/scripts/run_attention_lstm_autoencoder.py export

# Group first without fitting; remove --prepare-only only to explicitly train.
python shared/scripts/run_attention_lstm_autoencoder.py train-by-category --prepare-only
```

Confirm the real feature order, model sizes and training settings before new
training. The centralized YAML defaults are not evidence of how the teammate's
existing models were trained. `train-by-category` must not be used to refit on
injected data as a substitute for paired frozen-model evaluation.

### Frozen inference on a new time period

`infer` does not call native preparation, scaler fitting, training or
calibration. It loads a separate trained package for each configured category.
The default profile retains cutoff `2026-08-01`, uses August 31 observations
for context, and scores `[2026-09-01, 2026-10-01)` in Australia/Sydney time.
July rows in shared inputs are not used as a substitute for August 31 context.

For each configured category (1, 2, 4, 5), place the actual package at:

```text
runs/attention_lstm_autoencoder/models/category_<id>/
  best_model.pt
  config.json
  scaler.json
  metadata.json
  calibration/
    max_sorted.npy
    mean_sorted.npy
```

The expected metadata is the supplied pipeline's completed training metadata:
`source=held_out_calibration`, `split_config`, `binding`, threshold, score
definition, sample counts/approximation flag, and hashes for weights, scaler
and calibration arrays. Hashes, ordered features, cutoff and model/scoring
configuration are checked. Safe `torch.load(weights_only=True)` and strict
state loading reject old AE weights or mismatched training configurations;
there is no unsafe pickle fallback or partial/random-attention load.

```powershell
# Works without PyTorch/weights: report readiness, never create results or train.
# Exit 2 means required packages or PyTorch are missing/incompatible.
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_attention_lstm_autoencoder.py validate
```

After placing genuine packages and activating an isolated PyTorch environment:

```bash
python shared/scripts/run_attention_lstm_autoencoder.py validate
python shared/scripts/run_attention_lstm_autoencoder.py infer --max-channels 1 --output-dir runs/attention_lstm_autoencoder/results/september_smoke
python shared/scripts/run_attention_lstm_autoencoder.py infer
```

Use `--series-id` to select actual channels; `--max-channels` selects channels,
not fragments of a month. `--verbose` expands the validation channel report.
Paths can be changed in the inference JSON. Unsupported categories produce
explicit unavailable rows; a missing package for a selected supported category
stops before result creation. Nonempty results, original inputs and package
directories are protected from overwriting. No dependencies are installed
automatically. Native training is Linux-only; the new inference adapter is
cross-platform when compatible PyTorch and model packages are available.

## Input, window and output contract

Input is the existing shared CSV with timestamp plus eleven power features.
Inference uses the trained ordered subset of those features. Strict shared
normalization rejects duplicate/off-grid timestamps, wrong columns and
infinite values. It does not recompute positive/negative power.

Each interval uses 288 five-minute points (24 hours), advances three points
(15 minutes), and scores only its last three point reconstruction errors.
Max, Mean and population Std are retained. The first interval needs 285 real
pre-test points (23 hours 45 minutes); the shared full-day context supplies
them when available. Missing context is not zero-filled or replaced by July.
The trained bounded internal interpolation policy is retained, with original
observation counts/coverage. It is offline interpolation, not strict real-time
inference without future observations.

```text
runs/attention_lstm_autoencoder/results/<run_name>/
  channels/<series_id>_scores_15min.csv
  ALL_SCORES_15MIN.csv
  ALL_EVENTS.csv
  run_summary.csv
  channel_status.csv
  run_metadata.json
  inference_config.json
  dataset_profile.json
```

Score/event fields use the existing shared keys and statistics, with the
independent `model=attention_lstm_autoencoder`. Unavailable scores/flags stay
missing; unsupported-category coverage remains unknown rather than zero.
Import nullable flags as Pandas `boolean`. `calibration_version/model_version`
retain the original trained binding. `training_end_exclusive`, calibration
source and approximation status are separate explicit fields, not fabricated
date-prefixed versions. Stable event IDs include dataset, model version,
channel and start/end. Events describe occupied candidate bins, not confirmed
faults.

The default ensemble deliberately does not register this additional model.
`ensemble_eligible=false` prevents silently mixing it with the old AE or
counting two closely related variants as independent votes. Future model
selection/fusion requires an explicit policy decision and validation.

The earlier `lstm_results_new` files are not relabelled as Attention outputs:
matching column names alone do not prove they came from this exact network
version. Confirm their trained package/binding before importing them.

## Verification status

Adapter tests use synthetic inputs/packages and a documented NumPy scoring
stub to test I/O, period isolation, coverage, checksum/cutoff rejection,
unavailable states and output protection without claiming neural execution.
Optional PyTorch tests verify network forward/backward and old-weight rejection
when PyTorch is installed. This host currently lacks PyTorch and genuine
category model packages; no real neural training or inference was run.
