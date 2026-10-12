# Configuration

YAML configuration is centralized in this directory. The imported repository did not contain its original YAML files. The LSTM Autoencoder configuration was reconstructed from its saved checkpoint and updated to the aligned August time window.

Files:

- `common.yaml`: authoritative sampling interval, window policy, and train/test boundaries.
- `lstm_autoencoder.yaml`: executable LSTM Autoencoder data, split, model, and training settings.
- `lstm_vae.yaml`: documented LSTM-VAE time and window settings; the current runner still receives them through CLI/Slurm.
- `rpca.yaml`: documented RPCA time and daily-window settings; the current runner still receives them through CLI/Slurm.
- `matrix_profile.yaml`: documented Matrix Profile time and multi-scale settings; the current runner still receives them through CLI/Slurm.
- `lof.json`: executable settings for the separate batch LOF adapter; original features and parameters are retained, with `ensemble_eligible=false`.
- `attention_lstm_autoencoder.yaml`: complete candidate settings for a new, independent Attention AE training run; these are not inferred teammate production settings.
- `attention_lstm_autoencoder_inference.json`: frozen category packages and shared September input for the independent Attention inference entry point.

Slurm files remain under each model's `methods/<model>/jobs` directory.

Dataset-specific I/O profiles live in `datasets/`. `common.yaml` and the model
YAML files retain the original August defaults; later datasets do not silently
change those boundaries. `datasets/september_2026.json` is consumed by
`shared/scripts/align_dataset_workflow.py`. JSON keeps the lightweight data/CSV
adapter independent of a YAML or machine-learning runtime dependency.

The September profile declares the original-history and raw-delivery paths,
frozen July reference, August 31 window-only context, September test period,
shared derived input directory, saved model score inputs, review snapshot and
new output directory. Missing model score inputs are explicitly `null`; this
does not mean those models detected normal behaviour. Original training
histories, fitted scalers and ECDF artifacts are not replaced by the smaller
derived July/context/September input. The profile does not launch training.

`output_contract.yaml` also documents review sidecars. Those flags join on the
existing keys, preserve missing states and never replace model score flags.

LOF reads the same dataset profile's shared input path and target boundaries,
but is not added to that profile's frozen four-model `models` mapping. Its
standardizer and neighbourhood model are fitted within the target batch;
therefore its aligned output is a non-frozen comparison sidecar, not a fifth
ECDF-calibrated ensemble input. JSON is used consistently with dataset I/O
profiles. Configuration changes to the algorithm's features, neighbours,
contamination or mode are rejected by this preservation-only adapter.

Attention training configuration paths resolve from the repository root through
its integration entry point. Its native pipeline retains a separate held-out
calibration split. The new `infer` command reuses verified frozen packages for
a different target period without moving the training cutoff or fitting on
context/target rows. It does not change the dataset profile's four-model mapping
or load files from `legacy/`. Category packages remain generated artifacts under
`runs/attention_lstm_autoencoder/models/` and are not committed with code.
