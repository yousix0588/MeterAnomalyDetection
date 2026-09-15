# Configuration

YAML configuration is centralized in this directory. The imported repository did not contain its original YAML files. The LSTM Autoencoder configuration was reconstructed from its saved checkpoint and updated to the aligned August time window.

Files:

- `common.yaml`: authoritative sampling interval, window policy, and train/test boundaries.
- `lstm_autoencoder.yaml`: executable LSTM Autoencoder data, split, model, and training settings.
- `lstm_vae.yaml`: documented LSTM-VAE time and window settings; the current runner still receives them through CLI/Slurm.
- `rpca.yaml`: documented RPCA time and daily-window settings; the current runner still receives them through CLI/Slurm.
- `matrix_profile.yaml`: documented Matrix Profile time and multi-scale settings; the current runner still receives them through CLI/Slurm.

Slurm files remain under each model's `methods/<model>/jobs` directory.
