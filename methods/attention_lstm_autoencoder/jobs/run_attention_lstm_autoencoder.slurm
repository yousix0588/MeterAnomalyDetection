#!/bin/bash
#SBATCH --job-name=attention_ae
#SBATCH --account=punim1257
#SBATCH --partition=fos-gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --qos=fos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=attention_ae_%j.out
#SBATCH --error=attention_ae_%j.err

# Template uses the existing project's allocation settings; verify eligibility
# and activate a compatible Attention environment before submitting.
set -euo pipefail
PROJECT_DIR="${ATTENTION_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA environment required"'
# Explicit frozen inference only. This job never starts training.
python shared/scripts/run_attention_lstm_autoencoder.py infer --config configs/attention_lstm_autoencoder_inference.json
