#!/bin/bash
#SBATCH --job-name=lstm_ae_gpu
#SBATCH --account=punim1257
#SBATCH --partition=fos-gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --qos=fos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-10:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/lstm_autoencoder/logs/ae_gpu_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/lstm_autoencoder/logs/ae_gpu_%j.err

PROJECT_DIR=/data/projects/punim1257/Group14
cd "$PROJECT_DIR"

module purge
module load foss/2022a
module load Python/3.10.4
module load CUDA/12.0.0 2>/dev/null || true

source "$PROJECT_DIR/venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== GPU Environment Info ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Starting LSTM Autoencoder training & evaluation on aligned August data with GPU..."
python methods/lstm_autoencoder/main.py train --config configs/lstm_autoencoder.yaml

echo "Done."
