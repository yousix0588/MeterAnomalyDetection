#!/bin/bash
#SBATCH --job-name=lstm_ae
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/lstm_autoencoder/logs/ae_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/lstm_autoencoder/logs/ae_%j.err

PROJECT_DIR=/data/projects/punim1257/Group14
cd "$PROJECT_DIR"

module purge
module load foss/2022a
module load Python/3.10.4

source "$PROJECT_DIR/venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Starting LSTM Autoencoder training & evaluation on aligned August data..."
python methods/lstm_autoencoder/main.py train --config configs/lstm_autoencoder.yaml

echo "Done."
