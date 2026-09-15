#!/bin/bash
#SBATCH --job-name=vae_anomaly
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-49               
#SBATCH --output=/data/gpfs/projects/punim1257/Group14/runs/lstm_vae/logs/vae_%A_%a.out
#SBATCH --error=/data/gpfs/projects/punim1257/Group14/runs/lstm_vae/logs/vae_%A_%a.err

# ---- edit these paths for your project ----
PROJECT_DIR=/data/gpfs/projects/punim1257/Group14
CODE_DIR=$PROJECT_DIR/methods/lstm_vae/src
CSV_DIR=$PROJECT_DIR/data/processed/meter_csvs
OUT_DIR=$PROJECT_DIR/runs/lstm_vae/august
# ---------------------------------------------

mkdir -p "$OUT_DIR" "$PROJECT_DIR/runs/lstm_vae/logs"

module purge
module load foss/2022a
module load Python/3.10.4

source "$PROJECT_DIR/venv/bin/activate"

cd "$CODE_DIR"

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python run_train_test_anomaly.py \
    --csv_dir "$CSV_DIR" \
    --out_dir "$OUT_DIR" \
    --train_end 2026-08-01 \
    --test_start 2026-08-01 \
    --test_end 2026-09-01 \
    --window 288 \
    --stride 3 \
    --epochs 30 \
    --threshold_pct 99.5 \
    --array_index $SLURM_ARRAY_TASK_ID \
    --array_count $SLURM_ARRAY_TASK_COUNT
