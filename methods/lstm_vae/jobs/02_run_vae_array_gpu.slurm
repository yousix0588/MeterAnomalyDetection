#!/bin/bash
#SBATCH --job-name=vae_anom_gpu
#SBATCH --account=punim1257
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-49
#SBATCH --output=/data/projects/punim1257/Group14/runs/lstm_vae/logs/vae_gpu_%A_%a.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/lstm_vae/logs/vae_gpu_%A_%a.err

# ---- edit these paths for your project ----
PROJECT_DIR=/data/projects/punim1257/Group14
CODE_DIR=$PROJECT_DIR/methods/lstm_vae/src
CSV_DIR=$PROJECT_DIR/data/processed/meter_csvs
OUT_DIR=$PROJECT_DIR/runs/lstm_vae/august
# ---------------------------------------------

mkdir -p "$OUT_DIR" "$PROJECT_DIR/runs/lstm_vae/logs"

module purge
module load foss/2022a
module load Python/3.10.4
module load CUDA/12.0.0 2>/dev/null || true

source "$PROJECT_DIR/venv/bin/activate"

cd "$CODE_DIR"

echo "=== GPU Environment Info ==="
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
