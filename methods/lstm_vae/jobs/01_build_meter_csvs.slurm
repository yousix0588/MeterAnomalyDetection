#!/bin/bash
#SBATCH --job-name=build_meter_csvs
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=/data/gpfs/projects/punim1257/Group14/runs/lstm_vae/logs/build_%j.out
#SBATCH --error=/data/gpfs/projects/punim1257/Group14/runs/lstm_vae/logs/build_%j.err

# ---- edit these paths for your project ----
PROJECT_DIR=/data/gpfs/projects/punim1257/Group14

# BOTH datasets
DATA_PATH=(
    "$PROJECT_DIR/data/raw/RACV+InchScape_25Oct-19Aug.zip"
    "$PROJECT_DIR/data/raw/Aug_meters.zip"
)

CODE_DIR="$PROJECT_DIR/shared/data_tools"
OUT_DIR="$PROJECT_DIR/data/processed/meter_csvs"
# ---------------------------------------------

mkdir -p "$OUT_DIR" "$PROJECT_DIR/runs/lstm_vae/logs"

module purge
module load foss/2022a
module load Python/3.10.4

source "$PROJECT_DIR/venv/bin/activate"

cd "$CODE_DIR"

echo "Input datasets:"
printf '  %s\n' "${DATA_PATH[@]}"
echo

# Check that both datasets exist
for DATASET in "${DATA_PATH[@]}"; do
    if [ ! -f "$DATASET" ]; then
        echo "ERROR: Dataset not found:"
        echo "$DATASET"
        exit 1
    fi
done

python build_meter_csvs.py \
    --data "${DATA_PATH[@]}" \
    --out_dir "$OUT_DIR" \
    --date_end 2026-08-31 \
    --require_start 2026-08-01 \
    --require_end 2026-08-31

echo
echo "Build finished. Meter CSV count:"
ls "$OUT_DIR"/*.csv 2>/dev/null | wc -l
