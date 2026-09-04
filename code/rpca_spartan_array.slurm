#!/bin/bash
#SBATCH --job-name=rpca
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/punim1257/Group14/results/rpca_logs/rpca_%A_%a.out
#SBATCH --error=/data/projects/punim1257/Group14/results/rpca_logs/rpca_%A_%a.err

set -euo pipefail
module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
MANIFEST=$PROJECT/rpca_channels.csv
OUTDIR=$PROJECT/results/rpca_july
SCRIPT=$PROJECT/code/rpca_spartan_train_test.py

mkdir -p "$OUTDIR" "$PROJECT/results/rpca_logs"

# Reuse the project virtual environment, but do not install packages here.
source "$PROJECT/venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Array index is zero-based. The CSV has a header, hence +2 for sed.
LINE=$((SLURM_ARRAY_TASK_ID + 2))
METER_CHANNEL=$(sed -n "${LINE}p" "$MANIFEST" | cut -d',' -f1)

if [[ -z "$METER_CHANNEL" ]]; then
    echo "No meter_channel found for array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

CSV="$PROJECT/meter_csvs/${METER_CHANNEL}.csv"

echo "Running: $METER_CHANNEL"
echo "Input:   $CSV"
echo "Output:  $OUTDIR"

python "$SCRIPT" \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --test-start 2026-07-01 \
  --test-end 2026-08-01 \
  --threshold-quantile 0.99 \
  --power-cutoff 0.5
