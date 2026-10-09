#!/bin/bash
#SBATCH --job-name=rpca_augsep_meter
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/punim1257/Group14/results/rpca_augsep_logs/meter_%A_%a.out
#SBATCH --error=/data/projects/punim1257/Group14/results/rpca_augsep_logs/meter_%A_%a.err

set -euo pipefail

module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
source "$PROJECT/venv/bin/activate"

METER_MANIFEST="$PROJECT/rpca_meters_augsep_usable.csv"
CHANNEL_MANIFEST="$PROJECT/rpca_channels_augsep_usable.csv"
METER_DIR="$PROJECT/meter_csvs_augsep_rpca"
OUTDIR="$PROJECT/results/rpca_augsep_meter_q995"
SCRIPT="$PROJECT/code/rpca_spartan_train_test_meter.py"

mkdir -p "$OUTDIR" "$PROJECT/results/rpca_augsep_logs"

LINE=$((SLURM_ARRAY_TASK_ID + 2))
METER_ID=$(sed -n "${LINE}p" "$METER_MANIFEST" | tr -d '\r')

echo "Running meter: $METER_ID"

python "$SCRIPT" \
  --meter-id "$METER_ID" \
  --manifest "$CHANNEL_MANIFEST" \
  --meter-dir "$METER_DIR" \
  --outdir "$OUTDIR" \
  --test-start 2026-08-01 \
  --test-end 2026-10-01 \
  --threshold-quantile 0.995 \
  --power-cutoff 0.5
