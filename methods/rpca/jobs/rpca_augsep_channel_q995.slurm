#!/bin/bash
#SBATCH --job-name=rpca_augsep_ch
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/punim1257/Group14/results/rpca_augsep_logs/channel_%A_%a.out
#SBATCH --error=/data/projects/punim1257/Group14/results/rpca_augsep_logs/channel_%A_%a.err

set -euo pipefail

module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
source "$PROJECT/venv/bin/activate"

MANIFEST="$PROJECT/rpca_channels_augsep.csv"
OUTDIR="$PROJECT/results/rpca_augsep_channel_q995"
SCRIPT="$PROJECT/code/rpca_spartan_train_test.py"

mkdir -p "$OUTDIR"

LINE=$((SLURM_ARRAY_TASK_ID + 2))
METER_CHANNEL=$(sed -n "${LINE}p" "$MANIFEST" | tr -d '\r')

CSV="$PROJECT/meter_csvs_augsep_rpca/${METER_CHANNEL}.csv"

echo "Running: $METER_CHANNEL"

python "$SCRIPT" \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --test-start 2026-08-01 \
  --test-end 2026-10-01 \
  --threshold-quantile 0.995 \
  --power-cutoff 0.5
