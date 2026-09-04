#!/bin/bash
#SBATCH --job-name=rpca_events
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/data/projects/punim1257/Group14/results/rpca_logs/rpca_events_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/results/rpca_logs/rpca_events_%j.err

set -euo pipefail

module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
source "$PROJECT/venv/bin/activate"

python "$PROJECT/code/export_rpca_events.py" \
  --input-dir "$PROJECT/results/rpca_july" \
  --out "$PROJECT/results/rpca_july/RPCA_ALL_EVENTS.csv" \
  --max-gap-minutes 15
