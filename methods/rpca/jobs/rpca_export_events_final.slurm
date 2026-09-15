#!/bin/bash
#SBATCH --job-name=rpca_final
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/rpca/logs/rpca_final_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/rpca/logs/rpca_final_%j.err

set -euo pipefail

module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
source "$PROJECT/venv/bin/activate"

python "$PROJECT/methods/rpca/src/export_rpca_events_final.py" \
  --input-dir "$PROJECT/runs/rpca/july" \
  --out "$PROJECT/runs/rpca/july/RPCA_ALL_METERS_anomalies.csv" \
  --min-duration-minutes 15
