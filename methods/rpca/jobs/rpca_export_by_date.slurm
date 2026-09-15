#!/bin/bash
#SBATCH --job-name=rpca_bydate
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/rpca/logs/rpca_bydate_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/rpca/logs/rpca_bydate_%j.err

set -euo pipefail

module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
source "$PROJECT/venv/bin/activate"

python "$PROJECT/methods/rpca/src/export_rpca_anomalies_by_date.py" \
  --input-dir "$PROJECT/runs/rpca/july" \
  --out "$PROJECT/runs/rpca/july/RPCA_ALL_ANOMALIES_BY_DATE.csv"
