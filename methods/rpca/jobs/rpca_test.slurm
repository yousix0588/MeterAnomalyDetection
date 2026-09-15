#!/bin/bash
#SBATCH --job-name=rpca_test
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/rpca/logs/rpca_test_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/rpca/logs/rpca_test_%j.err

set -euo pipefail
module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14
METER_CHANNEL=DD03710150192_4

mkdir -p "$PROJECT/runs/rpca/test" "$PROJECT/runs/rpca/logs"
source "$PROJECT/venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python "$PROJECT/methods/rpca/src/rpca_spartan_train_test.py" \
  --csv "$PROJECT/data/processed/meter_csvs/${METER_CHANNEL}.csv" \
  --outdir "$PROJECT/runs/rpca/test" \
  --test-start 2026-07-01 \
  --test-end 2026-08-01 \
  --threshold-quantile 0.99 \
  --power-cutoff 0.0
