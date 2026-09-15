#!/usr/bin/env bash
#SBATCH --job-name=mp_roll30
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-99%20
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/run_%A_%a.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/run_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_rolling30
PART=$(printf 'part-%03d' "${SLURM_ARRAY_TASK_ID}")

source "${JOB_ROOT}/load_python.sh"
source "${PROJECT_ROOT}/venv_mp/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${APP_ROOT}"

python detect_multiscale_anomalies.py \
    --input-layout merged \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --manifest "${PROJECT_ROOT}/data/processed/meter_csvs/_manifest.csv" \
    --repository "${APP_ROOT}/meter_repository.json" \
    --history-mode rolling \
    --rolling-history-days 30 \
    --detection-start 2025-11-24 \
    --detection-end 2026-07-31 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 100 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-mode shard \
    --no-plots \
    --output-dir "${RESULT_ROOT}/shards/${PART}"
