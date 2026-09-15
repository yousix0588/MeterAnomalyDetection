#!/usr/bin/env bash
#SBATCH --job-name=mp_aug30
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-99%20
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aug_run_%A_%a.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aug_run_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_august_fixed_july
DETECTION_END=${DETECTION_END:-2026-08-31}
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
    --history-mode fixed \
    --detection-start 2026-08-01 \
    --detection-end "${DETECTION_END}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 100 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-mode shard \
    --no-plots \
    --output-dir "${RESULT_ROOT}/shards/${PART}"
