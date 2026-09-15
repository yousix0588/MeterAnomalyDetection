#!/usr/bin/env bash
#SBATCH --job-name=mp_aggregate
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aggregate_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aggregate_%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_rolling30

source "${JOB_ROOT}/load_python.sh"
source "${PROJECT_ROOT}/venv_mp/bin/activate"
cd "${APP_ROOT}"

python aggregate_multiscale_shards.py \
    --shards-dir "${RESULT_ROOT}/shards" \
    --output-dir "${RESULT_ROOT}/aggregate" \
    --expected-shards 100
