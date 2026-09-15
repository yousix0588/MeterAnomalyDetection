#!/usr/bin/env bash
#SBATCH --job-name=mp_preflight
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/preflight_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/preflight_%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_rolling30/preflight/${SLURM_JOB_ID}

source "${JOB_ROOT}/load_python.sh"
source "${PROJECT_ROOT}/venv_mp/bin/activate"
cd "${APP_ROOT}"
mkdir -p "${RESULT_ROOT}"

/usr/bin/time -v python detect_multiscale_anomalies.py \
    --input-layout merged \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --manifest "${PROJECT_ROOT}/data/processed/meter_csvs/_manifest.csv" \
    --repository "${APP_ROOT}/meter_repository.json" \
    --history-mode rolling \
    --rolling-history-days 30 \
    --detection-start 2025-11-24 \
    --detection-end 2026-07-31 \
    --channel-keys DD03710150192_4 \
    --workers 1 \
    --output-mode shard \
    --no-plots \
    --output-dir "${RESULT_ROOT}/part-000"

python aggregate_multiscale_shards.py \
    --shards-dir "${RESULT_ROOT}" \
    --output-dir "${RESULT_ROOT}/aggregate" \
    --expected-shards 1

python plot_multiscale_selected.py \
    --events "${RESULT_ROOT}/aggregate/ALL_EVENTS.csv" \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --output-dir "${RESULT_ROOT}/selected_plots" \
    --per-group 3
