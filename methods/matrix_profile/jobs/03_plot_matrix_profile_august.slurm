#!/usr/bin/env bash
#SBATCH --job-name=mp_aug_plot
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aug_plot_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aug_plot_%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_august_fixed_july
DETECTION_END=${DETECTION_END:-2026-08-31}

source "${JOB_ROOT}/load_python.sh"
source "${PROJECT_ROOT}/venv_mp/bin/activate"
export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"
cd "${APP_ROOT}"

python "${APP_ROOT}/plot_multiscale_selected.py" \
    --events "${RESULT_ROOT}/aggregate/ALL_EVENTS.csv" \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --output-dir "${RESULT_ROOT}/selected_plots_august" \
    --start-date 2026-08-01 \
    --end-date "${DETECTION_END}" \
    --max-plots 100 \
    --per-group 3
