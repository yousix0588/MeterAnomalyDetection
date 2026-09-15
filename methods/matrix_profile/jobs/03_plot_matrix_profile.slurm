#!/usr/bin/env bash
#SBATCH --job-name=mp_plot
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/plot_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/plot_%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_rolling30

source "${JOB_ROOT}/load_python.sh"
source "${PROJECT_ROOT}/venv_mp/bin/activate"
cd "${APP_ROOT}"

python plot_multiscale_selected.py \
    --events "${RESULT_ROOT}/aggregate/ALL_EVENTS.csv" \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --output-dir "${RESULT_ROOT}/selected_plots" \
    --per-group 3
