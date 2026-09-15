#!/usr/bin/env bash
#SBATCH --job-name=mp_aug_preflight
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aug_preflight_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/matrix_profile/logs/aug_preflight_%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
MANIFEST=${PROJECT_ROOT}/data/processed/meter_csvs/_manifest.csv
RESULT_ROOT=${PROJECT_ROOT}/runs/matrix_profile/results/matrix_profile_august_fixed_july/preflight/${SLURM_JOB_ID}
DETECTION_END=${DETECTION_END:-$(awk -F, 'NR > 1 { d=substr($7,1,10); if (d >= "2026-08-01" && d <= "2026-08-31" && d > latest) latest=d } END { print latest }' "${MANIFEST}")}

if [[ -z "${DETECTION_END}" ]]; then
    printf 'Manifest contains no August 2026 data: %s\n' "${MANIFEST}" >&2
    exit 1
fi
PREFLIGHT_CHANNEL=${PREFLIGHT_CHANNEL:-$(awk -F, -v end_date="${DETECTION_END}" 'NR > 1 && substr($7,1,10) >= end_date { print $1; exit }' "${MANIFEST}")}
if [[ -z "${PREFLIGHT_CHANNEL}" ]]; then
    printf 'No channel reaches August detection end %s.\n' "${DETECTION_END}" >&2
    exit 1
fi

source "${JOB_ROOT}/load_python.sh"
source "${PROJECT_ROOT}/venv_mp/bin/activate"
export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"
cd "${APP_ROOT}"
mkdir -p "${RESULT_ROOT}"
printf 'August detection range: 2026-08-01 through %s\n' "${DETECTION_END}"
printf 'Preflight channel: %s\n' "${PREFLIGHT_CHANNEL}"

/usr/bin/time -v python detect_multiscale_anomalies.py \
    --input-layout merged \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --manifest "${MANIFEST}" \
    --repository "${APP_ROOT}/meter_repository.json" \
    --history-mode fixed \
    --detection-start 2026-08-01 \
    --detection-end "${DETECTION_END}" \
    --channel-keys "${PREFLIGHT_CHANNEL}" \
    --workers 1 \
    --output-mode shard \
    --no-plots \
    --output-dir "${RESULT_ROOT}/part-000"

python aggregate_multiscale_shards.py \
    --shards-dir "${RESULT_ROOT}" \
    --output-dir "${RESULT_ROOT}/aggregate" \
    --expected-shards 1

python "${APP_ROOT}/plot_multiscale_selected.py" \
    --events "${RESULT_ROOT}/aggregate/ALL_EVENTS.csv" \
    --data-dir "${PROJECT_ROOT}/data/processed/meter_csvs" \
    --output-dir "${RESULT_ROOT}/selected_plots" \
    --start-date 2026-08-01 \
    --end-date "${DETECTION_END}" \
    --max-plots 100 \
    --per-group 3
