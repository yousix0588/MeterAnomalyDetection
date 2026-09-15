#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RUNTIME_ROOT=${PROJECT_ROOT}/runs/matrix_profile
LOG_ROOT=${RUNTIME_ROOT}/logs
RESULT_ROOT=${RUNTIME_ROOT}/results/matrix_profile_august_fixed_july
MANIFEST=${PROJECT_ROOT}/data/processed/meter_csvs/_manifest.csv
DETECTION_END=$(awk -F, 'NR > 1 { d=substr($7,1,10); if (d >= "2026-08-01" && d <= "2026-08-31" && d > latest) latest=d } END { print latest }' "${MANIFEST}")

if [[ -z "${DETECTION_END}" ]]; then
    printf 'Refusing to submit: manifest contains no August 2026 data.\n' >&2
    exit 1
fi
MANIFEST_CHANNELS=$(awk 'END { print NR - 1 }' "${MANIFEST}")
END_DATE_CHANNELS=$(awk -F, -v end_date="${DETECTION_END}" 'NR > 1 && substr($7,1,10) >= end_date { count++ } END { print count + 0 }' "${MANIFEST}")

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}/shards"
cd "${JOB_ROOT}"

if [[ -n "$(find "${RESULT_ROOT}/shards" -mindepth 1 -print -quit)" ]]; then
    printf 'Refusing to overwrite existing August shard outputs under %s\n' "${RESULT_ROOT}/shards" >&2
    exit 1
fi
if [[ -d "${RESULT_ROOT}/aggregate" || -d "${RESULT_ROOT}/selected_plots_august" ]]; then
    printf 'Refusing to overwrite existing August aggregate or plot outputs under %s\n' "${RESULT_ROOT}" >&2
    exit 1
fi

printf 'Manifest channels: %s; channels reaching %s: %s\n' \
    "${MANIFEST_CHANNELS}" "${DETECTION_END}" "${END_DATE_CHANNELS}"
printf 'Submitting August detection range 2026-08-01 through %s\n' "${DETECTION_END}"
ARRAY_JOB=$(sbatch --parsable --export=ALL,DETECTION_END="${DETECTION_END}" 01_run_matrix_profile_august_array.slurm)
AGGREGATE_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" 02_aggregate_matrix_profile_august.slurm)
PLOT_JOB=$(sbatch --parsable --dependency="afterok:${AGGREGATE_JOB}" --export=ALL,DETECTION_END="${DETECTION_END}" 03_plot_matrix_profile_august.slurm)

printf 'array_job=%s\naggregate_job=%s\nplot_job=%s\n' \
    "${ARRAY_JOB}" "${AGGREGATE_JOB}" "${PLOT_JOB}"
