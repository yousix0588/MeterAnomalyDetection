#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
RUNTIME_ROOT=${PROJECT_ROOT}/runs/matrix_profile
LOG_ROOT=${RUNTIME_ROOT}/logs
RESULT_ROOT=${RUNTIME_ROOT}/results/matrix_profile_rolling30

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}/shards"
cd "${JOB_ROOT}"

if [[ -n "$(find "${RESULT_ROOT}/shards" -mindepth 1 -print -quit)" ]]; then
    printf 'Refusing to overwrite existing shard outputs under %s\n' "${RESULT_ROOT}/shards" >&2
    exit 1
fi
if [[ -d "${RESULT_ROOT}/aggregate" || -d "${RESULT_ROOT}/selected_plots" ]]; then
    printf 'Refusing to overwrite existing aggregate or plot outputs under %s\n' "${RESULT_ROOT}" >&2
    exit 1
fi

ARRAY_JOB=$(sbatch --parsable 01_run_matrix_profile_array.slurm)
AGGREGATE_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" 02_aggregate_matrix_profile.slurm)
PLOT_JOB=$(sbatch --parsable --dependency="afterok:${AGGREGATE_JOB}" 03_plot_matrix_profile.slurm)

printf 'array_job=%s\naggregate_job=%s\nplot_job=%s\n' \
    "${ARRAY_JOB}" "${AGGREGATE_JOB}" "${PLOT_JOB}"
