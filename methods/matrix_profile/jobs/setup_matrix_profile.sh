#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data/projects/punim1257/Group14
APP_ROOT=${PROJECT_ROOT}/methods/matrix_profile/src
JOB_ROOT=${PROJECT_ROOT}/methods/matrix_profile/jobs
VENV_ROOT=${PROJECT_ROOT}/venv_mp
RUNTIME_ROOT=${PROJECT_ROOT}/runs/matrix_profile

source "${JOB_ROOT}/load_python.sh"
if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
    python -m venv "${VENV_ROOT}"
fi
source "${VENV_ROOT}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${JOB_ROOT}/requirements-multiscale.txt"
mkdir -p "${RUNTIME_ROOT}/logs"
mkdir -p "${RUNTIME_ROOT}/results/matrix_profile_rolling30/shards"

python -c "import numpy, pandas, matplotlib, ruptures; print('Matrix Profile environment ready')"
