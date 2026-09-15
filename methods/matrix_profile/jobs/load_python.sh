#!/usr/bin/env bash
set -euo pipefail

module purge
module load "${TOOLCHAIN_MODULE:-foss/2022a}"
module load "${PYTHON_MODULE:-Python/3.10.4}"
