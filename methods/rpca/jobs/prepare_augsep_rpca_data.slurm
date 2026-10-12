#!/bin/bash
#SBATCH --job-name=prep_augsep
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/punim1257/Group14/results/prep_augsep_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/results/prep_augsep_%j.err

set -euo pipefail

module purge
module load foss/2022a
module load Python/3.10.4

PROJECT=/data/projects/punim1257/Group14

source "$PROJECT/venv/bin/activate"

cd "$PROJECT"

python code/prepare_augsep_rpca_data.py
