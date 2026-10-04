#!/bin/bash
#SBATCH --job-name=vae_aggregate
#SBATCH --account=punim1257
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=/data/projects/punim1257/Group14/runs/lstm_vae/logs/aggregate_%j.out
#SBATCH --error=/data/projects/punim1257/Group14/runs/lstm_vae/logs/aggregate_%j.err

PROJECT_DIR=/data/projects/punim1257/Group14
CODE_DIR=$PROJECT_DIR/methods/lstm_vae/src
OUT_DIR=$PROJECT_DIR/runs/lstm_vae/august

mkdir -p "$PROJECT_DIR/runs/lstm_vae/logs"

module purge
module load foss/2022a
module load Python/3.10.4
source "$PROJECT_DIR/venv/bin/activate"

cd "$CODE_DIR"
python aggregate_array_results.py --out_dir "$OUT_DIR"

echo "Final consolidated anomaly file:"
wc -l "$OUT_DIR/ALL_METERS_anomalies.csv"
