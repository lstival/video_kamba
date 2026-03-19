#!/bin/bash
#SBATCH --job-name=analyse_grad
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/analyse_grad_%j.out
#SBATCH --error=logs/slurm/analyse_grad_%j.err

set -euo pipefail
cd /home/WUR/stiva001/WUR/video_kamba
source venv/bin/activate
export PYTHONPATH=.

NUM_EPOCHS=${1:-4}

python scripts/analyse_gradient_explosion.py \
    +num_epochs="$NUM_EPOCHS" \
    +artifacts_dir=artifacts/gradient_norms
