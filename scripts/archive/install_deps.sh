#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/install_deps_%j.out
#SBATCH --error=logs/slurm/install_deps_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Installing datasets and pycocotools..."
    pip install datasets pycocotools
    echo "Installation complete."
else
    echo "Error: venv not found." && exit 1
fi
