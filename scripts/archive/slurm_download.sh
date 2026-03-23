#!/bin/bash
#SBATCH --job-name=v_kamba_download
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/download_%j.out
#SBATCH --error=logs/slurm/download_%j.err

set -euo pipefail

# Determine project root
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Dataset Download"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found."
    exit 1
fi

export PYTHONPATH=.

# Run download script
python download_datasets.py

echo ""
echo "Job complete at $(date)."
