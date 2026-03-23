#!/bin/bash
#SBATCH --job-name=v_kamba_hmdb51
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/hmdb51_%j.out
#SBATCH --error=logs/slurm/hmdb51_%j.err

set -euo pipefail

# Determine project root
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "Current Directory: $(pwd)"
echo "Contents: $(ls -F)"

echo "=========================================="
echo "  Video Kamba: HMDB51 Classification"
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

# Run training
export PYTHONPATH=.
python train.py datamodule=hmdb51 "$@"

echo ""
echo "Job complete at $(date)."
