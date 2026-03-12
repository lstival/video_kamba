#!/bin/bash
#SBATCH --job-name=verify_unseen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/verify_unseen_%j.out
#SBATCH --error=logs/slurm/verify_unseen_%j.err

set -euo pipefail

# Find project root (assume current folder or one up)
PROJECT_ROOT=$(pwd)
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  YouTube-VOS Unseen Split Verification"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found. Running with system python."
fi

export PYTHONPATH=.

# Run verification script
python scripts/verify_yt_unseen.py

echo "=========================================="
echo "  Finish time: $(date)"
echo "=========================================="
