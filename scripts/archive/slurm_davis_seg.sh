#!/bin/bash
#SBATCH --job-name=v_kamba_davis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/davis_%j.out
#SBATCH --error=logs/slurm/davis_%j.err

set -euo pipefail

# Determine project root
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "Current Directory: $(pwd)"
echo "Contents: $(ls -F)"

echo "=========================================="
echo "  Video Kamba: DAVIS Segmentation"
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
# Pass any additional arguments from command line (like --fast_dev_run=True)
export PYTHONPATH=.
python train.py datamodule=davis \
    ++datamodule.num_workers=8 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    "$@"

echo ""
echo "Job complete at $(date)."
