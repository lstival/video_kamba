#!/bin/bash
#SBATCH --job-name=v_kamba_ytbvos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/youtubevos_%j.out
#SBATCH --error=logs/slurm/youtubevos_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "Current Directory: $(pwd)"

echo "=========================================="
echo "  Video Kamba: YouTube-VOS 2019 Segmentation"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

# Activate the jackan_gpu conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jackan_gpu

export PYTHONPATH=.

# Create slurm log directory if it doesn't exist
mkdir -p logs/slurm

# Run training with YouTube-VOS datamodule.
# Pass any additional overrides as arguments, e.g.:
#   sbatch slurm_youtubevos_seg.sh trainer.fast_dev_run=true
#   sbatch slurm_youtubevos_seg.sh trainer.max_epochs=30
python train.py \
    datamodule=youtubevos \
    model.num_seg_classes=11 \
    "$@"

echo ""
echo "Job complete at $(date)."
