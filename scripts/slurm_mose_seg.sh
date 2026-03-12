#!/bin/bash
#SBATCH --job-name=v_kamba_mose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/mose_%j.out
#SBATCH --error=logs/slurm/mose_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "Current Directory: $(pwd)"

echo "=========================================="
echo "  Video Kamba: MOSE 2023 Segmentation"
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

# Run training with MOSE datamodule.
# MOSE has 41.5% object disappearance rate; the obj_present mask in
# HybridVOSLoss handles absent objects correctly.
# Pass any additional overrides as arguments, e.g.:
#   sbatch slurm_mose_seg.sh trainer.fast_dev_run=true
python train.py \
    datamodule=mose \
    model.num_seg_classes=11 \
    ++datamodule.num_workers=8 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    "$@"

echo ""
echo "Job complete at $(date)."
