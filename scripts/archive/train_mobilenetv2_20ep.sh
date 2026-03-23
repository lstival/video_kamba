#!/bin/bash
#SBATCH --job-name=mobilenetv2_20ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/mobilenetv2_20ep_%j.out
#SBATCH --error=logs/slurm/mobilenetv2_20ep_%j.err

# ============================================================
# Phase: Train MobileNetV2 encoder backbones
#
# Created as a successor to train_propagation_v2.sh but using
# the mobileNetV2 architecture.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: MobileNetV2 Training"
echo "  Epochs    : 20"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.

# Ensure logs directory exists
mkdir -p logs/slurm

python train.py \
    model=mobilenetv2 \
    datamodule=davis \
    ++trainer.max_epochs=20 \
    ++trainer.precision="16-mixed" \
    ++logger.name="mobilenetv2_20ep" \
    "$@"

echo ""
echo "Training finished at: $(date)"
