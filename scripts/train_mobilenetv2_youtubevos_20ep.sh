#!/bin/bash
#SBATCH --job-name=mobilenetv2_ytvos_20ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/mobilenetv2_ytvos_20ep_%j.out
#SBATCH --error=logs/slurm/mobilenetv2_ytvos_20ep_%j.err

# ============================================================
# Phase: Train MobileNetV2 encoder backbones on YouTube-VOS
#
# Created as a variant of train_mobilenetv2_20ep.sh for the
# YouTube-VOS 2019 dataset using 480x480 resolution.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: MobileNetV2 YouTube-VOS Training"
echo "  Epochs    : 20"
echo "  Target Res: 480x480"
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
    datamodule=youtubevos \
    ++model.target_size=480 \
    ++model.max_epochs=20 \
    ++trainer.max_epochs=20 \
    ++trainer.precision="16-mixed" \
    ++logger.name="mobilenetv2_ytvos_20ep" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    "$@"

echo ""
echo "Training finished at: $(date)"
