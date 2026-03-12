#!/bin/bash
#SBATCH --job-name=vimtiny_ytvos_20ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vimtiny_ytvos_20ep_%j.out
#SBATCH --error=logs/slurm/vimtiny_ytvos_20ep_%j.err

# ============================================================
# Train lightweight Vision-Mamba + KAN on YouTube-VOS 2019.
#
# Highlights:
#   - trainable encoder_type=vision_mamba_tiny
#   - lightweight decoder widths (160/96/48)
#   - KAN-guided spatial Vision-Mamba + KAN temporal SSM
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Vision-Mamba Tiny (YouTube-VOS)"
echo "  Encoder   : vision_mamba_tiny (trainable)"
echo "  Epochs    : 20"
echo "  Target Res: 480x480"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
mkdir -p logs/slurm

python train.py \
    model=vision_mamba_tiny \
    datamodule=youtubevos \
    ++model.target_size=480 \
    ++datamodule.img_size=480 \
    ++datamodule.num_workers=8 \
    ++trainer.max_epochs=20 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="vimtiny_ytvos_20ep" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    "$@"

echo ""
echo "Training finished at: $(date)"
