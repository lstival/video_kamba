#!/bin/bash
#SBATCH --job-name=dinov2_ytvos_100ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/dinov2_ytvos_100ep_%j.out
#SBATCH --error=logs/slurm/dinov2_ytvos_100ep_%j.err

# ============================================================
# Phase: Train DINOv2 ViT-B/14 (frozen) on YouTube-VOS 2019
#
# Differences vs mobilenetv2_ytvos_20ep:
#   - encoder_type : dino  (frozen ViT-B/14, 768-dim)
#   - target_size  : 448   (14×14 token grid, DINOv2 native patch=14)
#   - dim_in       : 768   (default.yaml default — no override needed)
#   - backbone_lr  : lr * 0.01  (frozen — effectively 0, set in configure_optimizers)
#   - scheduled_sampling_rate: 0.3  (same exposure-bias fix as MobileNetV2 run)
#   - mem          : 48G   (ViT-B/14 activations larger than MobileNetV2)
#   - time         : 120h  (ViT forward pass slower per step)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: DINOv2 YouTube-VOS Training (100ep)"
echo "  Encoder   : DINOv2 ViT-B/14 (frozen)"
echo "  Epochs    : 100"
echo "  Target Res: 448x448"
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
    model=default \
    datamodule=youtubevos \
    ++model.encoder_type=dino \
    ++model.target_size=448 \
    ++model.scheduled_sampling_rate=0.3 \
    ++datamodule.img_size=448 \
    ++datamodule.num_workers=8 \
    ++trainer.max_epochs=100 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="dinov2_ytvos_100ep" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    "$@"

echo ""
echo "Training finished at: $(date)"
