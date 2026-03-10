#!/bin/bash
#SBATCH --job-name=pretrain_coco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/pretrain_coco_%j.out
#SBATCH --error=logs/slurm/pretrain_coco_%j.err

# ============================================================
# COCO Static-Image Pre-training (AOT-style)
#
# Generates synthetic VOS clips from COCO 2017 images via
# HuggingFace datasets (detection-datasets/coco).
#
# Usage:
#   sbatch scripts/pretrain_coco.sh
#
# After completion, fine-tune on YouTube-VOS:
#   sbatch scripts/finetune_dinov2_youtubevos.sh \
#       ++checkpoint=/path/to/checkpoints/best_coco.ckpt
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: COCO Static Pre-training"
echo "  Epochs  : 15"
echo "  LR      : 2e-4"
echo "  img_size: 448"
echo "  SSR     : 0.1  (near teacher-forcing — no real motion)"
echo "  Job ID  : ${SLURM_JOB_ID:-manual}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
# ── Avoid cross-device link errors and /tmp overflows ────────────────────────
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm checkpoints

python train.py \
    model=default \
    datamodule=coco_pretrain \
    ++model.encoder_type=dino \
    ++model.target_size=448 \
    ++model.max_epochs=15 \
    ++model.learning_rate=2e-4 \
    ++model.scheduled_sampling_rate=0.1 \
    ++model.use_kan_key_adapter=true \
    ++trainer.max_epochs=15 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="coco_pretrain_15ep" \
    ++callbacks.monitor=val_loss \
    ++callbacks.mode=min \
    "$@"

echo ""
echo "COCO pre-training finished at: $(date)"
echo "Next step: sbatch scripts/finetune_dinov2_youtubevos.sh ++checkpoint=<best_ckpt>"
