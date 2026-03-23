#!/bin/bash
#SBATCH --job-name=pretrain_coco_sota
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/pretrain_coco_sota_%j.out
#SBATCH --error=logs/slurm/pretrain_coco_sota_%j.err

# COCO static-image pre-training for vision_mamba_tiny_sota_light (DiagonalKANSSMCore).
#
# This is PHASE 1 of the fast validation protocol:
#   Phase 1 (this job) : COCO pretrain from scratch         ~15h / 20 epochs
#   Phase 2            : DAVIS fast validate (gate decision) ~16h / 100 epochs
#   Phase 3 (if gate)  : Full YTB+DAV fine-tune             ~71h / 50 epochs
#
# Why COCO (not ADE20K):
#   VOS requires per-INSTANCE mask understanding, not semantic class labels.
#   COCO provides polygon/RLE instance masks — directly aligned with VOS.
#   Reference: Cutie (CVPR 2024) uses COCO → YTB+DAV, no BL30K needed.
#
# Architecture: DiagonalKANSSMCore (log_A, D_skip, delta_proj, A_modulator).
# ALL existing checkpoints use IntricateKANSSMCore (dense A [N×N]) — incompatible.
# This script trains from scratch with the new architecture.
#
# Hyper-parameters:
#   learning_rate=1e-4  (higher than fine-tune 4e-5 — no pretrained init)
#   limit_train_batches=3000  (COCO ~118k images; cap epoch at 3000 steps)
#   accumulate_grad_batches=2  (effective batch = 4 × 2 = 8)
#   20 epochs × 3000 steps / ~4500 steps·h⁻¹ ≈ 13–15h total
#
# Output checkpoint: checkpoints/best_coco_sota_light.ckpt
# Usage: export COMET_API_KEY=<key> && sbatch scripts/pretrain_coco_sota_light.sh
#   OR:  add COMET_API_KEY to ${PROJECT_ROOT}/.env (preferred)

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm checkpoints

# Load secrets from .env (never committed to git)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

COCO_DATA_DIR="${PROJECT_ROOT}/data/coco"
if [ ! -f "${COCO_DATA_DIR}/annotations/instances_train2017.json" ]; then
    echo "Error: COCO annotations not found at ${COCO_DATA_DIR}/annotations/instances_train2017.json"
    exit 1
fi

echo "=========================================="
echo "  Video Kamba: COCO Pre-training (from scratch)"
echo "  Model     : vision_mamba_tiny_sota_light"
echo "  SSM Core  : DiagonalKANSSMCore (new arch)"
echo "  Datamodule: coco_pretrain (instance masks)"
echo "  Epochs    : 20 / limit 3000 steps/epoch"
echo "  Eff. batch: 8 (batch_size=4 × accum=2)"
echo "  LR        : 1e-4 (from scratch)"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

python train.py \
    model=vision_mamba_tiny_sota_light \
    datamodule=coco_pretrain \
    ++model.learning_rate=1e-4 \
    ++model.max_epochs=20 \
    ++datamodule.data_dir="${COCO_DATA_DIR}" \
    ++trainer.max_epochs=20 \
    ++trainer.accumulate_grad_batches=2 \
    ++trainer.limit_train_batches=3000 \
    ++logger.name="pretrain_coco_sota_light_v1" \
    ++logger.tags="[diagonal-kan,coco-pretrain,from-scratch,lightweight]" \
    ++callbacks.model_checkpoint.monitor=val_loss \
    ++callbacks.model_checkpoint.mode=min \
    ++callbacks.model_checkpoint.filename=best_coco_sota_light

echo ""
echo "=========================================="
echo "  COCO pre-training finished at $(date)"
echo "  Checkpoint: checkpoints/best_coco_sota_light.ckpt"
echo "  Next: sbatch scripts/ft_davis_fast_validate.sh"
echo "=========================================="
