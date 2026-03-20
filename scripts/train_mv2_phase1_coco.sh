#!/bin/bash
#SBATCH --job-name=mv2_phase1_coco
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/mv2_phase1_coco_%j.out
#SBATCH --error=logs/slurm/mv2_phase1_coco_%j.err

# ============================================================
# Phase 1 — COCO Pre-training (MobileNetV2 + KAN-SSM Temporal)
#
# Encoder: MobileNetV2 (ImageNet-1K pretrained, torchvision)
#          Stages 0+1 frozen (stride-4/8); stage 2+proj trainable
# Temporal: KangaSSM (DiagonalKANSSMCore, ssm_layers=2)
#
# Data: COCO 2017 (~118k images) converted to pseudo-video clips
#       via coherent affine kinematic prior (6 frames/clip)
#
# Budget: 20 epochs × 3000 steps × ~12s/step ≈ 20h
#
# Success criterion: val_loss < 0.5 at epoch 5
# Output: checkpoints/best_mv2_phase1_coco.ckpt
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  MV2 + KAN-SSM: Phase 1 — COCO Pretrain"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm checkpoints

# Mamba SSM uses Triton CUDA kernels that autotune on first run.
# TRITON_CACHE_AUTOTUNING=1 persists tuned configs to disk so
# subsequent jobs skip the 5-10 min recompilation step.
export TRITON_CACHE_AUTOTUNING=1
# Use /tmp to avoid home-directory quota limits on kernel cache writes.
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-manual}"
mkdir -p "$TRITON_CACHE_DIR"

# Reduce CUDA allocator fragmentation (recommended by OOM error message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

python train.py \
    model=mobilenetv2_kan_temporal \
    datamodule=coco_pretrain \
    +experiment=mv2_phase1_coco \
    ++logger.name="mv2_phase1_coco" \
    ++logger.tags="[mobilenetv2,kan-ssm,coco-pretrain,phase1,imagenet-init]" \
    ++callbacks.model_checkpoint.dirpath="checkpoints" \
    ++trainer.num_sanity_val_steps=0 \
    "$@"

echo ""
echo "Phase 1 finished at: $(date)"
echo "Checkpoint: checkpoints/best_mv2_phase1_coco.ckpt"
