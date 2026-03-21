#!/bin/bash
#SBATCH --job-name=mv2_phase1b_bl30k
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm/mv2_phase1b_bl30k_%j.out
#SBATCH --error=logs/slurm/mv2_phase1b_bl30k_%j.err

# ============================================================
# Phase 1b — BL30K Synthetic VOS Pre-training (MV2 + KAN-SSM)
#
# Initialise from Phase 1 COCO checkpoint.
# Same backbone freeze (mv2_freeze_at=2) as Phase 1.
#
# Purpose: bridge COCO kinematic pseudo-video → real VOS domain.
# Enables fair comparison with TrickVOS, WarpFormer-S, AOTT
# (all use BL30K pre-training).
#
# Data: BL30K (~30K synthetic video sequences, binary masks)
#       Expected at: data/BL30K/
#
# Budget: 30 epochs × 3000 steps × ~12s/step ≈ 30h
#
# Success criterion: val_J_and_F > 0.45 at epoch 15
# Gate: if val_J_and_F < 0.35 at epoch 10 — diagnose Phase 1 ckpt
#
# Output: checkpoints/best_mv2_phase1b_bl30k.ckpt
# Next:   sbatch scripts/train_mv2_phase2_davis.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase1_coco.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Error: Phase 1 checkpoint not found at $CKPT"
    echo "Run sbatch scripts/train_mv2_phase1_coco.sh first."
    exit 1
fi

BL30K_DIR="${PROJECT_ROOT}/data/BL30K"
if [ ! -d "$BL30K_DIR" ]; then
    echo "Error: BL30K data not found at $BL30K_DIR"
    echo "Download from https://henghuiding.github.io/MIVOS/ (BL30K section)"
    exit 1
fi

echo "=========================================="
echo "  MV2 + KAN-SSM: Phase 1b — BL30K Pretrain"
echo "  Checkpoint : $CKPT"
echo "  Data       : $BL30K_DIR"
echo "  Job ID     : ${SLURM_JOB_ID:-manual}"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "  Gate       : val_J_and_F > 0.45 @ epoch 15"
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

export TRITON_CACHE_AUTOTUNING=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-manual}"
mkdir -p "$TRITON_CACHE_DIR"

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
    datamodule=bl30k_pretrain \
    +experiment=mv2_phase1b_bl30k \
    +pretrained_weights="$CKPT" \
    ++logger.name="mv2_phase1b_bl30k" \
    ++logger.tags="[mobilenetv2,kan-ssm,bl30k-pretrain,phase1b,synthetic-vos]" \
    ++callbacks.model_checkpoint.dirpath="checkpoints" \
    ++callbacks.model_checkpoint.filename="best_mv2_phase1b_bl30k" \
    ++trainer.num_sanity_val_steps=2 \
    "$@"

echo ""
echo "Phase 1b finished at: $(date)"
echo "Checkpoint: checkpoints/best_mv2_phase1b_bl30k.ckpt"
echo ""
echo "Gate check: verify val_J_and_F > 0.45 at epoch 15 in Comet."
echo "  If gate passed: sbatch scripts/train_mv2_phase2_davis.sh"
echo "  Phase 2 will auto-load best_mv2_phase1b_bl30k.ckpt"
