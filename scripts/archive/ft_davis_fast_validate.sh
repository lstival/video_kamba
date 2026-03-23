#!/bin/bash
#SBATCH --job-name=davis_fast_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/davis_fast_val_%j.out
#SBATCH --error=logs/slurm/davis_fast_val_%j.err

# PHASE 2 — Fast DAVIS fine-tune for architecture gate decision.
#
# Protocol:
#   Phase 1: COCO pretrain (pretrain_coco_sota_light.sh)       ~15h DONE
#   Phase 2: DAVIS fast validate (THIS JOB)                    ~16h
#   Phase 3: Full YTB+DAV (ft_coco_ytb_dav_50ep.sh)           ~71h
#
# Goal: get a reliable val_J_and_F signal in ~16h before committing
# to the 71h full fine-tune. DAVIS has only 301 training clips vs.
# 333k in YouTube-VOS — no WeightedRandomSampler overhead.
#
#   limit_train_batches=500 → ~10-12 min/epoch
#   100 epochs × 12 min ≈ 20h max (24h wall time has generous margin)
#
# GATE DECISION (check Comet after epoch 50):
#   val_J_and_F > 0.55 → architecture validated → sbatch ft_coco_ytb_dav_50ep.sh
#   val_J_and_F < 0.55 → investigate SSM convergence before committing
#
# Reference: MobileVOS (CVPR 2023) achieves ~78% J&F with <5M params on DAVIS.
# Reaching 55%+ at epoch 50 (DAVIS-only, limited batches) is a realistic gate.
#
# Usage: sbatch scripts/ft_davis_fast_validate.sh
#   (COMET_API_KEY loaded from ${PROJECT_ROOT}/.env)

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

CKPT="${PROJECT_ROOT}/checkpoints/best_coco_sota_light.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "Error: COCO pretrain checkpoint not found at $CKPT"
    echo "Run pretrain_coco_sota_light.sh first (Phase 1)."
    exit 1
fi

echo "=========================================="
echo "  Video Kamba: DAVIS Fast Validation (Phase 2)"
echo "  Checkpoint: $CKPT"
echo "  Model     : vision_mamba_tiny_sota_light"
echo "  Datamodule: davis (301 train clips)"
echo "  Epochs    : 100 / limit 500 steps/epoch"
echo "  ~10-12 min/epoch → ~16h total"
echo "  GATE      : val_J_and_F > 0.55 @ epoch 50"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

python train.py \
    model=vision_mamba_tiny_sota_light \
    datamodule=davis \
    +pretrained_weights="$CKPT" \
    ++trainer.max_epochs=100 \
    ++trainer.accumulate_grad_batches=2 \
    ++trainer.limit_train_batches=500 \
    ++logger.name="ft_davis_fast_validate_v1" \
    ++logger.tags="[diagonal-kan,davis-only,fast-validate,gate-decision,coco-init]" \
    ++callbacks.model_checkpoint.monitor=val_J_and_F \
    ++callbacks.model_checkpoint.mode=max \
    ++callbacks.model_checkpoint.filename=best_davis_sota_light

echo ""
echo "=========================================="
echo "  DAVIS fast validation finished at $(date)"
echo "  Check Comet: ft_davis_fast_validate_v1"
echo "  If val_J_and_F > 0.55 @ epoch 50:"
echo "    → sbatch scripts/ft_coco_ytb_dav_50ep.sh"
echo "=========================================="
