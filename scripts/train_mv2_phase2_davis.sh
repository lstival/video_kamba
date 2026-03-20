#!/bin/bash
#SBATCH --job-name=mv2_phase2_davis
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm/mv2_phase2_davis_%j.out
#SBATCH --error=logs/slurm/mv2_phase2_davis_%j.err

# ============================================================
# Phase 2 — DAVIS Fine-tuning (MobileNetV2 + KAN-SSM Temporal)
#
# Initialise from Phase 1 checkpoint (best_mv2_phase1_coco.ckpt).
# All MobileNetV2 stages unfrozen with backbone_lr_multiplier=0.05
# (effective LR 1.5e-6 — preserves ImageNet features).
#
# Data: DAVIS 2017 semi-supervised (60 train / 30 val sequences)
# Budget: 100 epochs × 500 steps × ~12s/step ≈ 17h
#
# Gate decision @ epoch 50 (check Comet: mv2_phase2_davis):
#   val_J_and_F > 0.55 → proceed to Phase 3
#   val_J_and_F < 0.55 → diagnose (check Phase 1 ckpt quality)
#
# Output: checkpoints/best_mv2_phase2_davis.ckpt
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

echo "=========================================="
echo "  MV2 + KAN-SSM: Phase 2 — DAVIS Fine-tune"
echo "  Checkpoint : $CKPT"
echo "  Job ID     : ${SLURM_JOB_ID:-manual}"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "  Gate       : val_J_and_F > 0.55 @ epoch 50"
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

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

python train.py \
    model=mobilenetv2_kan_temporal \
    datamodule=davis \
    +experiment=mv2_phase2_davis \
    +pretrained_weights="$CKPT" \
    ++datamodule.img_size=480 \
    ++logger.name="mv2_phase2_davis" \
    ++logger.tags="[mobilenetv2,kan-ssm,davis-finetune,phase2,scheduled-sampling]" \
    ++callbacks.model_checkpoint.dirpath="checkpoints" \
    ++callbacks.model_checkpoint.filename="best_mv2_phase2_davis" \
    "$@"

echo ""
echo "Phase 2 finished at: $(date)"
echo ""
echo "Gate check: open Comet run 'mv2_phase2_davis' and verify"
echo "  val_J_and_F @ epoch 50 > 0.55 before submitting Phase 3."
echo "  If gate passed: sbatch scripts/train_mv2_phase3_ytbdav.sh"
