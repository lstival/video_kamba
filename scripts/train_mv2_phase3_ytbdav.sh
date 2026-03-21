#!/bin/bash
#SBATCH --job-name=mv2_phase3_ytbdav
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/mv2_phase3_ytbdav_%j.out
#SBATCH --error=logs/slurm/mv2_phase3_ytbdav_%j.err

# ============================================================
# Phase 3 — Joint YouTube-VOS + DAVIS (MobileNetV2 + KAN-SSM)
#
# Initialise from Phase 2 checkpoint (best_mv2_phase2_davis.ckpt).
#
# Fixes vs. job 65764336 (OOM on step 1):
#   - batch_size: 4→2, accumulate_grad_batches: 2→4 (same effective=8, half peak VRAM)
#   - gradient_clip_val: 0.5→0.1 (backbone norms hit 5e3 on first backward)
#   - precision: fp16→bf16-mixed (fp16 saturates at 65504; proj_out hit 2e+04)
#   - use_checkpointing: true (was missing from Phase 3, adds ~30% VRAM saving)
#   - lr_warmup_epochs: 2→5, lr_warmup_start_factor: 0.1→0.01 (stages 0-1 fire cold)
#   - mem: 64G→80G (match Phase 2 which was stable)
#
# Data: YouTube-VOS 2019 + DAVIS 2017 joint (DAVIS = 25% per epoch)
# Budget: 50 epochs × 3500 steps × ~1.4h/epoch ≈ 70h
#
# Target: val_J_and_F > 0.65 at epoch 50
# Reference: MobileVOS (CVPR 2023) ≈ 0.78 J&F (same backbone, Transformer)
#
# Output: checkpoints/best_mv2_phase3_ytbdav.ckpt
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase2_davis.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Error: Phase 2 checkpoint not found at $CKPT"
    echo "Run sbatch scripts/train_mv2_phase2_davis.sh first."
    echo "Ensure val_J_and_F > 0.55 @ epoch 50 before proceeding."
    exit 1
fi

echo "=========================================="
echo "  MV2 + KAN-SSM: Phase 3 — YTB+DAV Joint"
echo "  Checkpoint : $CKPT"
echo "  Job ID     : ${SLURM_JOB_ID:-manual}"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "  Target     : val_J_and_F > 0.65 @ epoch 50"
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
    datamodule=ytv_dav_joint \
    +experiment=mv2_phase3_ytbdav \
    +pretrained_weights="$CKPT" \
    ++logger.name="mv2_phase3_ytbdav" \
    ++logger.tags="[mobilenetv2,kan-ssm,ytbdav-joint,phase3,scheduled-sampling,davis-init]" \
    ++callbacks.model_checkpoint.dirpath="checkpoints" \
    ++callbacks.model_checkpoint.filename="best_mv2_phase3_ytbdav" \
    "$@"

echo ""
echo "Phase 3 finished at: $(date)"
echo "Best checkpoint: checkpoints/best_mv2_phase3_ytbdav.ckpt"
