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
# Phase 2 — MV2 training (DAVIS or BL30K/DL30K)
#
# Initialise from Phase 1b checkpoint (best_mv2_phase1b_bl30k.ckpt).
# Falls back to Phase 1 COCO checkpoint if Phase 1b is absent.
#
# Dataset mode (env var TRAIN_DATASET):
#   - bl30k (default): uses BL30K/DL30K synthetic clips
#   - davis: DAVIS 2017 semi-supervised (60 train / 30 val sequences)
#
# For BL30K/DL30K with limited home quota, set:
#   BL30K_DIR=/path/on/scratch_or_shared_storage/BL30K
# and keep data outside ${PROJECT_ROOT}/data.
#
# Budget: 100 epochs × 500 steps × ~12s/step ≈ 17h
#
# Gate decision @ epoch 50 (check Comet: mv2_phase2_davis):
#   val_J_and_F > 0.55 → proceed to Phase 3
#   val_J_and_F < 0.55 → diagnose (check Phase 1b ckpt quality)
#
# Output: checkpoints/best_mv2_phase2_davis.ckpt
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

TRAIN_DATASET="$(printf '%s' "${TRAIN_DATASET:-bl30k}" | tr '[:upper:]' '[:lower:]')"

DATAMODULE="davis_vos"
DATA_DESC="DAVIS 2017"
LOGGER_NAME="mv2_phase2_davis"
LOGGER_TAGS="[mobilenetv2,kan-ssm,davis-finetune,phase2,scheduled-sampling]"
OUT_CKPT="best_mv2_phase2_davis"
DATA_ARGS=()

if [ "$TRAIN_DATASET" = "bl30k" ] || [ "$TRAIN_DATASET" = "dl30k" ]; then
    BL30K_DIR="${BL30K_DIR:-${PROJECT_ROOT}/data/BL30K}"
    if [ ! -d "$BL30K_DIR" ] && [ -d "/scratch/${USER}/BL30K" ]; then
        BL30K_DIR="/scratch/${USER}/BL30K"
    fi

    if [ ! -d "$BL30K_DIR" ]; then
        echo "Error: BL30K/DL30K data directory not found: $BL30K_DIR"
        echo "Set BL30K_DIR to a scratch/shared path, e.g. /scratch/${USER}/BL30K"
        exit 1
    fi

    if ! find "$BL30K_DIR" -maxdepth 1 -type d -name 'BL30K_part*' | grep -q .; then
        echo "Error: no BL30K_part* folders found in $BL30K_DIR"
        echo "You can stage only a subset of parts to fit disk quota."
        exit 1
    fi

    DATAMODULE="bl30k_pretrain"
    DATA_DESC="BL30K/DL30K @ $BL30K_DIR"
    LOGGER_NAME="mv2_phase2_dl30k"
    LOGGER_TAGS="[mobilenetv2,kan-ssm,bl30k,phase2,quota-aware]"
    OUT_CKPT="best_mv2_phase2_dl30k"
    DATA_ARGS=(++datamodule.data_dir="$BL30K_DIR")
elif [ "$TRAIN_DATASET" != "davis" ]; then
    echo "Error: unsupported TRAIN_DATASET='$TRAIN_DATASET' (use 'bl30k' or 'davis')."
    exit 1
fi

# Prefer Phase 1b (BL30K) checkpoint; fall back to Phase 1 (COCO) if absent.
CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase1b_bl30k.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Phase 1b checkpoint not found at $CKPT"
    CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase1_coco.ckpt"
    echo "Falling back to Phase 1 checkpoint: $CKPT"
fi
if [ ! -f "$CKPT" ]; then
    echo "Error: No pre-training checkpoint found."
    echo "Run Phase 1b: sbatch scripts/train_mv2_phase1b_bl30k.sh"
    echo "Or Phase 1:   sbatch scripts/train_mv2_phase1_coco.sh"
    exit 1
fi

echo "=========================================="
echo "  MV2 + KAN-SSM: Phase 2"
echo "  Checkpoint : $CKPT"
echo "  Dataset    : $DATA_DESC"
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
    datamodule="$DATAMODULE" \
    +experiment=mv2_phase2_davis \
    +pretrained_weights="$CKPT" \
    ++logger.name="$LOGGER_NAME" \
    ++logger.tags="$LOGGER_TAGS" \
    ++callbacks.model_checkpoint.dirpath="checkpoints" \
    ++callbacks.model_checkpoint.filename="$OUT_CKPT" \
    "${DATA_ARGS[@]}" \
    "$@"

echo ""
echo "Phase 2 finished at: $(date)"
echo ""
echo "Gate check: open Comet run '$LOGGER_NAME' and verify"
echo "  val_J_and_F @ epoch 50 > 0.55 before submitting Phase 3."
echo "  If gate passed: sbatch scripts/train_mv2_phase3_ytbdav.sh"
