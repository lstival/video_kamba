#!/bin/bash
#SBATCH --job-name=dinov2_ytvos_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/dinov2_ytvos_ft_%j.out
#SBATCH --error=logs/slurm/dinov2_ytvos_ft_%j.err

# ============================================================
# Fine-tune: DINOv2 on YouTube-VOS, starting from a trained
#            checkpoint produced by train_dinov2_youtubevos_20ep.sh
#
# Usage:
#   sbatch scripts/finetune_dinov2_youtubevos.sh \
#       ++checkpoint=/path/to/checkpoints/best.ckpt
#
# What changes vs the original 100-ep run:
#   learning_rate  : 1e-4 → 2e-5   (5× lower — preserve learned weights)
#   max_epochs     : 100  → 30      (fine-tuning converges fast)
#   scheduled_sampling_rate: 0.3 → 0.5  (more self-prediction, less GT crutch)
#   logger.name    : tracks as a separate Comet experiment
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# ── Resolve checkpoint path ──────────────────────────────────────────────────
# Can be passed as a Hydra override: ++checkpoint=/abs/path/to/best.ckpt
# If not passed, try to find the best checkpoint from the most recent run.
CKPT_PROVIDED=false
CKPT_OVERRIDE=""
for arg in "$@"; do
    if [[ "$arg" == ++checkpoint=* ]]; then
        CKPT_PROVIDED=true
        CKPT_OVERRIDE="$arg"
        break
    fi
done

if [ "$CKPT_PROVIDED" = false ]; then
    echo "No checkpoint override provided. Searching for best checkpoint..."
    # Look in the canonical checkpoints/ directory (set by callbacks.dirpath)
    BEST_CKPT=$(find checkpoints/ -name "*.ckpt" ! -name "last.ckpt" \
        2>/dev/null | sort | tail -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "ERROR: No .ckpt file found in checkpoints/."
        echo "Pass one explicitly: sbatch finetune_dinov2_youtubevos.sh ++checkpoint=/abs/path/best.ckpt"
        exit 1
    fi
    CKPT_OVERRIDE="++checkpoint=\"${PROJECT_ROOT}/${BEST_CKPT}\""
    echo "Auto-selected checkpoint: $BEST_CKPT"
fi

# We want to make sure the checkpoint value is quoted if it's passed in "$@"
# to avoid Hydra grammar errors with paths containing '='.
# We'll build a new array of arguments.
FINAL_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == ++checkpoint=* ]]; then
        CKPT_PROVIDED=true
        VAL="${arg#++checkpoint=}"
        # Remove existing quotes if any, then wrap in double quotes
        VAL="${VAL%\"}"
        VAL="${VAL#\"}"
        VAL="${VAL%\'}"
        VAL="${VAL#\'}"
        FINAL_ARGS+=("++checkpoint=\"$VAL\"")
    else
        FINAL_ARGS+=("$arg")
    fi
done

# If not provided, add the auto-selected one
if [ "$CKPT_PROVIDED" = false ]; then
    FINAL_ARGS=("${CKPT_OVERRIDE}" "${FINAL_ARGS[@]}")
fi

echo "=========================================="
echo "  Video Kamba: DINOv2 Fine-tune (YouTube-VOS)"
echo "  Encoder   : DINOv2 ViT-B/14 (frozen)"
echo "  Epochs    : 30"
echo "  LR        : 2e-5 (5x lower than base run)"
echo "  SSR       : 0.5  (more self-prediction)"
echo "  Checkpoint: $CKPT_OVERRIDE"
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

mkdir -p logs/slurm

python train.py \
    model=default \
    datamodule=youtubevos \
    ++model.encoder_type=dino \
    ++model.target_size=448 \
    ++model.max_epochs=30 \
    ++model.learning_rate=2e-5 \
    ++model.scheduled_sampling_rate=0.5 \
    ++datamodule.img_size=448 \
    ++trainer.max_epochs=30 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="dinov2_ytvos_ft_30ep" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    "${FINAL_ARGS[@]}"

echo ""
echo "Fine-tuning finished at: $(date)"
