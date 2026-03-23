#!/bin/bash
#SBATCH --job-name=ft_ytvos_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/ft_ytvos_v2_%j.out
#SBATCH --error=logs/slurm/ft_ytvos_v2_%j.err

# ============================================================
# YouTube-VOS Fine-tuning v2 — after 30-ep COCO pretraining
#
# Improvements vs finetune_coco_to_ytvos.sh:
#   - SSR increased 0.3 → 0.5: forces model to propagate its
#     own masks earlier, closing the seen→unseen gap
#   - Extended to 50 epochs (vs 30) for more temporal exposure
#   - Resolution 480 (native YouTube-VOS, vs 448)
#   - LR kept at 2e-5 (stable with a well-converged COCO ckpt)
#
# Usage:
#   sbatch scripts/finetune_ytvos_v2.sh
#
# Or pass a different checkpoint:
#   sbatch scripts/finetune_ytvos_v2.sh \
#       checkpoint=/path/to/custom.ckpt
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# Default: use the 30-ep COCO checkpoint; override via $1 if needed
CKPT="${PROJECT_ROOT}/checkpoints/best_coco_30ep.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "WARNING: best_coco_30ep.ckpt not found, falling back to best_coco.ckpt"
    CKPT="${PROJECT_ROOT}/checkpoints/best_coco.ckpt"
fi

echo "=========================================="
echo "  Video Kamba: Fine-tune COCO → YouTube-VOS (v2)"
echo "  Checkpoint  : $CKPT"
echo "  Epochs      : 50"
echo "  LR          : 2e-5"
echo "  img_size    : 480"
echo "  SSR         : 0.5  (forces own-mask propagation)"
echo "  Job ID      : ${SLURM_JOB_ID:-manual}"
echo "  Node        : $(hostname)"
echo "  Start       : $(date)"
echo "=========================================="

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No COCO checkpoint found. Run pretrain_coco_30ep.sh first."
    exit 1
fi

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm

python train.py \
    model=default \
    datamodule=youtubevos \
    ++model.encoder_type=dino \
    ++model.target_size=480 \
    ++model.max_epochs=50 \
    ++model.learning_rate=2e-5 \
    ++model.scheduled_sampling_rate=0.5 \
    ++model.use_kan_key_adapter=true \
    ++datamodule.img_size=480 \
    ++trainer.max_epochs=50 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="ft_ytvos_v2_50ep" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    +checkpoint="${CKPT}" \
    "$@"

echo ""
echo "YouTube-VOS fine-tuning v2 done: $(date)"
