#!/bin/bash
#SBATCH --job-name=vim_dynamic_coco_30ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vim_dynamic_coco_30ep_%j.out
#SBATCH --error=logs/slurm/vim_dynamic_coco_30ep_%j.err

# ============================================================
# Train Dynamic Vision-Mamba (Option 1) on COCO Warmup.
#
# Highlights:
#   - use_identity_modulation=True (Dynamic Tracker)
#   - vision_mamba_tiny encoder (trainable)
#   - Increased backbone LR for from-scratch warmup
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
# shellcheck disable=SC2021
echo "  Video Kamba: Dynamic Vision-Mamba (COCO)"
echo "  Encoder   : vision_mamba_tiny (Dynamic Modulation)"
echo "  Epochs    : 30"
echo "  Target Res: 480x480"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
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

# We increase the learning rate for the scratch-trained backbone (warmup)
# and enable identity modulation.
python train.py \
    model=vision_mamba_tiny \
    datamodule=coco_pretrain \
    ++model.use_identity_modulation=True \
    ++model.learning_rate=2e-4 \
    ++model.target_size=480 \
    ++datamodule.img_size=480 \
    ++datamodule.streaming=False \
    ++datamodule.cache_dir="${PROJECT_ROOT}/data/coco_cache" \
    ++datamodule.num_workers=2 \
    ++trainer.max_epochs=30 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++trainer.limit_train_batches=5000 \
    ++trainer.val_check_interval=0.25 \
    ++trainer.limit_val_batches=200 \
    ++logger.name="vim_dynamic_coco_30ep" \
    ++callbacks.monitor=val_loss \
    ++callbacks.mode=min \
    "$@"

echo ""
echo "Training finished at: $(date)"
