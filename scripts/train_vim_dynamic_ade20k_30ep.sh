#!/bin/bash
#SBATCH --job-name=vim_dynamic_ade20k_30ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vim_dynamic_ade20k_30ep_%j.out
#SBATCH --error=logs/slurm/vim_dynamic_ade20k_30ep_%j.err

# ============================================================
# Train Dynamic Vision-Mamba (Option 1) on ADE20K Pretrain.
#
# Highlights:
#   - use_identity_modulation=True (Dynamic Tracker)
#   - vision_mamba_tiny encoder (trainable)
#   - ADE20K dataset for higher quality masks
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Dynamic Vision-Mamba (ADE20K)"
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

python train.py \
    model=vision_mamba_tiny \
    datamodule=ade20k_pretrain \
    ++model.use_identity_modulation=True \
    ++model.learning_rate=4e-4 \
    ++model.target_size=480 \
    ++datamodule.batch_size=32 \
    ++datamodule.num_workers=8 \
    ++datamodule.streaming=false \
    ++datamodule.cache_dir="${PROJECT_ROOT}/data/ade20k_cache" \
    ++trainer.max_epochs=30 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++trainer.limit_train_batches=2000 \
    ++trainer.val_check_interval=0.5 \
    ++trainer.limit_val_batches=200 \
    ++logger.name="vim_dynamic_ade20k_30ep" \
    ++callbacks.monitor=val_loss \
    ++callbacks.mode=min \
    "$@"

echo ""
echo "ADE20K Training finished at: $(date)"
