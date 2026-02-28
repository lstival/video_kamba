#!/bin/bash
#SBATCH --job-name=train_best_davis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_best_%j.out
#SBATCH --error=logs/slurm/train_best_%j.err

# ==========================================
# Best Model Training & Evaluation (DAVIS)
# 100 Epochs - Tactic D+E: Loss Optimization
# ==========================================

set -euo pipefail

echo "=========================================="
echo "  VOS Best Model Pipeline Started"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# 1. Training Phase
# Usage: sbatch scripts/train_best_davis.sh [hydra overrides]
echo -e "\n[1/2] Training best configuration (Tactic D+E)..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=100 \
    ++model.vos_loss_beta=0.2 \
    ++model.consistency_weight=0.1 \
    ++logger.name="best_model_davis_100ep" \
    "$@"

# 2. Identify the best checkpoint
# Hydra setup in config.yaml: logs/runs/%Y-%m-%d_%H-%M-%S
# We look for the most recent directory in logs/runs
LATEST_RUN=$(ls -td logs/runs/*/ | head -1)
echo "Looking for checkpoints in $LATEST_RUN/checkpoints"

# Try to find the checkpoint with the lowest val_loss in the filename
# Filename pattern: epoch=X-val_loss=Y.ckpt
BEST_CKPT=$(find "$LATEST_RUN/checkpoints" -name "epoch=*.ckpt" | grep -v "last.ckpt" | sort -t'=' -k3n | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "Warning: No checkpoints in logs/runs. Checking logs/video_mamba (Comet)..."
    LATEST_COMET_RUN=$(ls -td logs/video_mamba/*/ | head -1)
    if [ -n "$LATEST_COMET_RUN" ]; then
        BEST_CKPT=$(find "$LATEST_COMET_RUN/checkpoints" -name "epoch=*.ckpt" | grep -v "last.ckpt" | sort -t'=' -k3n | head -1)
    fi
fi

if [ -z "$BEST_CKPT" ]; then
    echo "Warning: Specific epoch checkpoint not found. Trying last.ckpt..."
    BEST_CKPT=$(find "$LATEST_RUN/checkpoints" -name "last.ckpt" | head -1) || { [ -n "${LATEST_COMET_RUN:-}" ] && BEST_CKPT=$(find "$LATEST_COMET_RUN/checkpoints" -name "last.ckpt" | head -1); }
fi

if [ -z "${BEST_CKPT:-}" ]; then
    echo "Error: No checkpoints found."
    exit 1
fi

echo -e "\nFound best checkpoint: $BEST_CKPT"

# 3. Evaluation Phase (Submitted as a new Slurm job)
echo -e "\n[2/2] Submitting DAVIS Evaluation Job..."
sbatch scripts/evaluate_davis.sh checkpoint="$BEST_CKPT" datamodule=davis

echo -e "\n=========================================="
echo "Pipeline Training Finished at $(date)"
echo "Evaluation job submitted. Check squeue for status."
echo "=========================================="
