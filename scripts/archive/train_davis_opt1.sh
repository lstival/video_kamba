#!/bin/bash
#SBATCH --job-name=train_opt1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_opt1_%j.out
#SBATCH --error=logs/slurm/train_opt1_%j.err

# ==========================================
# Option 1 Training & Evaluation (DAVIS)
# High Capacity SSM + Modulate + Grad Accum
# ==========================================

set -euo pipefail

echo "=========================================="
echo "  Option 1 Pipeline Started"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# 1. Training Phase
echo -e "\n[1/2] Training Option 1 (High Capacity SSM + Modulate)..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=100 \
    ++trainer.accumulate_grad_batches=2 \
    ++model.ssm_d_state=64 \
    ++model.ssm_layers=2 \
    ++model.identity_mode="modulate" \
    ++model.use_checkpointing=True \
    ++model.vos_loss_beta=0.2 \
    ++model.consistency_weight=0.1 \
    ++logger.name="opt1_davis_modulate_64" \
    "$@"

# 2. Identify the best checkpoint
LATEST_RUN=$(ls -td logs/runs/*/ | head -1)
echo "Looking for checkpoints in $LATEST_RUN/checkpoints"

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

# 3. Evaluation Phase
echo -e "\n[2/2] Submitting DAVIS Evaluation Job (Opt1)..."
sbatch scripts/evaluate_davis.sh checkpoint="$BEST_CKPT" datamodule=davis

echo -e "\n=========================================="
echo "Pipeline Training Finished at $(date)"
echo "=========================================="
