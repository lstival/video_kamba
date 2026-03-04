#!/bin/bash
#SBATCH --job-name=vos_finetune_davis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/finetune_davis_%j.out
#SBATCH --error=logs/slurm/finetune_davis_%j.err

set -euo pipefail

# ==============================================================================
# DAVIS Fine-tuning Script (Advanced Architecture)
# ==============================================================================
# Usage: sbatch scripts/finetune_advanced_davis.sh checkpoint=/path/to/pretrained.ckpt
# ==============================================================================

source venv/bin/activate
export PYTHONPATH=.

echo "=============================================================================="
echo "  DAVIS Fine-tuning Pipeline Started"
echo "  Target: 80%+ J&F"
echo "  Pre-trained Weights: ${1:-None}"
echo "=============================================================================="

# 1. Fine-tuning Phase
# Uses the recursive architecture + multi-scale features (defaults on this branch)
# Plus the best loss configuration (Tactic D+E)
python train.py \
    datamodule=davis \
    trainer.max_epochs=100 \
    model.learning_rate=1e-5 \
    model.vos_loss_beta=0.2 \
    model.consistency_weight=0.1 \
    logger=comet \
    ++logger.project_name="vos-advanced-finetune" \
    ++logger.experiment_name="davis-fine-tuning-v1" \
    "$@"

# 2. Identify the best checkpoint from this run
LATEST_RUN=$(ls -td logs/runs/*/ | head -1)
echo "Looking for fine-tuned checkpoints in $LATEST_RUN/checkpoints"

BEST_CKPT=$(find "$LATEST_RUN/checkpoints" -name "epoch=*.ckpt" | grep -v "last.ckpt" | sort -t'=' -k3n | head -1)

if [ -z "$BEST_CKPT" ]; then
    # Fallback to Comet logs if Hydra is inconsistent
    LATEST_COMET_RUN=$(ls -td logs/video_mamba/*/ | head -1)
    if [ -n "$LATEST_COMET_RUN" ]; then
        BEST_CKPT=$(find "$LATEST_COMET_RUN/checkpoints" -name "epoch=*.ckpt" | grep -v "last.ckpt" | sort -t'=' -k3n | head -1)
    fi
fi

if [ -n "$BEST_CKPT" ]; then
    echo -e "\nFound best fine-tuned checkpoint: $BEST_CKPT"
    echo "Submitting final DAVIS Evaluation..."
    sbatch scripts/evaluate_davis.sh checkpoint="$BEST_CKPT" datamodule=davis
else
    echo "Warning: No checkpoint found for evaluation."
fi

echo -e "\n=============================================================================="
echo "  Fine-tuning Pipeline Finished at $(date)"
echo "=============================================================================="
