#!/bin/bash
#SBATCH --job-name=vos_pretrain_yt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/pretrain_yt_%j.out
#SBATCH --error=logs/slurm/pretrain_yt_%j.err

set -euo pipefail

# 1. Environment Setup
source venv/bin/activate
export PYTHONPATH=.

# 2. Training on YouTube-VOS
# Using the new advanced architecture on the current branch.
# Hierarchical Decoder + Recursive Memory Bank are now default in VideoMambaSystem on this branch.
echo "Starting YouTube-VOS Pre-training on $(hostname)"

python train.py \
    datamodule=youtubevos \
    datamodule.batch_size=2 \
    trainer.max_epochs=20 \
    trainer.devices=1 \
    model.learning_rate=1e-5 \
    logger=comet \
    ++logger.project_name="vos-advanced-pretrain" \
    ++logger.experiment_name="ytvos-recursive-hfusion-v1"

echo "Pre-training Finished."
