#!/bin/bash
#SBATCH --job-name=kan_ytvos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/kan_ytvos_%j.out
#SBATCH --error=logs/slurm/kan_ytvos_%j.err

set -euo pipefail

# 1. Environment Setup
# Assuming 'venv' is the primary environment as seen in other production scripts.
source venv/bin/activate
export PYTHONPATH=.

# 2. Training on YouTube-VOS with KAN Decoder
echo "Starting YouTube-VOS Training with KAN Decoder on $(hostname)"
echo "Current Branch: $(git rev-parse --abbrev-ref HEAD)"

python train.py \
    datamodule=youtubevos \
    datamodule.batch_size=2 \
    trainer.max_epochs=20 \
    trainer.devices=1 \
    model.fusion_mode=kan_spatial \
    model.modulator_type=kan \
    model.learning_rate=1e-5 \
    logger=comet \
    ++logger.project="vos-kan-decoder" \
    ++logger.name="ytvos-kan-spatial-v1"

echo "Training Finished."
