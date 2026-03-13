#!/bin/bash
#SBATCH --job-name=vos_yt_100ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/train_yt_100ep_%j.out
#SBATCH --error=logs/slurm/train_yt_100ep_%j.err

set -euo pipefail

# 1. Environment Setup
source venv/bin/activate
export PYTHONPATH=.

# 2. Training on YouTube-VOS
echo "Starting YouTube-VOS Training (100 Epochs) on $(hostname)"

python train.py \
    datamodule=youtubevos \
    datamodule.batch_size=2 \
    ++datamodule.num_workers=8 \
    trainer.max_epochs=100 \
    trainer.devices=1 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    model.learning_rate=1e-5 \
    logger=comet \
    ++logger.project_name="vos-advanced-pretrain" \
    ++logger.experiment_name="ytvos-100-epochs"

echo "Training Finished."
