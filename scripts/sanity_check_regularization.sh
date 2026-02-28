#!/bin/bash
#SBATCH --job-name=vos_sanity_reg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/sanity_reg_%j.out
#SBATCH --error=logs/slurm/sanity_reg_%j.err

# ==========================================
# Video Mamba: Regularization Sanity Check
# 10 epochs on a subset to verify convergence
# ==========================================

echo "=========================================="
echo "  VOS Regularization Sanity Check"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Overfit on a single batch (sanity check)
# Using ++ to force adding keys to trainer config
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=10 \
    ++trainer.limit_train_batches=1 \
    ++trainer.limit_val_batches=1 \
    model.learning_rate=1e-4 \
    logger=none \
    callbacks.save_last=false

echo "=========================================="
echo "Sanity Check Finished at $(date)"
echo "=========================================="
