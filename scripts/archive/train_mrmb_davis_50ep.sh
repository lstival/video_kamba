#!/bin/bash
#SBATCH --job-name=mrmb_davis
#SBATCH --output=logs/memory_bank_propagation/davis_%j.out
#SBATCH --error=logs/memory_bank_propagation/davis_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Training with Reference Anchors (MRMB)
python train.py \
    datamodule=davis \
    trainer.max_epochs=50 \
    model.learning_rate=1e-4 \
    model.propagation_mode=soft_mask \
    +model.use_ref_context=True \
    model.fusion_mode=kan_spatial \
    logger=comet
