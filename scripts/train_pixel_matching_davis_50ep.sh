#!/bin/bash
#SBATCH --job-name=pixel_match
#SBATCH --output=logs/pixel_matching/davis_%j.out
#SBATCH --error=logs/pixel_matching/davis_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

mkdir -p logs/pixel_matching

# Experiment 1: Direct Pixel-level matching vs SSM
python train.py \
    datamodule=davis \
    trainer.max_epochs=50 \
    model.learning_rate=1e-4 \
    model.propagation_mode=pixel_matching \
    +model.use_ref_context=True \
    model.fusion_mode=kan_spatial \
    logger=comet
