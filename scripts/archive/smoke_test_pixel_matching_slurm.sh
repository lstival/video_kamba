#!/bin/bash
#SBATCH --job-name=smoke_pixel
#SBATCH --output=logs/smoke_pixel_%j.out
#SBATCH --error=logs/smoke_pixel_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

mkdir -p logs

# Smoke test for pixel matching via SLURM
python train.py \
    datamodule=davis \
    trainer.max_epochs=1 \
    model.learning_rate=1e-4 \
    model.propagation_mode=pixel_matching \
    +model.use_ref_context=True \
    model.fusion_mode=kan_spatial \
    logger=none \
    +trainer.limit_train_batches=2 \
    +trainer.limit_val_batches=2
