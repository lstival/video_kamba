#!/bin/bash
#SBATCH --job-name=mrmb_ytvos
#SBATCH --output=logs/memory_bank_propagation/youtubevos_%j.out
#SBATCH --error=logs/memory_bank_propagation/youtubevos_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Training with Reference Anchors (MRMB)
python train.py \
    datamodule=youtubevos \
    ++datamodule.num_workers=8 \
    trainer.max_epochs=50 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    model.learning_rate=1e-4 \
    model.propagation_mode=soft_mask \
    +model.use_ref_context=True \
    model.fusion_mode=kan_spatial \
    logger=comet
