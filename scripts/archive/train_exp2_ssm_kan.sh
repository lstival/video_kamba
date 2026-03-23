#!/bin/bash
#SBATCH --job-name=exp2_ssm_kan
#SBATCH --output=logs/exp2_ssm_kan_%j.out
#SBATCH --error=logs/exp2_ssm_kan_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source venv/bin/activate

python train.py \
    datamodule=davis \
    trainer.max_epochs=50 \
    model.learning_rate=1e-4 \
    model.propagation_mode=soft_mask \
    +model.use_ref_context=True \
    model.fusion_mode=kan_spatial \
    logger=none
