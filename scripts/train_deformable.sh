#!/bin/bash
#SBATCH --job-name=train_def
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_deformable_%j.out
#SBATCH --error=logs/slurm/train_deformable_%j.err

source venv/bin/activate
export PYTHONPATH=.

# Train on YouTubeVOS for 50 epochs using the Deformable upblock
python train.py \
    datamodule=youtubevos \
    trainer.max_epochs=50 \
    model.fusion_mode='deformable'
