#!/bin/bash
#SBATCH --job-name=train_davis_soft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_davis_soft_%j.out
#SBATCH --error=logs/slurm/train_davis_soft_%j.err

source venv/bin/activate
export PYTHONPATH=.

# Train on DAVIS for 50 epochs using the new soft mask architecture
python train.py \
    datamodule=davis \
    trainer.max_epochs=50
