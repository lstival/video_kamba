#!/bin/bash
#SBATCH --job-name=rsp_davis_50ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rsp_masked_decoder/davis_%j.out
#SBATCH --error=logs/rsp_masked_decoder/davis_%j.err

mkdir -p logs/rsp_masked_decoder

# Activate environment
source venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run training for 50 epochs on DAVIS
python train.py \
    datamodule=davis \
    trainer.max_epochs=50 \
    trainer.precision=16-mixed \
    model.propagation_mode=soft_mask \
    model.learning_rate=1e-4 \
    hydra.run.dir=logs/rsp_masked_decoder/davis_${SLURM_JOB_ID} \
    seed=42
