#!/bin/bash
#SBATCH --job-name=rsp_trial_2epochs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/rsp_trial_%j.out
#SBATCH --error=slurm_logs/rsp_trial_%j.err

mkdir -p slurm_logs

# Activate environment
source venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run training for 2 epochs on YouTube-VOS
# Overriding epochs and possibly batch size if needed for the trial
python train.py \
    datamodule=youtubevos \
    trainer.max_epochs=2 \
    trainer.precision=16-mixed \
    model.propagation_mode=soft_mask \
    seed=42
