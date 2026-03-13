#!/bin/bash
#SBATCH --job-name=smoke_test_rsp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/smoke_test_%j.out
#SBATCH --error=slurm_logs/smoke_test_%j.err

mkdir -p slurm_logs

# Activate environment
source venv/bin/python -m venv venv --system-site-packages || true
source venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run smoke test
python scripts/smoke_test_gradient.py
