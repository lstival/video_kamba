#!/bin/bash
#SBATCH --job-name=grad_check
#SBATCH --output=logs/grad_check_%j.out
#SBATCH --error=logs/grad_check_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

python scripts/verify_gradient_flow.py
