#!/bin/bash
#SBATCH --job-name=analyze_conv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/analyze_convergence_%j.out

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "${PROJECT_ROOT}" || exit 1

source venv/bin/activate

python scripts/analyze_convergence.py
