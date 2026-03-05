#!/bin/bash
#SBATCH --job-name=exp1_entropy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/exp1_%j.out
#SBATCH --error=logs/slurm/exp1_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Experiment 1 — Activation Entropy"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Required overrides:
#   kan_ckpt=<path>   e.g. lightning_logs/version_kan/checkpoints/best.ckpt
#   mlp_ckpt=<path>   e.g. lightning_logs/version_mlp/checkpoints/best.ckpt
#
# Example:
#   sbatch scripts/slurm_exp1.sh \
#       kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \
#       mlp_ckpt=lightning_logs/version_mlp/checkpoints/best.ckpt

python scripts/exp1_activation_entropy.py \
    data_dir=data/DAVIS/DAVIS \
    output_dir=results/exp1 \
    max_batches=50 \
    seed=42 \
    "$@"

echo ""
echo "Experiment 1 complete at $(date)."
