#!/bin/bash
#SBATCH --job-name=exp3_rbf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/exp3_%j.out
#SBATCH --error=logs/slurm/exp3_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Experiment 3 — RBF Activation Profile"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Required overrides:
#   kan_ckpt=<path>  e.g. lightning_logs/version_kan/checkpoints/best.ckpt
#
# Example:
#   sbatch scripts/slurm_exp3.sh \
#       kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt

python scripts/exp3_rbf_profile.py \
    data_dir=data/DAVIS/DAVIS \
    output_dir=results/exp3 \
    n_top_channels=5 \
    max_batches=30 \
    seed=42 \
    "$@"

echo ""
echo "Experiment 3 complete at $(date)."
