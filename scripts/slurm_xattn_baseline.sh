#!/bin/bash
#SBATCH --job-name=v_kamba_xattn_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/xattn_baseline_%j.out
#SBATCH --error=logs/slurm/xattn_baseline_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Cross-Attention Baseline Training (Ablation)"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate

export PYTHONPATH=.

# Train Cross-Attention baseline: fusion_mode=cross_attn, modulator_type=mlp
python train.py \
    datamodule=davis \
    model=xattn_baseline \
    "$@"

echo ""
echo "Cross-Attention baseline training complete at $(date)."
