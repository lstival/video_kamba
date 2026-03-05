#!/bin/bash
#SBATCH --job-name=exp2_iou_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/exp2_%j.out
#SBATCH --error=logs/slurm/exp2_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Experiment 2 — Spatial Gate Fidelity"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Required overrides:
#   kan_ckpt=<path>    e.g. lightning_logs/version_kan/checkpoints/best.ckpt
#   xattn_ckpt=<path>  e.g. lightning_logs/version_xattn/checkpoints/best.ckpt
#
# Example:
#   sbatch scripts/slurm_exp2.sh \
#       kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \
#       xattn_ckpt=lightning_logs/version_xattn/checkpoints/best.ckpt

python scripts/exp2_spatial_gate_iou.py \
    data_dir=data/DAVIS/DAVIS \
    output_dir=results/exp2 \
    tau=0.5 \
    save_visualizations=true \
    vis_max_seqs=10 \
    seed=42 \
    "$@"

echo ""
echo "Experiment 2 complete at $(date)."
