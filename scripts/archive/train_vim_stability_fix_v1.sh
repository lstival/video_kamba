#!/bin/bash
#SBATCH --job-name=vim_stability_fix_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vim_stability_fix_v1_%j.out
#SBATCH --error=logs/slurm/vim_stability_fix_v1_%j.err

# ============================================================
# Train Dynamic Vision-Mamba with Stability Fixes (v1)
#
# Addresses issues from Job 65725194:
#   - Exploding gradients in PropagationAttention
#   - Aggressive Learning Rate (remaps 4e-4 -> 1e-4)
#   - Missing per-layer gradient tracking
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Stability Fix v1"
echo "  Experiment: stability_fix_v1"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm checkpoints

# Load secrets from .env (never committed to git)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

# Launch training with the stability_fix_v1 experiment override
python train.py \
    model=vision_mamba_tiny \
    datamodule=ade20k_pretrain \
    +experiment=stability_fix_v1 \
    ++model.use_identity_modulation=True \
    ++trainer.max_epochs=30 \
    ++logger.name="vim_stability_fix_v1" \
    "$@"

echo ""
echo "Stability Fix Training finished at: $(date)"
