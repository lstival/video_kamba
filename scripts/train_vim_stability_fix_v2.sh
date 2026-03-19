#!/bin/bash
#SBATCH --job-name=vim_stability_fix_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vim_stability_fix_v2_%j.out
#SBATCH --error=logs/slurm/vim_stability_fix_v2_%j.err

# ============================================================
# Train Dynamic Vision-Mamba with Stability Fixes (v2)
#
# Addresses issues from vim_stability_fix_v1 (Job 65730252):
#   - BCNorm (Mamba-3): RMSNorm after B/C in DiagonalKANSSMCore
#   - Post-scan LayerNorm in KangaSSM
#   - Hidden state clamping [-10, 10]
#   - KAN init: Xavier base_weight, rbf std 0.05
#   - gradient_clip_val: 0.1 → 1.0 (standard Mamba)
#   - Coherent motion trajectories (6 frames, smooth pan/zoom)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Stability Fix v2"
echo "  Experiment: stability_fix_v2"
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

# Launch training with the stability_fix_v2 experiment override
python train.py \
    model=vision_mamba_tiny \
    datamodule=ade20k_pretrain \
    +experiment=stability_fix_v2 \
    ++model.use_identity_modulation=True \
    ++trainer.max_epochs=30 \
    ++logger.name="vim_stability_fix_v2" \
    "$@"

echo ""
echo "Stability Fix v2 Training finished at: $(date)"
