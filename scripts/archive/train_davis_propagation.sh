#!/bin/bash
#SBATCH --job-name=vos_propagation_davis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/propagation_davis_%j.out
#SBATCH --error=logs/slurm/propagation_davis_%j.err

# ============================================================
# DAVIS training with MemoryBank + PropagationAttention (fix)
# Branch: fix/claude_decoder
# 50 epochs — DAVIS semi-supervised VOS
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  VOS Propagation Fix — DAVIS Training"
echo "  Branch  : fix/claude_decoder"
echo "  Job ID  : ${SLURM_JOB_ID:-manual}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.

python train.py \
    datamodule=davis \
    ++trainer.max_epochs=50 \
    ++trainer.precision="16-mixed" \
    ++model.prop_d_key=256 \
    ++model.prop_d_value=256 \
    ++model.prop_n_heads=8 \
    ++model.max_mem_frames=5 \
    ++model.memory_update_freq=1 \
    ++model.scheduled_sampling_rate=0.0 \
    ++model.fusion_mode="kan_spatial" \
    ++model.modulator_type="kan" \
    ++model.vos_loss_beta=0.5 \
    ++logger.name="propagation_fix_davis_50ep" \
    "$@"

echo ""
echo "Training finished at: $(date)"
