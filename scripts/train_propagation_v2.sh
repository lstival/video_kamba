#!/bin/bash
#SBATCH --job-name=vos_prop_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/prop_v2_%j.out
#SBATCH --error=logs/slurm/prop_v2_%j.err

# ============================================================
# Phase 1 validation run — 20 epochs
#
# Fixes applied vs propagation_fix_davis_50ep:
#   1. KAN RBF weight init std: 0.02 → 0.1
#      (gates start with meaningful activations; avoids 0.5-neutral phase)
#   2. MemoryBank void-pixel fix: void patches zeroed before ID embedding
#   3. Gradient clipping (norm, val=1.0) via trainer default config
#      (prevents gradient explosions through the T=16 propagation loop)
#   5. Differential LR: memory_bank + prop_attention get 10× LR (1e-3 vs 1e-4)
#      Diagnostic showed attention entropy = 0.978 (near-uniform) — Q-K projections
#      never learned to align because decoder skip connections (frozen Hiera) provided
#      a gradient shortcut. Higher LR forces Q-K to become discriminative faster.
#
# Compare on Comet: train_loss_epoch curve slope vs baseline.
# Target: smooth monotonic decrease in train_loss, lower val_loss by ep 15.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  VOS Propagation v2 — DAVIS Training"
echo "  Fix: KAN gate init + void-pixel mem"
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
    ++trainer.max_epochs=20 \
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
    ++logger.name="propagation_v3_20ep" \
    "$@"

echo ""
echo "Training finished at: $(date)"
