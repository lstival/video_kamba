#!/bin/bash
#SBATCH --job-name=vim_stability_fix_v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vim_stability_fix_v3_%j.out
#SBATCH --error=logs/slurm/vim_stability_fix_v3_%j.err

# ============================================================
# Train Dynamic Vision-Mamba with Stability Fixes (v3)
#
# Addresses the persistent gradient explosions from v2 (job 65737509):
#
#   Root cause:
#     - out_proj.weight in KangaSSM / PropagationAttention was the primary
#       explosion site (peak norms 2.05e+05 and 2.98e+05 respectively).
#     - KAN modulator softplus (unbounded) amplified SSM output by up to
#       1.8e+04× at stage3_ssm.C_modulator.
#     - BCNorm affine gamma itself grew to 1.2e+03, negating normalisation.
#     - propagation_lr_multiplier=5.0 exacerbated attention explosion.
#
#   Architecture fixes (v3):
#     1. SwiGLU gated out_proj in KangaSSM: Linear(D→2D), up * silu(gate)
#     2. SwiGLU gated proj_out in PropagationAttention
#     3. KAN modulator activation: softplus → tanh_bounded → [0.5, 1.5]
#     4. BCNorm: elementwise_affine=False (no learnable gamma)
#
#   Config fixes (v3):
#     5. propagation_lr_multiplier: 5.0 → 1.0
#     6. temporal_lr_multiplier: 0.5 (new separate LR group for KangaSSM)
#     7. trainable_backbone_lr_multiplier: 0.1 → 0.05
#     8. gradient_clip_val: 1.0 → 0.5
#     9. LR: 5e-5 → 3e-5
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Stability Fix v3"
echo "  Experiment: stability_fix_v3"
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

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

python train.py \
    model=vision_mamba_tiny \
    datamodule=ade20k_pretrain \
    +experiment=stability_fix_v3 \
    ++model.use_identity_modulation=True \
    ++trainer.max_epochs=30 \
    ++logger.name="vim_stability_fix_v3" \
    "$@"

echo ""
echo "Stability Fix v3 Training finished at: $(date)"
