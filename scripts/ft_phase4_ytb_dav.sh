#!/bin/bash
#SBATCH --job-name=ft_phase4_ytbdav
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/ft_phase4_ytbdav_%j.out
#SBATCH --error=logs/slurm/ft_phase4_ytbdav_%j.err

# Phase 4 (v2): Stabilised joint fine-tuning on YouTube-VOS + DAVIS.
#
# ── Root-cause discovery (2026-03-17) ────────────────────────────────────────
# Phase 3 AND Phase 4-v1 both started from `best_coco_sota_light.ckpt`
# (Phase 1 output) instead of `best_davis_sota_light.ckpt` (Phase 2 output).
#
# COCO pre-training never exposes the propagation components (MemoryBank,
# PropagationAttention, KangaSSM) to a VOS sequence — COCO uses single-image
# batches with no reference frame.  Those components were effectively random
# at the start of Phase 3/4-v1, explaining:
#   - Phase 3: peaked at epoch 0, degraded every subsequent epoch.
#   - Phase 4-v1: peaked at step 1749 (val_J_and_F ≈ 0.394) then declined
#     to 0.344 by step 5249 as the model overfitted to YouTube-VOS patterns.
#
# Phase 2 (ft_davis_fast_validate.sh) ran successfully and produced
# `checkpoints/best_davis_sota_light.ckpt` (DAVIS-only, 100 epochs, VOS-
# trained propagation components).  This checkpoint is the correct init.
#
# ── Changes vs Phase 4-v1 ────────────────────────────────────────────────────
#   - Starting checkpoint:    epoch=0-step=1750-v1.ckpt  → best_davis_sota_light.ckpt
#   - logger name:            v1                         → v2
#   - davis_sampling_ratio:   0.25                       → 0.40  (more DAVIS to
#                                                          resist YTB over-fitting)
#   All other hyper-parameters unchanged from Phase 4-v1 (LR 5e-6, warmup 10ep,
#   sched-sampling 0.1→0.5 over 25ep, grad_clip 0.3, 50 epochs).
#
# ── Epoch budget ─────────────────────────────────────────────────────────────
#   50 epochs × 3500 limit_train_batches × batch_size=4 × accum=2
#   = 1400 optimiser steps/epoch × 50 = 70 000 optimiser steps.
#   Wall time: ~1.42 h/epoch × 50 ≈ 71 h (fits 72 h SLURM limit).
#
# Usage:
#   sbatch scripts/ft_phase4_ytb_dav.sh

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm

# Load secrets from .env (never committed to git)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

# Phase 2 best checkpoint: DAVIS-only VOS fine-tune (100 epochs from COCO init).
# This is the correct starting point — propagation components fully trained for VOS.
CKPT="${PROJECT_ROOT}/checkpoints/best_davis_sota_light.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "Error: Phase 3 best checkpoint not found at $CKPT"
    echo "Available checkpoints:"
    ls "${PROJECT_ROOT}/checkpoints/"*.ckpt
    exit 1
fi

echo "=========================================="
echo "  Video Kamba: Phase 4-v2 — Stabilised FT (from Phase 2)"
echo "  Checkpoint : $CKPT"
echo "  Model      : vision_mamba_tiny_sota_light"
echo "  SSM Core   : DiagonalKANSSMCore"
echo "  Datamodule : ytv_dav_joint (DAVIS 40% / YTB 60%)"
echo "  LR         : 5e-6 (warmup 10ep from factor 0.05)"
echo "  Sched.samp.: 0.1 → 0.5 over 25 epochs"
echo "  Grad clip  : 0.3"
echo "  Epochs     : 50"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "=========================================="

python train.py \
    model=vision_mamba_tiny_sota_light \
    datamodule=ytv_dav_joint \
    "+pretrained_weights='${CKPT}'" \
    ++trainer.max_epochs=50 \
    ++trainer.accumulate_grad_batches=2 \
    ++trainer.limit_train_batches=3500 \
    ++trainer.gradient_clip_val=0.1 \
    ++logger.name="ft_phase4_ytbdav_50ep_sota_light_v3" \
    ++logger.tags="[diagonal-kan,phase4-v3,from-phase2,davis40pct,stabilised-ft,low-lr,gradual-curriculum,cv-tracking,frozen-backbone]" \
    ++callbacks.model_checkpoint.monitor=val_J_and_F \
    ++callbacks.model_checkpoint.mode=max \
    ++model.learning_rate=5e-6 \
    ++model.lr_warmup_epochs=10 \
    ++model.lr_warmup_start_factor=0.05 \
    ++model.scheduled_sampling_start=0.1 \
    ++model.scheduled_sampling_end=0.5 \
    ++model.scheduled_sampling_warmup_epochs=25 \
    ++datamodule.davis_sampling_ratio=0.40 \
    "+callbacks.freeze_backbone._target_=training.callbacks.FreezeBackboneCallback" \
    "+callbacks.freeze_backbone.module_names=[feature_extractor]" \
    "+callbacks.per_sample_gradient_tracker._target_=training.callbacks.PerSampleGradientTracker" \
    "+callbacks.per_sample_gradient_tracker.output_dir=${PROJECT_ROOT}/artifacts/gradient_tracking" \
    "+callbacks.per_sample_gradient_tracker.log_every_n_steps=50" \
    "+callbacks.layer_gradient_norm_tracker._target_=training.callbacks.LayerGradientNormTracker" \
    "+callbacks.layer_gradient_norm_tracker.output_dir=${PROJECT_ROOT}/artifacts/gradient_norms" \
    "+callbacks.layer_gradient_norm_tracker.log_every_n_steps=50" \
    "+callbacks.per_sample_loss_trajectory_tracker._target_=training.callbacks.PerSampleLossTrajectoryTracker" \
    "+callbacks.per_sample_loss_trajectory_tracker.output_dir=${PROJECT_ROOT}/artifacts/loss_trajectories" \
    "+callbacks.per_sample_loss_trajectory_tracker.log_every_n_steps=50"

echo ""
echo "=========================================="
echo "  Phase 4 finished at $(date)"
echo "  Check Comet for run: ft_phase4_ytbdav_50ep_sota_light_v2"
echo "=========================================="
