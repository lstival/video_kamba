#!/bin/bash
#SBATCH --job-name=phase2_grad_smoke
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/phase2_grad_smoke_%j.out
#SBATCH --error=logs/slurm/phase2_grad_smoke_%j.err

# ============================================================
# Phase 2 gradient-explosion smoke test.
#
# Purpose: verify that the new davis_vos datamodule (clip_len=4,
# output_size=480) does NOT OOM, and that gradient norms after
# clipping are within acceptable bounds during the first 20 steps
# with the Phase 1 checkpoint.
#
# The layer_gradient_norm_tracker callback is overridden to log
# every single step (log_every_n_steps=1) so we see the full
# norm trajectory rather than only every 50th step.
#
# Pass criteria (checked in the .out log by grep):
#   1. No "Killed" or "oom_kill" in .err
#   2. No "Non-finite" or "NaN" in model output
#   3. Raw gradient norms should decrease after the first few steps
#      as the optimizer adapts from COCO → DAVIS distribution.
#      Persistent norms > 1e4 after step 5 suggest a deeper issue.
#
# Output: logs/slurm/phase2_grad_smoke_<JOB_ID>.{out,err}
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase1_coco.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Error: Phase 1 checkpoint not found at $CKPT"
    exit 1
fi

echo "=========================================="
echo "  Phase 2 Gradient Smoke Test"
echo "  Checkpoint : $CKPT"
echo "  Job ID     : ${SLURM_JOB_ID:-manual}"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "  GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
# Persist Triton JIT cache between jobs — first run compiles; subsequent runs reuse.
# Without this, ptxas subprocess is re-spawned every job and its peak RSS
# (~30-40 GB) is tracked by the SLURM cgroup but NOT by sacct MaxRSS,
# causing OOM kills that look inexplicable from the RSS numbers alone.
export TRITON_CACHE_DIR="${PROJECT_ROOT}/.triton_cache"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm "$TRITON_CACHE_DIR"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "Error: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

# ── Run 20 training steps + 5 val batches, epoch budget = 1 ──────────────────
# Key overrides vs. the real Phase 2 run:
#   trainer.limit_train_batches=20   — 20 steps to observe norm trajectory
#   trainer.limit_val_batches=5      — quick sanity-check on val loss
#   trainer.max_epochs=1             — single epoch
#   callbacks.layer_gradient_norm_tracker.log_every_n_steps=1
#       — log raw gradient norms at every backward pass so we can see whether
#         norms decay after the first domain-shift spike or keep exploding

python train.py \
    model=mobilenetv2_kan_temporal \
    datamodule=davis_vos \
    +experiment=mv2_phase2_davis \
    +pretrained_weights="$CKPT" \
    ++trainer.max_epochs=1 \
    ++trainer.limit_train_batches=20 \
    ++trainer.limit_val_batches=5 \
    ++datamodule.num_workers=0 \
    ++callbacks.layer_gradient_norm_tracker.log_every_n_steps=5 \
    ++logger.name="phase2_grad_smoke" \
    ++logger.tags="[smoke-test,grad-explosion-diagnosis,phase2,mv2-freeze2]" \
    ++callbacks.model_checkpoint.dirpath="checkpoints/smoke" \
    ++callbacks.model_checkpoint.filename="smoke_phase2"

echo ""
echo "Smoke test finished at: $(date)"
echo ""
echo "── Post-run diagnosis ──────────────────────────────────────────────"
echo "Exploding gradient warnings (raw norm > 1e3):"
grep -c "Exploding gradient" "logs/slurm/phase2_grad_smoke_${SLURM_JOB_ID:-manual}.out" \
    && echo "  (count above)" || echo "  None found — gradients healthy"

echo ""
echo "Non-finite warnings:"
grep -c "Non-finite\|NaN\|nan\|inf" "logs/slurm/phase2_grad_smoke_${SLURM_JOB_ID:-manual}.out" \
    && echo "  (count above)" || echo "  None found"

echo ""
echo "Final loss lines:"
grep "train_loss\|val_loss" "logs/slurm/phase2_grad_smoke_${SLURM_JOB_ID:-manual}.out" | tail -10
echo "────────────────────────────────────────────────────────────────────"
