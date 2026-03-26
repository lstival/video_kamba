#!/bin/bash
#SBATCH --job-name=mlp_abl_phase2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm/mlp_abl_phase2_%j.out
#SBATCH --error=logs/slurm/mlp_abl_phase2_%j.err

# ═══════════════════════════════════════════════════════════════════════════════
#  MLP Ablation — Phase 2: DAVIS Fine-tuning
#
#  Fine-tunes the MLP ablation baseline on DAVIS 2017 starting from the
#  Phase 1 COCO checkpoint.  Identical hyperparameters to the KAN model's
#  Phase 2 (mv2_ssm_mem_phase2_davis).
#
#  Uses +pretrained_weights (NOT +checkpoint) so that the optimizer state
#  and epoch counter restart from zero.  Phase 1 trained on COCO; Phase 2
#  switches to DAVIS with a new OneCycleLR schedule.
#
#  Prerequisites:
#    checkpoints/mlp_ablation/best_mlp_abl_phase1_coco.ckpt  (Phase 1 output)
#
#  Output:
#    checkpoints/mlp_ablation/best_mlp_abl_phase2_davis.ckpt
#
#  Eval (after gate passes):
#    sbatch scripts/eval_davis_full_sequence.sh \
#      checkpoints/mlp_ablation/best_mlp_abl_phase2_davis.ckpt
#
#  Gate: val_J_and_F > 0.55 @ epoch 50  →  run evaluation
#
#  Budget: 100 epochs × 500 steps × ~12 s/step ≈ 17 h
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

PHASE1_CKPT="${PROJECT_ROOT}/checkpoints/mlp_ablation/best_mlp_abl_phase1_coco.ckpt"

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    echo "       Run Phase 1 first: sbatch scripts/slurm_train_mlp_abl_phase1_coco.sh"
    exit 1
fi

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm checkpoints/mlp_ablation

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "ERROR: COMET_API_KEY not set. Add it to ${PROJECT_ROOT}/.env"
    exit 1
fi

echo "=========================================="
echo "  MLP Ablation — Phase 2: DAVIS"
echo "  Job ID     : ${SLURM_JOB_ID:-manual}"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "  Pretrained : $PHASE1_CKPT"
echo "  Output     : checkpoints/mlp_ablation/best_mlp_abl_phase2_davis.ckpt"
echo "  Gate       : val_J_and_F > 0.55 @ epoch 50"
echo "=========================================="

python train.py \
    datamodule=davis_vos \
    +experiment=mv2_mlp_abl_phase2_davis \
    +pretrained_weights="$PHASE1_CKPT" \
    ++logger.name="mlp_abl_phase2_davis" \
    ++logger.tags="[mlp-ablation,phase2,davis,kan-ablation-baseline]" \
    "$@"

echo ""
echo "Phase 2 finished at: $(date)"
echo ""
echo "Gate check: open Comet run 'mlp_abl_phase2_davis' and verify"
echo "  val_J_and_F @ epoch 50 > 0.55 before running evaluation."
echo ""
echo "If gate passed, run the full-sequence DAVIS benchmark:"
echo "  sbatch scripts/eval_davis_full_sequence.sh \\"
echo "    checkpoints/mlp_ablation/best_mlp_abl_phase2_davis.ckpt"
