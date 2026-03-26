#!/bin/bash
#SBATCH --job-name=mlp_abl_phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/mlp_abl_phase1_%j.out
#SBATCH --error=logs/slurm/mlp_abl_phase1_%j.err

# ═══════════════════════════════════════════════════════════════════════════════
#  MLP Ablation — Phase 1: COCO Pre-training
#
#  Trains the MLP ablation baseline (MV2 + MLPModulator + ConcatUpBlock)
#  on COCO pseudo-video clips.  Identical hyperparameters to the KAN model's
#  Phase 1 (mv2_ssm_mem_phase1_coco) — only the modulation mechanism differs.
#
#  Output:
#    checkpoints/mlp_ablation/best_mlp_abl_phase1_coco.ckpt
#
#  Next step (after gate passes):
#    sbatch scripts/slurm_train_mlp_abl_phase2_davis.sh
#
#  Gate: val_loss < 0.50 @ epoch 10  →  submit Phase 2
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

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
echo "  MLP Ablation — Phase 1: COCO"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "  Output : checkpoints/mlp_ablation/best_mlp_abl_phase1_coco.ckpt"
echo "  Gate   : val_loss < 0.50 @ epoch 10"
echo "=========================================="

python train.py \
    datamodule=coco_pretrain \
    +experiment=mv2_mlp_abl_phase1_coco \
    ++logger.name="mlp_abl_phase1_coco" \
    ++logger.tags="[mlp-ablation,phase1,coco,kan-ablation-baseline]" \
    "$@"

echo ""
echo "Phase 1 finished at: $(date)"
echo ""
echo "Gate check: open Comet run 'mlp_abl_phase1_coco' and verify"
echo "  val_loss @ epoch 10 < 0.50 before submitting Phase 2."
echo "  If gate passed: sbatch scripts/slurm_train_mlp_abl_phase2_davis.sh"
