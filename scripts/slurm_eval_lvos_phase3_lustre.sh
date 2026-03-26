#!/bin/bash
#SBATCH --job-name=eval_lvos_p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/eval_lvos_p3_%j.out
#SBATCH --error=logs/slurm/eval_lvos_p3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leandroteso@gmail.com

# ============================================================
#  LVOS Phase 3 Evaluation — LVOS train-set J&F
#
#  Evaluates the Phase 3 LVOS-fine-tuned checkpoint on the
#  LVOS V2 train split using the full-sequence sliding-window
#  SSM protocol (eval_lvos_train.py).
#
#  Metrics are computed only on annotated keyframes; all frames
#  are propagated through the model.
#
#  Depends on: train_lvos_p3 (job 65972789)
#
#  Submit with dependency:
#    sbatch --dependency=afterok:65972789 \
#           scripts/slurm_eval_lvos_phase3_lustre.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

echo "=========================================="
echo "  LVOS Phase 3 Evaluation (LVOS train J&F)"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase3_lvos.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Phase 3 LVOS checkpoint not found: ${CKPT}"
    echo "       Ensure train_lvos_p3 (job 65972789) completed successfully."
    exit 1
fi
echo "  Checkpoint : ${CKPT}"

LVOS_DIR="${PROJECT_ROOT}/data/LVOS"
if [ ! -d "${LVOS_DIR}/train/JPEGImages" ]; then
    echo "ERROR: LVOS train data not found at ${LVOS_DIR}/train/JPEGImages"
    echo "       Run: sbatch scripts/slurm_setup_lvos_lustre.sh"
    exit 1
fi
echo "  LVOS data  : ${LVOS_DIR}"
echo ""

python scripts/eval_lvos_train.py \
    +analysis=lvos_train \
    analysis.checkpoint="${CKPT}" \
    analysis.clip_len=12 \
    analysis.output_size=480 \
    analysis.save_predictions=true \
    analysis.device=cuda \
    analysis.seed=42

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
