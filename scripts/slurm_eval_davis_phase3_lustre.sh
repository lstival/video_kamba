#!/bin/bash
#SBATCH --job-name=eval_davis_p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_davis_p3_%j.out
#SBATCH --error=logs/slurm/eval_davis_p3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leandroteso@gmail.com

# ============================================================
#  DAVIS Phase 3 Evaluation — Full-sequence J&F benchmark
#
#  Evaluates the Phase 3 DAVIS-fine-tuned checkpoint on the
#  DAVIS-2017 val set using the sliding-window SSM protocol.
#
#  Depends on: train_davis_p3 (job 65973683)
#
#  Submit with dependency:
#    sbatch --dependency=afterok:65973683 \
#           scripts/slurm_eval_davis_phase3_lustre.sh
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
echo "  DAVIS Phase 3 Evaluation"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase3_davis.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Phase 3 DAVIS checkpoint not found: ${CKPT}"
    echo "       Ensure train_davis_p3 (job 65973683) completed successfully."
    exit 1
fi
echo "  Checkpoint : ${CKPT}"

DAVIS_DIR="${PROJECT_ROOT}/data/DAVIS/DAVIS"
if [ ! -d "${DAVIS_DIR}/JPEGImages" ]; then
    echo "ERROR: DAVIS data not found at ${DAVIS_DIR}/JPEGImages"
    exit 1
fi
echo "  DAVIS data : ${DAVIS_DIR}"
echo ""

python scripts/eval_davis_full_sequence.py \
    +analysis=davis_full_sequence \
    analysis.checkpoint="${CKPT}" \
    analysis.clip_len=4 \
    analysis.output_size=480 \
    analysis.save_predictions=true \
    analysis.device=cuda \
    analysis.seed=42 \
    analysis.output_dir="${PROJECT_ROOT}/eval_results/davis_p3_phase3"

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
