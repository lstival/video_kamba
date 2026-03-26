#!/bin/bash
#SBATCH --job-name=eval_ytb_p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/eval_ytb_p3_%j.out
#SBATCH --error=logs/slurm/eval_ytb_p3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leandroteso@gmail.com

# ============================================================
#  YouTube-VOS Phase 3 Evaluation — J&F (Seen / Unseen)
#
#  Evaluates the Phase 3 YouTube-VOS-fine-tuned checkpoint on
#  the YouTube-VOS 2019 val set (seen + unseen categories).
#
#  Depends on: train_ytb_p3 (job 65973684)
#
#  Submit with dependency:
#    sbatch --dependency=afterok:65973684 \
#           scripts/slurm_eval_ytb_phase3_lustre.sh
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
echo "  YouTube-VOS Phase 3 Evaluation"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase3_ytb.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Phase 3 YouTube-VOS checkpoint not found: ${CKPT}"
    echo "       Ensure train_ytb_p3 (job 65973684) completed successfully."
    exit 1
fi
echo "  Checkpoint     : ${CKPT}"

YTV_DIR="${PROJECT_ROOT}/data/YouTubeVOS"
if [ ! -d "${YTV_DIR}/valid/JPEGImages" ]; then
    echo "ERROR: YouTube-VOS val data not found at ${YTV_DIR}/valid/JPEGImages"
    exit 1
fi
echo "  YouTube-VOS data : ${YTV_DIR}"
echo ""

python scripts/eval_youtubevos.py \
    checkpoint="${CKPT}" \
    datamodule=youtubevos \
    ++datamodule.batch_size=4 \
    ++datamodule.num_workers=8 \
    ++datamodule.img_size=480

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
