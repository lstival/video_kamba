#!/bin/bash
#SBATCH --job-name=train_ytb_p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/ytb_p3_train_%j.out

# ============================================================
#  YouTube-VOS Phase 3 Training — Fine-tuning from BL30K
#
#  Branches from best_slim_v2_phase2_bl30k-v2.ckpt (job 65781828).
#  Parallel to LVOS and DAVIS Phase 3 branches.
#
#  Submit:
#    sbatch scripts/slurm_train_ytb_phase3_lustre.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# Load secrets — overrides any stale env vars inherited from the login shell
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

echo "=========================================="
echo "  YouTube-VOS Phase 3 Training"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

# Verify BL30K checkpoint exists
CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k-v2.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: BL30K checkpoint not found: ${CKPT}"
    echo "       Ensure BL30K Phase 1b training (job 65781828) completed."
    exit 1
fi
echo "  BL30K checkpoint : ${CKPT}"

# Verify YouTube-VOS data is present
YTV_DIR="${PROJECT_ROOT}/data/YouTubeVOS"
if [ ! -d "${YTV_DIR}/train/JPEGImages" ]; then
    echo "ERROR: YouTube-VOS train data not found at ${YTV_DIR}/train/JPEGImages"
    exit 1
fi
echo "  YouTube-VOS data : ${YTV_DIR}"
echo ""

# Run Phase 3 training
python train.py \
    datamodule=youtubevos \
    +experiment=mv2_ssm_mem_phase3_ytb \
    pretrained_weights="${CKPT}" \
    trainer.max_epochs=50 \
    trainer.precision="bf16-mixed"

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
