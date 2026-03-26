#!/bin/bash
#SBATCH --job-name=train_joint_p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/joint_p3_train_%j.out

# ============================================================
#  Phase 3 — Joint Generalist Fine-tuning
#
#  Trains on DAVIS (15%) + YouTube-VOS (60%) + LVOS (25%)
#  simultaneously from the BL30K Phase 1b checkpoint, using
#  patch_wise_id_bank, DTSM, and scheduled sampling.
#
#  Requires:
#    1. Phase 1b BL30K checkpoint:
#         checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k-v2.ckpt
#    2. DAVIS at  data/DAVIS/DAVIS
#    3. YouTube-VOS at data/YouTubeVOS
#    4. LVOS V2 at data/LVOS
#
#  Gate decision at epoch 30 (check Comet):
#    val_J_and_F > 0.62 → continue to epoch 60
#    val_J_and_F < 0.55 → check scheduled-sampling ramp & id_bank LR
#
#  Submit (immediately):
#    sbatch scripts/slurm_train_phase3_joint_lustre.sh
#
#  Submit (depends on a prior job, e.g. Phase 1b 65781828):
#    sbatch --dependency=afterok:65781828 scripts/slurm_train_phase3_joint_lustre.sh
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
echo "  Phase 3 Joint Training"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

# ── Verify Phase 1b checkpoint ─────────────────────────────────────────────
CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k-v2.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Phase 1b checkpoint not found: ${CKPT}"
    echo "       Ensure Phase 1b (BL30K) training completed before running this job."
    exit 1
fi
echo "  Phase 1b checkpoint : ${CKPT}"

# ── Verify datasets ────────────────────────────────────────────────────────
DAVIS_DIR="${PROJECT_ROOT}/data/DAVIS/DAVIS"
if [ ! -d "${DAVIS_DIR}/JPEGImages" ]; then
    echo "ERROR: DAVIS data not found at ${DAVIS_DIR}/JPEGImages"
    exit 1
fi
echo "  DAVIS               : ${DAVIS_DIR}"

YTB_DIR="${PROJECT_ROOT}/data/YouTubeVOS"
if [ ! -d "${YTB_DIR}/train/JPEGImages" ]; then
    echo "ERROR: YouTube-VOS data not found at ${YTB_DIR}/train/JPEGImages"
    exit 1
fi
echo "  YouTube-VOS         : ${YTB_DIR}"

LVOS_DIR="${PROJECT_ROOT}/data/LVOS"
if [ ! -d "${LVOS_DIR}/train/JPEGImages" ]; then
    echo "ERROR: LVOS train data not found at ${LVOS_DIR}/train/JPEGImages"
    echo "       Run: sbatch scripts/slurm_setup_lvos_lustre.sh"
    exit 1
fi
echo "  LVOS                : ${LVOS_DIR}"
echo ""

# ── Run Phase 3 joint training ─────────────────────────────────────────────
# The experiment config sets:
#   datamodule=phase3_joint  (DAVIS 15% / YTB 60% / LVOS 25%)
#   model=mv2_ssm_memory     (patch_wise_id_bank + DTSM + scheduled sampling)
#   max_epochs=60, limit_train_batches=2500, accumulate_grad_batches=2
#   Gate: val_J_and_F > 0.62 at epoch 30
python train.py \
    +experiment=mv2_ssm_mem_phase3_joint \
    pretrained_weights="${CKPT}"

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
