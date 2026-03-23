#!/bin/bash
#SBATCH --job-name=train_lvos_p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/lvos_p3_train_%j.out

# ============================================================
#  LVOS V2 Phase 3 Training — Long-range DTSM Fine-tuning
#
#  Requires:
#    1. Phase 1b BL30K checkpoint (best_slim_v2_phase2_bl30k.ckpt)
#    2. LVOS V2 dataset at data/LVOS (or symlinked from Lustre)
#       Run scripts/slurm_setup_lvos_lustre.sh first.
#
#  Submit (depends on Phase 1b job 65781828):
#    sbatch --dependency=afterok:65781828 scripts/slurm_train_lvos_phase3_lustre.sh
#
#  Submit (manual, after Phase 1b completes):
#    sbatch scripts/slurm_train_lvos_phase3_lustre.sh
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
echo "  LVOS Phase 3 Training"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

# Verify Phase 1b checkpoint exists
CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Phase 1b checkpoint not found: ${CKPT}"
    echo "       Ensure Phase 1b (BL30K) training completed before running this job."
    exit 1
fi
echo "  Phase 1b checkpoint : ${CKPT}"

# Verify LVOS data is present
LVOS_DIR="${PROJECT_ROOT}/data/LVOS"
if [ ! -d "${LVOS_DIR}/train/JPEGImages" ]; then
    echo "ERROR: LVOS train data not found at ${LVOS_DIR}/train/JPEGImages"
    echo "       Run: sbatch scripts/slurm_setup_lvos_lustre.sh"
    exit 1
fi
echo "  LVOS data           : ${LVOS_DIR}"
echo ""

# Run Phase 3 training
python train.py \
    datamodule=lvos \
    +experiment=mv2_ssm_mem_phase3_lvos \
    trainer.max_epochs=60 \
    trainer.precision="bf16-mixed"

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
