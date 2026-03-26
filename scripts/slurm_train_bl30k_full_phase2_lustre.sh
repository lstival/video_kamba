#!/bin/bash
#SBATCH --job-name=train_bl30k_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/bl30k_full_p2_train_%j.out

# ============================================================
#  Phase 2 — Full BL30K Synthetic Video Pre-training
#
#  Architecture-fixed run: patch_wise_id_bank + DTSM from
#  Phase 1 COCO checkpoint.  Data lives on Lustre (25K seqs,
#  768×512, ~630 GB total).
#
#  Requires:
#    1. Phase 1 COCO checkpoint:
#         checkpoints/ssm_mem_v2_aot/best_slim_v2_phase1_coco.ckpt
#    2. Full BL30K on Lustre:
#         /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/BL30K
#         (Layout A: BL30K/JPEGImages/<seq_id>/)
#
#  Gate at epoch 25 (check Comet):
#    val_loss < 0.30 → healthy, continue
#    val_loss > 0.45 → check DTSM gradient norms
#
#  Output checkpoint:
#    checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k_full.ckpt
#
#  Submit:
#    sbatch scripts/slurm_train_bl30k_full_phase2_lustre.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
LUSTRE_BL30K="/lustre/scratch/WUR/AIN/stiva001/video_kamba_data/BL30K"

cd "$PROJECT_ROOT"

mkdir -p logs/slurm

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# Load secrets
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

echo "=========================================="
echo "  Phase 2 Full BL30K Training"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="
echo ""

# ── Verify Phase 1 COCO checkpoint ────────────────────────────────────────
CKPT="${PROJECT_ROOT}/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase1_coco.ckpt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Phase 1 COCO checkpoint not found: ${CKPT}"
    exit 1
fi
echo "  Phase 1 COCO checkpoint : ${CKPT}"

# ── Verify BL30K on Lustre ─────────────────────────────────────────────────
BL30K_IMAGES="${LUSTRE_BL30K}/BL30K/JPEGImages"
if [ ! -d "${BL30K_IMAGES}" ]; then
    echo "ERROR: BL30K data not found at ${BL30K_IMAGES}"
    exit 1
fi
SEQ_COUNT=$(ls "${BL30K_IMAGES}" | wc -l)
echo "  BL30K data              : ${LUSTRE_BL30K}"
echo "  Sequences               : ${SEQ_COUNT}"
echo ""

# ── Run Phase 2 BL30K training ────────────────────────────────────────────
# datamodule.data_dir points to Lustre; the datamodule's Layout A detector
# resolves BL30K/JPEGImages and BL30K/Annotations automatically.
python train.py \
    +experiment=mv2_ssm_mem_phase2_bl30k_full \
    datamodule.data_dir="${LUSTRE_BL30K}" \
    pretrained_weights="${CKPT}"

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
