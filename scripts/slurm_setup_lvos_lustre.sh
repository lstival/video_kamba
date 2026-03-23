#!/bin/bash
#SBATCH --job-name=lvos_dl_lustre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/lvos_setup_%j.out

# ============================================================
#  LVOS V2 FULL DATASET — Lustre Download + Extract
#  Target: /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/LVOS
#
#  Downloads train (2 image parts + annotations, ~32 GB) and
#  valid (images + annotations, ~12.5 GB) from Google Drive via gdown.
#  Each zip is deleted after extraction to keep peak disk ~25 GB.
#
#  To download a subset only, set SPLITS below, e.g.:
#    SPLITS="valid"          # validation set only
#    SPLITS="train valid"    # both (default)
#
#  Submit:
#    sbatch scripts/slurm_setup_lvos_lustre.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

echo "=========================================="
echo "  LVOS V2 Download to Lustre"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

# ── Activate venv ──────────────────────────────────────────
source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# ── Paths ──────────────────────────────────────────────────
LUSTRE_BASE="/lustre/scratch/WUR/AIN/stiva001/video_kamba_data"
DOWNLOAD_DIR="${LUSTRE_BASE}/lvos_zips"    # Temporary zip storage
EXTRACT_DIR="${LUSTRE_BASE}/LVOS"          # Final dataset location
SYMLINK_TARGET="${PROJECT_ROOT}/data/LVOS" # Project symlink → Lustre

echo "  Zips (temp)  : ${DOWNLOAD_DIR}"
echo "  Dataset      : ${EXTRACT_DIR}"
echo "  Project link : ${SYMLINK_TARGET}"
echo ""

# ── Splits to download ──────────────────────────────────────
# Edit or override: SPLITS="valid" sbatch scripts/slurm_setup_lvos_lustre.sh
SPLITS="${SPLITS:-train valid}"

echo "  Splits : ${SPLITS}"
echo ""

# ── Ensure Lustre directories exist ────────────────────────
mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_DIR}"

# ── Create symlink from project data/ to Lustre ────────────
if [ ! -L "${SYMLINK_TARGET}" ]; then
    if [ -d "${SYMLINK_TARGET}" ]; then
        echo "WARNING: ${SYMLINK_TARGET} is a real directory (not a symlink). Leaving as-is."
    else
        ln -s "${EXTRACT_DIR}" "${SYMLINK_TARGET}"
        echo "Created symlink: ${SYMLINK_TARGET} → ${EXTRACT_DIR}"
    fi
else
    echo "Symlink already exists: ${SYMLINK_TARGET} → $(readlink "${SYMLINK_TARGET}")"
fi

echo ""

# ── Run download + extraction ───────────────────────────────
# shellcheck disable=SC2086
python scripts/download_lvos.py \
    --splits       ${SPLITS} \
    --download-dir "${DOWNLOAD_DIR}" \
    --extract-dir  "${EXTRACT_DIR}" \
    --cleanup

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "  Dataset at: ${EXTRACT_DIR}"
echo "=========================================="
