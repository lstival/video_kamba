#!/bin/bash
#SBATCH --job-name=bl30k_dl_lustre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/bl30k_setup_%j.out

# ============================================================
#  BL30K FULL DATASET — Lustre Download + Extract
#  Target: /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/BL30K
#
#  Downloads all 6 segments (a–f, ~700 GB total) sequentially.
#  Each tar is deleted after extraction to keep peak usage ~120 GB.
#
#  To download a subset only, set SEGMENTS below, e.g.:
#    SEGMENTS="a b"        # just the first two segments
#    SEGMENTS="c d e f"    # resume from segment c
#
#  Submit:
#    sbatch scripts/slurm_setup_bl30k_lustre.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

echo "=========================================="
echo "  BL30K Full Download to Lustre"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

# ── Activate venv ──────────────────────────────────────────
source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# ── Paths ──────────────────────────────────────────────────
LUSTRE_BASE="/lustre/scratch/WUR/AIN/stiva001/video_kamba_data"
DOWNLOAD_DIR="${LUSTRE_BASE}/tars"        # Temporary tar storage (one at a time)
EXTRACT_DIR="${LUSTRE_BASE}/BL30K"        # Final dataset location on Lustre
SYMLINK_TARGET="${PROJECT_ROOT}/data/BL30K"  # Project symlink → Lustre

echo "  Tars (temp)  : ${DOWNLOAD_DIR}"
echo "  Dataset      : ${EXTRACT_DIR}"
echo "  Project link : ${SYMLINK_TARGET}"
echo ""

# ── Segments to download ────────────────────────────────────
# Edit this to download a subset, e.g. SEGMENTS="a b" or SEGMENTS="c d e f"
SEGMENTS="${SEGMENTS:-a b c d e f}"

echo "  Segments  : ${SEGMENTS}"
echo "  Est. size : $(echo "$SEGMENTS" | wc -w | xargs -I{} echo '{} * 120 GB')"
echo ""

# ── Ensure Lustre directories exist ────────────────────────
mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_DIR}"

# ── Create symlink from project data/ to Lustre ────────────
if [ ! -L "${SYMLINK_TARGET}" ]; then
    if [ -d "${SYMLINK_TARGET}" ]; then
        echo "WARNING: ${SYMLINK_TARGET} is a real directory (not a symlink). Leaving it as-is."
    else
        ln -s "${EXTRACT_DIR}" "${SYMLINK_TARGET}"
        echo "Created symlink: ${SYMLINK_TARGET} → ${EXTRACT_DIR}"
    fi
else
    echo "Symlink already exists: ${SYMLINK_TARGET} → $(readlink ${SYMLINK_TARGET})"
fi

echo ""

# ── Run download + extraction ───────────────────────────────
# shellcheck disable=SC2086
python scripts/download_bl30k_subset.py \
    --segments       ${SEGMENTS} \
    --download-dir   "${DOWNLOAD_DIR}" \
    --extract-dir    "${EXTRACT_DIR}" \
    --cleanup

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "  Dataset at: ${EXTRACT_DIR}"
echo "=========================================="
