#!/bin/bash
#SBATCH --job-name=bl30k_dl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/bl30k_download_%j.out
#SBATCH --error=logs/slurm/bl30k_download_%j.err

# ============================================================
#  BL30K Segment 1 — SLURM download + extract job
#
#  Downloads BL30K_a.tar to the scratch space of the node
#  and extracts it directly into the project data directory.
#
#  Submit:
#    sbatch scripts/slurm_download_bl30k.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

echo "=========================================="
echo "  BL30K Download & Extract"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

# ── Activate venv ──────────────────────────────────────────
source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# ── Paths ──────────────────────────────────────────────────
# /tmp has 653G free on this cluster (/scratch does not exist)
SCRATCH_DIR="${TMPDIR:-/tmp/${USER}}"
DOWNLOAD_PATH="${SCRATCH_DIR}/BL30K_a.tar"
EXTRACT_DIR="${PROJECT_ROOT}/data/BL30K"

echo "  Archive   : ${DOWNLOAD_PATH}"
echo "  Extract   : ${EXTRACT_DIR}"
echo ""

# ── Run download + extraction ───────────────────────────────
python scripts/download_bl30k_subset.py \
    --download-path "${DOWNLOAD_PATH}" \
    --extract-dir   "${EXTRACT_DIR}" \
    --cleanup

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
