#!/bin/bash
#SBATCH --job-name=vis_mca_memory
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/vis_mca_memory_%j.out
#SBATCH --error=logs/slurm/vis_mca_memory_%j.err

# ============================================================
# Visualise MCA memory dynamic frame (Fig. 3 style).
#
# Runs vis_mca_memory.py on a set of DAVIS sequences and saves
# one PNG per sequence to vis_mca_memory/.
#
# Set CKPT below to the desired checkpoint before submitting.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# ── Checkpoint — update after Phase 2 finishes ──────────────────────────────
CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase2_davis.ckpt"
if [ ! -f "$CKPT" ]; then
    # Fallback to Phase 1 COCO checkpoint for inspection during Phase 2 training
    CKPT="${PROJECT_ROOT}/checkpoints/best_mv2_phase1_coco.ckpt"
    echo "Warning: Phase 2 checkpoint not found — using Phase 1 COCO checkpoint."
fi

echo "=========================================="
echo "  MCA Memory Visualisation"
echo "  Checkpoint : $CKPT"
echo "  Job ID     : ${SLURM_JOB_ID:-manual}"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.

mkdir -p logs/slurm vis_mca_memory

python scripts/vis_mca_memory.py \
    --checkpoint "$CKPT" \
    --sequences blackswan camel bear bmx-trees breakdance \
    --n_frames 5 \
    --img_size 480 \
    --output vis_mca_memory/ \
    --davis_root data/DAVIS/DAVIS

echo ""
echo "Done at: $(date)"
echo "Output:  vis_mca_memory/"
