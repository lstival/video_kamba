#!/bin/bash
#SBATCH --job-name=eval_full_ytvos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/eval_full_ytvos_%j.out
#SBATCH --error=logs/slurm/eval_full_ytvos_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate
export PYTHONPATH=.

# Workaround for the '=' in path using symlink
CKPT_PATH="/home/WUR/stiva001/WUR/video_kamba/logs/video_mamba/e851ee2d361446b8be7b1fb8875eaf6f/checkpoints/epoch=29-step=52050.ckpt"
TEMP_CKPT="eval_best_yt_$(date +%s).ckpt"
ln -sf "$CKPT_PATH" "$TEMP_CKPT"
CLEAN_CKPT="$PROJECT_ROOT/$TEMP_CKPT"

echo "=========================================="
echo "  YouTube-VOS Full Evaluation Pipeline"
echo "  Checkpoint: $CKPT_PATH"
echo "  Start time: $(date)"
echo "=========================================="

# 1. Quantitative Evaluation (Seen/Unseen)
echo "Step 1: Quantitative Evaluation (Seen/Unseen)..."
python scripts/eval_youtubevos.py checkpoint="$CLEAN_CKPT" datamodule=youtubevos +output_subdir=youtubevos_full

# 2. Metrics Manifest (Top-5 Selection)
echo "Step 2: Metrics Manifest (Top-5 Selection)..."
python scripts/eval_metrics.py checkpoint="$CLEAN_CKPT" datamodule=youtubevos +output_subdir=youtubevos_full

# 3. Visualizations (GIFs)
echo "Step 3: Visualizations (GIFs)..."
# This picks the latest manifest automatically
python scripts/eval_visuals.py checkpoint="$CLEAN_CKPT" datamodule=youtubevos +output_subdir=youtubevos_full


# Cleanup temporary symlink
rm "$TEMP_CKPT"

echo "=========================================="
echo "  Full Evaluation Finished at $(date)"
echo "=========================================="
