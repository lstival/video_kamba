#!/bin/bash
#SBATCH --job-name=vimtiny_coco_30ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/vimtiny_coco_30ep_%j.out
#SBATCH --error=logs/slurm/vimtiny_coco_30ep_%j.err

# ============================================================
# Train lightweight Vision-Mamba + KAN on COCO Pretrain (real instance masks).
#
# Highlights:
#   - trainable encoder_type=vision_mamba_tiny
#   - lightweight decoder widths (160/96/48)
#   - KAN-guided spatial Vision-Mamba + KAN temporal SSM
#   - COCO polygon/RLE instance masks (no bbox-filled masks)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: Vision-Mamba Tiny (COCO)"
echo "  Encoder   : vision_mamba_tiny (trainable)"
echo "  Epochs    : 30"
echo "  Target Res: 480x480"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
mkdir -p logs/slurm

COCO_DATA_DIR="${PROJECT_ROOT}/data/coco"

if [ ! -f "${COCO_DATA_DIR}/annotations/instances_train2017.json" ]; then
    echo "Error: COCO annotations not found in ${COCO_DATA_DIR}."
    echo "Run: sbatch scripts/download_coco.sh"
    exit 1
fi

python train.py \
    model=vision_mamba_tiny \
    datamodule=coco_pretrain \
    ++datamodule.data_dir="${COCO_DATA_DIR}" \
    ++model.target_size=480 \
    ++datamodule.img_size=480 \
    ++datamodule.num_workers=8 \
    ++trainer.max_epochs=30 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="vimtiny_coco_30ep" \
    ++callbacks.monitor=val_loss \
    ++callbacks.mode=min \
    "$@"

echo ""
echo "Training finished at: $(date)"
