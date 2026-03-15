#!/bin/bash
#SBATCH --job-name=pretrain_coco_30ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/pretrain_coco_30ep_%j.out
#SBATCH --error=logs/slurm/pretrain_coco_30ep_%j.err

# ============================================================
# COCO Static-Image Pre-training (real instance masks) — Resumed to 30 epochs
#
# Current state: best_coco.ckpt = epoch=2 (only 3 epochs done)
# This script resumes from that checkpoint and trains to 30 ep,
# giving the propagation / memory modules enough time to converge
# before fine-tuning on YouTube-VOS.
#
# Key improvements vs original 15-ep script:
#   - Resumes from checkpoints/best_coco.ckpt (epoch=2)
#   - Trains to 30 epochs total (27 more epochs)
#   - Resolution 480 to match YouTube-VOS native resolution
#   - SSR 0.15: model starts practising own predictions
#   - Uses official COCO instance masks (polygon/RLE via pycocotools)
#   - Saves best val_loss checkpoint to checkpoints/best_coco_30ep.ckpt
#
# After completion:
#   sbatch scripts/finetune_ytvos_v2.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

RESUME_CKPT="${PROJECT_ROOT}/checkpoints/best_coco.ckpt"

echo "=========================================="
echo "  Video Kamba: COCO Pre-training (resume → 30 ep)"
echo "  Resuming from : $RESUME_CKPT (epoch=2)"
echo "  Target epochs : 30"
echo "  LR            : 2e-4"
echo "  img_size      : 480"
echo "  SSR           : 0.15"
echo "  Job ID        : ${SLURM_JOB_ID:-manual}"
echo "  Node          : $(hostname)"
echo "  Start         : $(date)"
echo "=========================================="

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: Checkpoint not found at $RESUME_CKPT"
    exit 1
fi

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm checkpoints

COCO_DATA_DIR="${PROJECT_ROOT}/data/coco"
if [ ! -f "${COCO_DATA_DIR}/annotations/instances_train2017.json" ]; then
    echo "ERROR: COCO instance annotations not found in ${COCO_DATA_DIR}"
    echo "Run: sbatch scripts/download_coco.sh"
    exit 1
fi

# limit_train_batches caps each epoch to 5000 batches (~20k images).
# This reduces epoch time from ~8h to ~30 min after the tensor-augment fix,
# while still covering the full COCO dataset across 30 epochs (30 × 20k = 600k passes).
python train.py \
    model=default \
    datamodule=coco_pretrain \
    ++model.encoder_type=dino \
    ++model.target_size=480 \
    ++model.max_epochs=30 \
    ++model.learning_rate=2e-4 \
    ++model.scheduled_sampling_rate=0.15 \
    ++model.use_kan_key_adapter=true \
    ++trainer.max_epochs=30 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++trainer.limit_train_batches=5000 \
    ++logger.name="coco_pretrain_30ep" \
    ++callbacks.monitor=val_loss \
    ++callbacks.mode=min \
    ++datamodule.num_workers=8 \
    ++datamodule.data_dir="${COCO_DATA_DIR}" \
    +checkpoint="${RESUME_CKPT}" \
    "$@"

echo ""
echo "COCO 30-epoch pre-training done: $(date)"
echo "Checkpoint saved to: checkpoints/best_coco_30ep.ckpt (via ModelCheckpoint callback)"
echo "Next: sbatch scripts/finetune_ytvos_v2.sh"
