#!/bin/bash
#SBATCH --job-name=ft_coco_ytbdav
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/ft_coco_ytbdav_%j.out
#SBATCH --error=logs/slurm/ft_coco_ytbdav_%j.err

# Fine-tune COCO-pretrained MobileNetV2+KAN-SSM on joint YouTube-VOS + DAVIS.
# Mirrors the AOT PRE_YTB_DAV protocol:
#   - DAVIS (60 train seqs, dense annotations, hard scenes)
#   - YouTube-VOS (3471 train seqs, sparse annotations, diverse categories)
# WeightedRandomSampler ensures DAVIS = 25 % of every epoch.
#
# Starting checkpoint: COCO pretrain (Comet ID faf0ef46ac7549e189881e18742b0448)
# Usage: sbatch scripts/ft_coco_ytb_dav_50ep.sh

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
export TMPDIR="${PROJECT_ROOT}/tmp"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" logs/slurm

# Best COCO pretrained checkpoint (MobileNetV2+KAN-SSM)
CKPT="${PROJECT_ROOT}/checkpoints/best_coco_vimtiny.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "Error: COCO checkpoint not found at $CKPT"
    exit 1
fi

echo "=========================================="
echo "  Video Kamba: COCO -> YTB+DAV joint FT"
echo "  Comet pretrain: faf0ef46ac7549e189881e18742b0448"
echo "  Checkpoint: $CKPT"
echo "  Model     : vision_mamba_tiny_sota_light"
echo "  Datamodule: ytv_dav_joint (DAVIS 25% / YTB 75%)"
echo "  Epochs    : 50"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

python train.py \
    model=vision_mamba_tiny_sota_light \
    datamodule=ytv_dav_joint \
    +checkpoint="$CKPT" \
    ++trainer.max_epochs=50 \
    ++trainer.accumulate_grad_batches=2 \
    ++logger.name="ft_coco_ytbdav_50ep_mobilenet_kan" \
    ++logger.tags="[mobilenet,kan-ssm,ytb_dav_joint,lightweight]" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    ++model.scheduled_sampling_start=0.3 \
    ++model.scheduled_sampling_end=0.7 \
    ++model.scheduled_sampling_warmup_epochs=10

echo ""
echo "=========================================="
echo "  Fine-tuning finished at $(date)"
echo "  Check Comet for run: ft_coco_ytbdav_50ep_mobilenet_kan"
echo "=========================================="
