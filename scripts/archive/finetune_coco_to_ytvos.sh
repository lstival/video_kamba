#!/bin/bash
#SBATCH --job-name=ft_coco_ytvos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ft_coco_ytvos_%j.out
#SBATCH --error=logs/slurm/ft_coco_ytvos_%j.err

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

CKPT="/home/WUR/stiva001/WUR/video_kamba/checkpoints/best_coco.ckpt"

echo "=========================================="
echo "  Video Kamba: Fine-tuning COCO -> YouTube-VOS"
echo "  Checkpoint: $CKPT"
echo "=========================================="

python train.py \
    model=default \
    datamodule=youtubevos \
    ++model.encoder_type=dino \
    ++model.target_size=448 \
    ++model.max_epochs=30 \
    ++model.learning_rate=2e-5 \
    ++model.scheduled_sampling_rate=0.3 \
    ++datamodule.img_size=448 \
    ++datamodule.num_workers=8 \
    ++trainer.max_epochs=30 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=0.5 \
    ++logger.name="ft_coco_to_ytvos" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    +checkpoint="$CKPT"

echo "Finished YouTube-VOS fine-tuning."
