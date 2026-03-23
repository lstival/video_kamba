#!/bin/bash
#SBATCH --job-name=ft_coco_davis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/ft_coco_davis_%j.out
#SBATCH --error=logs/slurm/ft_coco_davis_%j.err

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
echo "  Video Kamba: Fine-tuning COCO -> DAVIS"
echo "  Checkpoint: $CKPT"
echo "=========================================="

python train.py \
    datamodule=davis \
    trainer.max_epochs=100 \
    trainer.precision=16-mixed \
    trainer.gradient_clip_val=1.0 \
    model.learning_rate=1e-4 \
    model.vos_loss_beta=0.2 \
    +model.consistency_weight=0.1 \
    ++datamodule.augment_train=true \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max \
    ++logger.project="video-mamba" \
    ++logger.name="ft_coco_to_davis_v3" \
    +checkpoint="$CKPT"


echo "Finished DAVIS fine-tuning."
