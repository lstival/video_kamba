#!/bin/bash
#SBATCH --job-name=ft_ade_davis_50ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/ft_ade_davis_50ep_%j.out
#SBATCH --error=logs/slurm/ft_ade_davis_50ep_%j.err

# Fine-tune Vision-Mamba Tiny (Pretrained on ADE20K) on DAVIS 2017 for 50 epochs.
# Pretrain Job: 65612878

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

CKPT="/home/WUR/stiva001/WUR/video_kamba/checkpoints/ade20k_ft_base/ade20k_epoch27_step140000.ckpt"

echo "=========================================="
echo "  Video Kamba: Fine-tuning ADE20K -> DAVIS"
echo "  Checkpoint: $CKPT"
echo "  Target: 50 Epochs"
echo "=========================================="

# Start fine-tuning from the ADE20K pre-trained checkpoint
python train.py \
    model=vision_mamba_tiny \
    datamodule=davis \
    +checkpoint="$CKPT" \
    ++model.learning_rate=1e-4 \
    ++trainer.max_epochs=50 \
    ++trainer.accumulate_grad_batches=2 \
    ++datamodule.img_size=448 \
    ++datamodule.augment_train=True \
    ++logger.name="ft_ade_to_davis_50ep_final" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max

echo "Finished DAVIS fine-tuning."
