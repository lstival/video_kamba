#!/bin/bash
#SBATCH --job-name=ft_ade_ytvos_50ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ft_ade_ytvos_50ep_%j.out
#SBATCH --error=logs/slurm/ft_ade_ytvos_50ep_%j.err

# Fine-tune Vision-Mamba Tiny (Pretrained on ADE20K) on YouTube-VOS 2018/2019 for 50 epochs.
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
echo "  Video Kamba: Fine-tuning ADE20K -> YouTube-VOS"
echo "  Checkpoint: $CKPT"
echo "  Target: 50 Epochs"
echo "=========================================="

# Start fine-tuning from the ADE20K pre-trained checkpoint
python train.py datamodule=youtubevos +checkpoint="$CKPT" \
    ++model.learning_rate=4e-5 \
    ++trainer.max_epochs=50 \
    ++trainer.accumulate_grad_batches=2 \
    ++logger.name="ft_ade_to_ytvos_50ep_final" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max

echo "Finished YouTube-VOS fine-tuning."
