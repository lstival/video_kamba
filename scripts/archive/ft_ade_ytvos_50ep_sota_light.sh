#!/bin/bash
#SBATCH --job-name=ft_ade_ytvos_sota_lw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ft_ade_ytvos_sota_lw_%j.out
#SBATCH --error=logs/slurm/ft_ade_ytvos_sota_lw_%j.err

# Fine-tune ADE20K-pretrained lightweight Vision-Mamba Tiny on YouTube-VOS.
# Keeps model under 5M params while increasing temporal depth (ssm_layers=2)
# and using a scheduled-sampling curriculum for stabler memory updates.

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
echo "  Video Kamba: ADE20K -> YouTube-VOS (light)"
echo "  Checkpoint: $CKPT"
echo "  Model     : vision_mamba_tiny_sota_light"
echo "  Epochs    : 50"
echo "=========================================="

python train.py \
    model=vision_mamba_tiny_sota_light \
    datamodule=youtubevos \
    +checkpoint="$CKPT" \
    ++trainer.max_epochs=50 \
    ++trainer.accumulate_grad_batches=2 \
    ++logger.name="ft_ade_to_ytvos_sota_light" \
    ++callbacks.monitor=val_vos_J_and_F \
    ++callbacks.mode=max

echo "Finished YouTube-VOS fine-tuning (light SOTA preset)."
