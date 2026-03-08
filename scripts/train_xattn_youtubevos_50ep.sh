#!/bin/bash
#SBATCH --job-name=train_xattn_ytvos
#SBATCH --output=logs/cross_attention_bridge/youtubevos_%j.out
#SBATCH --error=logs/cross_attention_bridge/youtubevos_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

export PYTHONPATH=$PYTHONPATH:.
source venv/bin/activate

mkdir -p logs/cross_attention_bridge

python train.py \
    datamodule=youtubevos \
    model=default \
    trainer.max_epochs=50 \
    model.fusion_mode=kan_cross_attn \
    model.scheduled_sampling_rate=0.5 \
    +model.use_ref_context=True
