#!/bin/bash
#SBATCH --job-name=xattn_smoke
#SBATCH --output=logs/xattn_smoke_%j.out
#SBATCH --error=logs/xattn_smoke_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

export PYTHONPATH=$PYTHONPATH:.
source venv/bin/activate

# Run a very short training run on DAVIS to verify the cross-attention bridge
python train.py \
    experiment=davis \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=2 \
    trainer.limit_val_batches=2 \
    data.batch_size=1 \
    model.fusion_mode=kan_cross_attn \
    +model.scheduled_sampling_rate=0.5 \
    +model.use_ref_context=True \
    logger=none
