#!/bin/bash
#SBATCH --job-name=rsp_youtubevos_50ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rsp_masked_decoder/youtubevos_%j.out
#SBATCH --error=logs/rsp_masked_decoder/youtubevos_%j.err

mkdir -p logs/rsp_masked_decoder

# Activate environment
source venv/bin/activate

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run training for 50 epochs on YouTube-VOS
python train.py \
    datamodule=youtubevos \
    ++datamodule.num_workers=8 \
    trainer.max_epochs=50 \
    trainer.precision=16-mixed \
    ++trainer.gradient_clip_val=0.5 \
    model.propagation_mode=soft_mask \
    model.learning_rate=1e-4 \
    hydra.run.dir=logs/rsp_masked_decoder/youtubevos_${SLURM_JOB_ID} \
    seed=42
