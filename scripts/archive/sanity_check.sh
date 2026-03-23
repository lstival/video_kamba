#!/bin/bash
#SBATCH --job-name=video_mamba_sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/sanity_%j.out
#SBATCH --error=logs/slurm/sanity_%j.err

set -euo pipefail

# Determine project root
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Mamba Sanity Check (Overfit Single Batch)"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Start time: $(date)"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found."
    exit 1
fi

export PYTHONPATH=.

# Run training with overfit_batches=1
# This is a classical diagnostic to ensure the model can learn at least one batch.
# If loss doesn't go to ~0 with 100-200 epochs, there's a bug in the model or loss function.
python train.py \
    trainer.overfit_batches=1 \
    trainer.max_epochs=200 \
    datamodule=hmdb51 \
    datamodule.batch_size=2 \
    logger=null

echo ""
echo "=========================================="
echo "Sanity Check Finished at $(date)"
echo "=========================================="
