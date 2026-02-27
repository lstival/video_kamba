#!/bin/bash
#SBATCH --job-name=vos_diagnostic_suite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/diagnostic_%j.out
#SBATCH --error=logs/slurm/diagnostic_%j.err

# ==========================================
# Video Mamba VOS Diagnostic Suite
# Runs 4 ablations for 5 epochs each
# ==========================================

echo "=========================================="
echo "  VOS Diagnostic Suite Started"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

EPOCHS=5

# 1. Run A: Baseline (Temporal + Context + Regularization)
echo -e "\n[1/3] Running Run A: Baseline (Context ON)..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.use_ref_context=True \
    ++logger.name="diag_baseline"

# 2. Run B: Spatial Only (Context OFF)
# This checks if the SSM/Reference frame is actually contributing.
echo -e "\n[2/3] Running Run B: Spatial Only (Context OFF)..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.use_ref_context=False \
    ++logger.name="diag_no_context"

# 3. Run C: Dense State (Temporal Capacity Test)
# This checks if increasing memory helps.
echo -e "\n[3/3] Running Run C: Temporal Capacity (d_state=32)..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.temporal_model.d_state=32 \
    ++logger.name="diag_high_capacity"

echo -e "\n=========================================="
echo "Diagnostic Suite Finished at $(date)"
echo "=========================================="
