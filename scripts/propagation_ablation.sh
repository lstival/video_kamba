#!/bin/bash
#SBATCH --job-name=prop_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/prop_ablation_%j.out
#SBATCH --error=logs/slurm/prop_ablation_%j.err

# ==========================================================
# SOTA Information Propagation Ablation Suite
# Tests 6 key propagation strategies inspired by:
#   - AOT/AOST: Hierarchical object association + ID propagation
#   - XMem: Sensory + Long-term memory banks  
#   - DEVA: Decoupled video segmentation
#   - Cutie: Memory readout with suppression
# ==========================================================

set -euo pipefail

source venv/bin/activate
export PYTHONPATH=.

EPOCHS=10
DATASET=youtubevos

echo "=========================="
echo "  Propagation Ablation"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Dataset: $DATASET"
echo "  Epochs/run: $EPOCHS"
echo "=========================="

# ---- Run 1: Direct Feature Feedback (Option 2) ----
echo -e "\n[1/6] Direct Feature Feedback"
python train.py \
    datamodule=$DATASET \
    trainer.max_epochs=$EPOCHS \
    model.propagation_mode=direct_feature \
    ++logger.experiment_name="prop_direct_feature_${EPOCHS}ep"

# ---- Run 2: Soft-Mask + High-Capacity SSM ----
echo -e "\n[2/6] Soft-Mask + High-Capacity SSM (d_state=64, layers=2)"
python train.py \
    datamodule=$DATASET \
    trainer.max_epochs=$EPOCHS \
    model.propagation_mode=soft_mask \
    model.ssm_d_state=64 \
    model.ssm_layers=2 \
    ++logger.experiment_name="prop_large_ssm_${EPOCHS}ep"

# ---- Run 3: Direct Feature + High-Capacity SSM ----
echo -e "\n[3/6] Direct Feature + High-Capacity SSM"
python train.py \
    datamodule=$DATASET \
    trainer.max_epochs=$EPOCHS \
    model.propagation_mode=direct_feature \
    model.ssm_d_state=64 \
    model.ssm_layers=2 \
    ++logger.experiment_name="prop_direct_large_ssm_${EPOCHS}ep"

# ---- Run 4: Soft-Mask + Identity Concat ----
echo -e "\n[4/6] Soft-Mask + Identity Concat"
python train.py \
    datamodule=$DATASET \
    trainer.max_epochs=$EPOCHS \
    model.propagation_mode=soft_mask \
    model.identity_mode=concat \
    ++logger.experiment_name="prop_softmask_concat_${EPOCHS}ep"

# ---- Run 5: Direct Feature + Identity Concat ----
echo -e "\n[5/6] Direct Feature + Identity Concat"
python train.py \
    datamodule=$DATASET \
    trainer.max_epochs=$EPOCHS \
    model.propagation_mode=direct_feature \
    model.identity_mode=concat \
    ++logger.experiment_name="prop_direct_concat_${EPOCHS}ep"

# ---- Run 6: Full Best Config ----
echo -e "\n[6/6] Full Best Config (Direct + Large SSM + Concat)"
python train.py \
    datamodule=$DATASET \
    trainer.max_epochs=$EPOCHS \
    model.propagation_mode=direct_feature \
    model.ssm_d_state=64 \
    model.ssm_layers=2 \
    model.identity_mode=concat \
    ++logger.experiment_name="prop_full_best_${EPOCHS}ep"

echo -e "\n=========================="
echo "Propagation Ablation Finished at $(date)"
echo "=========================="
