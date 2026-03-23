#!/bin/bash
#SBATCH --job-name=v2p2_bl30k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/bl30k_phase2_%j.out

# ============================================================
#  BL30K Phase 2 (Synthetic Video) Training
#  Depends on Phase 1 (COCO) and BL30K download.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# Activate venv
source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# Run training
# Note: config selects best_slim_v2_phase1_coco-v1.ckpt
python train.py \
    datamodule=bl30k \
    +experiment=mv2_ssm_mem_phase2_bl30k \
    trainer.max_epochs=50 \
    trainer.precision="bf16-mixed"
