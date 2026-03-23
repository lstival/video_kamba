#!/bin/bash
#SBATCH --job-name=train_bl30k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/bl30k_train_%j.out

# ============================================================
#  BL30K Phase 2 Training
#  Loads from last phase 1 checkpoint.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# Load secrets — overrides any stale env vars inherited from the login shell
# (e.g. COMET_API_KEY=your-key-here set in .bashrc)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Run training
# Note: config selects last-v1.ckpt
python train.py \
    datamodule=bl30k \
    +experiment=mv2_ssm_mem_phase2_bl30k \
    trainer.max_epochs=50 \
    trainer.precision="bf16-mixed"
