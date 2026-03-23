#!/bin/bash
#SBATCH --job-name=pt_ade_mv2_diag_kan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/pt_ade_mv2_diag_kan_%j.out
#SBATCH --error=logs/slurm/pt_ade_mv2_diag_kan_%j.err

# ============================================================
# Pre-training MobileNetV2 + DiagonalKANSSMCore on ADE20K
# Branch: feat/mobilenet-diagonal-kan-ssm
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Pre-train: MobileNetV2 + DiagKAN on ADE20K"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=========================================="

# Activate Python environment
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate video_kamba 2>/dev/null || conda activate base
fi

export PYTHONPATH=.
mkdir -p logs/slurm

python train.py \
    model=mobilenetv2 \
    datamodule=ade20k_pretrain \
    ++datamodule.num_workers=8 \
    ++trainer.max_epochs=100 \
    ++trainer.precision="16-mixed" \
    ++trainer.gradient_clip_val=1.0 \
    ++logger.name="pt_ade_mv2_diag_kan" \
    "$@"

echo ""
echo "Pre-training finished at: $(date)"