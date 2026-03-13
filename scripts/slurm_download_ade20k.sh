#!/bin/bash
#SBATCH --job-name=download_ade20k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/download_ade20k_%j.out
#SBATCH --error=logs/slurm/download_ade20k_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Kamba: ADE20K Download"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
# Shared cache location used by pretrain scripts
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

python scripts/download_ade20k.py --cache_dir "${PROJECT_ROOT}/data/ade20k_cache"

echo ""
echo "ADE20K Download complete at $(date)."
