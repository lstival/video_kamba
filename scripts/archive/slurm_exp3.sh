#!/bin/bash
#SBATCH --job-name=exp3_rbf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/exp3_%j.out
#SBATCH --error=logs/slurm/exp3_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Experiment 3 — RBF Activation Profile"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Required overrides:
#   kan_ckpt=<path>  e.g. lightning_logs/version_kan/checkpoints/best.ckpt
#
# Example:
#   sbatch scripts/slurm_exp3.sh \
#       kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt

# Workaround for Hydra parsing '=' in paths
CLEAN_ARGS=()
for arg in "$@"; do
    if [[ $arg == *_ckpt=* ]]; then
        KEY="${arg%%=*}"
        VALUE="${arg#*=}"
        # Strip literal single or double quotes if present
        VALUE="${VALUE#\'}"
        VALUE="${VALUE%\'}"
        VALUE="${VALUE#\"}"
        VALUE="${VALUE%\"}"
        
        if [[ $VALUE == *"="* ]]; then
            echo "Detected '=' in $KEY path. Creating symlink workaround..."
            TEMP_CKPT="eval_tmp_${KEY}_$(date +%s).ckpt"
            ln -sf "$(realpath "$VALUE")" "$TEMP_CKPT"
            CLEAN_ARGS+=("$KEY=$PROJECT_ROOT/$TEMP_CKPT")
        else
            CLEAN_ARGS+=("$KEY=$(realpath "$VALUE")")
        fi
    else
        CLEAN_ARGS+=("$arg")
    fi
done

python scripts/exp3_rbf_profile.py \
    data_dir=data/DAVIS/DAVIS \
    output_dir=results/exp3 \
    n_top_channels=5 \
    max_batches=30 \
    seed=42 \
    "${CLEAN_ARGS[@]}"

# Cleanup temporary symlinks
for arg in "${CLEAN_ARGS[@]}"; do
    if [[ $arg == *=*/eval_tmp_*_*.ckpt ]]; then
        VAL="${arg#*=}"
        rm "$VAL"
    fi
done

echo ""
echo "Experiment 3 complete at $(date)."
