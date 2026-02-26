#!/bin/bash
#SBATCH --job-name=video_mamba_eval_davis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/eval_davis_%j.out
#SBATCH --error=logs/slurm/eval_davis_%j.err

set -euo pipefail

# Usage: sbatch scripts/evaluate_davis.sh checkpoint='path/to/ckpt' datamodule=davis
# Or for manual run: ./scripts/evaluate_davis.sh checkpoint='path/to/ckpt' datamodule=davis

if [ $# -eq 0 ]; then
    echo "Usage: sbatch scripts/evaluate_davis.sh checkpoint='path/to/ckpt' [datamodule=davis] [other hydra options]"
    exit 1
fi

# Determine project root
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Mamba DAVIS Evaluation"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Node      : $(hostname)"
echo "  Args      : $@"
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

# Workaround for Hydra parsing '=' in paths
CLEAN_ARGS=()
for arg in "$@"; do
    if [[ $arg == checkpoint=* ]]; then
        CKPT_PATH="${arg#checkpoint=}"
        if [[ $CKPT_PATH == *"="* ]]; then
            echo "Detected '=' in checkpoint path. Creating symlink workaround..."
            TEMP_CKPT="eval_tmp_davis_$(date +%s).ckpt"
            ln -sf "$(realpath "$CKPT_PATH")" "$TEMP_CKPT"
            CLEAN_ARGS+=("checkpoint=$TEMP_CKPT")
        else
            CLEAN_ARGS+=("$arg")
        fi
    else
        CLEAN_ARGS+=("$arg")
    fi
done

# Check if datamodule is provided, otherwise default to davis
if [[ ! "$*" == *"datamodule="* ]]; then
    CLEAN_ARGS+=("datamodule=davis")
fi

# 1. Run Metrics Evaluation
echo ""
echo "[1/2] Running Metrics Evaluation (DAVIS)..."
python scripts/eval_metrics.py "${CLEAN_ARGS[@]}"

# 2. Run Visuals Evaluation
echo ""
echo "[2/2] Running Visuals Evaluation (DAVIS)..."
python scripts/eval_visuals.py "${CLEAN_ARGS[@]}"

# Cleanup temporary symlink
for arg in "${CLEAN_ARGS[@]}"; do
    if [[ $arg == checkpoint=eval_tmp_davis_* ]]; then
        rm "${arg#checkpoint=}"
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Finished at $(date)"
echo "Check 'eval_results/' for visualizations and metrics."
echo "=========================================="
