#!/bin/bash
#SBATCH --job-name=video_mamba_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err

set -euo pipefail

# Usage: sbatch scripts/evaluate.sh checkpoint='path/to/ckpt' datamodule=hmdb51
# Or for manual run: ./scripts/evaluate.sh checkpoint='path/to/ckpt' datamodule=hmdb51

if [ $# -eq 0 ]; then
    echo "Usage: sbatch scripts/evaluate.sh checkpoint='path/to/ckpt' [datamodule=hmdb51] [other hydra options]"
    exit 1
fi

# Determine project root
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Video Mamba Evaluation (Hydra Mode)"
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

# 1. Run Metrics Evaluation
echo ""
echo "[1/2] Running Metrics Evaluation..."

# Workaround for Hydra parsing '=' in paths
CLEAN_ARGS=()
for arg in "$@"; do
    if [[ $arg == checkpoint=* ]]; then
        CKPT_PATH="${arg#checkpoint=}"
        if [[ $CKPT_PATH == *"="* ]]; then
            echo "Detected '=' in checkpoint path. Creating symlink workaround..."
            TEMP_CKPT="eval_tmp_$(date +%s).ckpt"
            ln -sf "$(realpath "$CKPT_PATH")" "$TEMP_CKPT"
            CLEAN_ARGS+=("checkpoint=$TEMP_CKPT")
        else
            CLEAN_ARGS+=("$arg")
        fi
    else
        CLEAN_ARGS+=("$arg")
    fi
done

# Check if datamodule is provided, otherwise default to hmdb51
if [[ ! "$*" == *"datamodule="* ]]; then
    CLEAN_ARGS+=("datamodule=hmdb51")
fi

python scripts/eval_metrics.py "${CLEAN_ARGS[@]}"

# 2. Run Visuals Evaluation
echo ""
echo "[2/2] Running Visuals Evaluation..."
python scripts/eval_visuals.py "${CLEAN_ARGS[@]}"

# Cleanup temporary symlink
for arg in "${CLEAN_ARGS[@]}"; do
    if [[ $arg == checkpoint=eval_tmp_* ]]; then
        rm "${arg#checkpoint=}"
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Finished at $(date)"
echo "Check 'eval_results/' for visualizations and metrics."
echo "=========================================="
