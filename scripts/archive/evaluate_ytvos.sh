#!/bin/bash
#SBATCH --job-name=eval_ytvos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_ytvos_%j.out
#SBATCH --error=logs/slurm/eval_ytvos_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  YouTube-VOS Evaluation (Seen/Unseen Split)"
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

# Workaround for Hydra parsing '=' in paths (copied from evaluate.sh)
CLEAN_ARGS=()
for arg in "$@"; do
    if [[ $arg == checkpoint=* ]]; then
        CKPT_PATH="${arg#checkpoint=}"
        CKPT_PATH="${CKPT_PATH#\'}"
        CKPT_PATH="${CKPT_PATH%\'}"
        CKPT_PATH="${CKPT_PATH#\"}"
        CKPT_PATH="${CKPT_PATH%\"}"
        
        if [[ $CKPT_PATH == *"="* ]]; then
            echo "Detected '=' in checkpoint path. Creating symlink workaround..."
            TEMP_CKPT="eval_tmp_yt_$(date +%s).ckpt"
            ln -sf "$(realpath "$CKPT_PATH")" "$TEMP_CKPT"
            CLEAN_ARGS+=("checkpoint=$PROJECT_ROOT/$TEMP_CKPT")
        else
            CLEAN_ARGS+=("checkpoint=$(realpath "$CKPT_PATH")")
        fi
    else
        CLEAN_ARGS+=("$arg")
    fi
done

# Check if datamodule is provided, otherwise default to youtubevos
if [[ ! "$*" == *"datamodule="* ]]; then
    CLEAN_ARGS+=("datamodule=youtubevos")
fi

python scripts/eval_youtubevos.py "${CLEAN_ARGS[@]}"

# Cleanup temporary symlink
for arg in "${CLEAN_ARGS[@]}"; do
    if [[ $arg == checkpoint=$PROJECT_ROOT/eval_tmp_yt_* ]]; then
        rm "${arg#checkpoint=}"
    fi
done

echo ""
echo "Evaluation Finished at $(date)"
echo "=========================================="
