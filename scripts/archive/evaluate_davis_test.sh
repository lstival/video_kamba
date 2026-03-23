#!/bin/bash
#SBATCH --job-name=eval_davis_test
#SBATCH --output=logs/slurm/eval_davis_test_%j.out
#SBATCH --error=logs/slurm/eval_davis_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==========================================
# Video Mamba DAVIS Test Evaluation
# ==========================================

# Print job info
echo "=========================================="
echo "  Video Mamba DAVIS TEST Evaluation"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURM_NODELIST"
echo "  Args      : $@"
echo "  Start time: $(date)"
echo "=========================================="

# Load environment
source venv/bin/activate
export PYTHONPATH=.

# Handle checkpoint path (Hydra/Omegaconf has issues with '=' in paths)
# We check if the passed checkpoint argument contains an '=' and create a symlink if so.
CKPT_PATH=""
for arg in "$@"; do
    if [[ $arg == checkpoint=* ]]; then
        CKPT_PATH="${arg#*=}"
    fi
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: No checkpoint provided."
    exit 1
fi

if [[ "$CKPT_PATH" == *"="* ]]; then
    echo "Detected '=' in checkpoint path. Creating symlink workaround..."
    TMP_CKPT="eval_tmp_test_${SLURM_JOB_ID}.ckpt"
    ln -sf "$CKPT_PATH" "$TMP_CKPT"
    CKPT_ARG="+checkpoint=$(pwd)/$TMP_CKPT"
else
    CKPT_ARG="+checkpoint=$CKPT_PATH"
fi

# 1. Run Metrics Evaluation on 'test-dev' split
echo -e "\n[1/2] Running Metrics Evaluation (DAVIS TEST)..."
python scripts/eval_metrics.py "$CKPT_ARG" datamodule=davis ++datamodule.test_split=test-dev ++output_subdir=test

# 2. Run Visuals Evaluation for Top 5
echo -e "\n[2/2] Running Visuals Evaluation (DAVIS TEST)..."
python scripts/eval_visuals.py $CKPT_ARG datamodule=davis ++datamodule.test_split=test-dev ++output_subdir=test

# Cleanup symlink if created
if [[ -f "$TMP_CKPT" ]]; then
    rm "$TMP_CKPT"
fi

echo -e "\n=========================================="
echo "  Finished at: $(date)"
echo "=========================================="
