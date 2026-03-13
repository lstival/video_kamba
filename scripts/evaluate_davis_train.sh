#!/bin/bash
#SBATCH --job-name=eval_davis_train
#SBATCH --output=logs/slurm/eval_davis_train_%j.out
#SBATCH --error=logs/slurm/eval_davis_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==========================================
# Video Mamba DAVIS Train Evaluation
# ==========================================

echo "=========================================="
echo "  Video Mamba DAVIS TRAIN Evaluation"
echo "  Job ID    : ${SLURM_JOB_ID:-manual}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Checkpoint path logic (same as test script)
CKPT_PATH="/home/WUR/stiva001/WUR/video_kamba/logs/video_mamba/9490650e134e45d4972c02e5a40c54e5/checkpoints/epoch=99-step=22100.ckpt"

# Workaround for '=' in path
if [[ "$CKPT_PATH" == *"="* ]]; then
    TMP_CKPT="eval_tmp_train_${SLURM_JOB_ID:-manual}.ckpt"
    ln -sf "$CKPT_PATH" "$TMP_CKPT"
    CKPT_ARG="checkpoint=$(pwd)/$TMP_CKPT"
else
    CKPT_ARG="checkpoint=$CKPT_PATH"
fi

# 1. Run Metrics Evaluation on 'train' split
echo -e "\n[1/2] Running Metrics Evaluation (DAVIS TRAIN)..."
python scripts/eval_metrics.py $CKPT_ARG datamodule=davis ++datamodule.test_split=train ++output_subdir=train

# 2. Run Visuals Evaluation for Top 5 Train
echo -e "\n[2/2] Running Visuals Evaluation (DAVIS TRAIN)..."
python scripts/eval_visuals.py $CKPT_ARG datamodule=davis ++datamodule.test_split=train ++output_subdir=train

# Cleanup
if [[ -f "$TMP_CKPT" ]]; then
    rm "$TMP_CKPT"
fi

echo -e "\nFinished at: $(date)"
