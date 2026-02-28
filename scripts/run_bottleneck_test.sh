#!/bin/bash
#SBATCH --job-name=vos_bottleneck_diagnostic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/bottleneck_%j.out
#SBATCH --error=logs/slurm/bottleneck_%j.err

# ==========================================
# Video Mamba VOS Bottleneck Isolation
# ==========================================

echo "=========================================="
echo "  VOS Bottleneck Isolation Tests"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

# Checkpoint path logic (workaround for '=' in path)
CKPT_PATH="/home/WUR/stiva001/WUR/video_kamba/logs/video_mamba/bdccdcdfd80243798d620af25df24e4a/checkpoints/epoch=99-step=22100.ckpt"

if [[ "$CKPT_PATH" == *"="* ]]; then
    TMP_CKPT="eval_tmp_btlnck_${SLURM_JOB_ID}.ckpt"
    ln -sf "$CKPT_PATH" "$TMP_CKPT"
    CKPT_ARG="checkpoint=$(pwd)/$TMP_CKPT"
else
    CKPT_ARG="checkpoint=$CKPT_PATH"
fi

# Run the python diagnostic script
python scripts/vos_bottleneck_test.py $CKPT_ARG datamodule=davis

# Cleanup
if [[ -f "$TMP_CKPT" ]]; then
    rm "$TMP_CKPT"
fi

echo "=========================================="
echo "Bottleneck Diagnostic Finished at $(date)"
echo "=========================================="
