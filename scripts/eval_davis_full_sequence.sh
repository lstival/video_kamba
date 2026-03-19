#!/bin/bash
#SBATCH --job-name=eval_davis_full_seq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# ── Full-sequence DAVIS benchmark evaluation (sliding-window SSM) ─────────────
#
# Usage:
#   sbatch scripts/eval_davis_full_sequence.sh \
#     checkpoints/epoch=0-step=1750-v1.ckpt
#
# The positional argument is the checkpoint path (required).
# Paths are resolved relative to the project root.

set -euo pipefail

CHECKPOINT="${1:?Usage: sbatch eval_davis_full_sequence.sh <checkpoint_path>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "[eval_davis_full_sequence] project root : ${PROJECT_ROOT}"
echo "[eval_davis_full_sequence] checkpoint   : ${CHECKPOINT}"
echo "[eval_davis_full_sequence] GPU          : ${CUDA_VISIBLE_DEVICES:-unset}"

cd "${PROJECT_ROOT}"

mkdir -p logs

source activate jackan_gpu

python scripts/eval_davis_full_sequence.py \
    +analysis=davis_full_sequence \
    analysis.checkpoint="${CHECKPOINT}" \
    analysis.clip_len=4 \
    analysis.output_size=480 \
    analysis.save_predictions=true \
    analysis.device=cuda \
    analysis.seed=42

echo "[eval_davis_full_sequence] done."
