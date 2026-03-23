#!/bin/bash
#SBATCH --job-name=cleanup_old_coco
#SBATCH --output=logs/slurm/cleanup_old_coco_%j.out
#SBATCH --error=logs/slurm/cleanup_old_coco_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "${PROJECT_ROOT}"

NEW_COCO_ROOT="${PROJECT_ROOT}/data/coco"

REQUIRED_PATHS=(
    "${NEW_COCO_ROOT}/annotations/instances_train2017.json"
    "${NEW_COCO_ROOT}/annotations/instances_val2017.json"
    "${NEW_COCO_ROOT}/train2017"
    "${NEW_COCO_ROOT}/val2017"
)

for p in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: New COCO dataset is incomplete. Missing: ${p}"
        echo "Cleanup aborted to avoid data loss."
        exit 1
    fi
done

OLD_LOCAL_CACHE="${PROJECT_ROOT}/data/coco_cache"
OLD_HF_DATASETS="${PROJECT_ROOT}/.cache/huggingface/datasets/detection-datasets___coco"
OLD_HF_HUB="${PROJECT_ROOT}/.cache/huggingface/hub/datasets--detection-datasets--coco"

mkdir -p logs/slurm

echo "=== Cleaning old COCO cache artifacts ==="
echo "Start: $(date)"
echo "Project root: ${PROJECT_ROOT}"

echo "--- Size before cleanup ---"
du -sh "${OLD_LOCAL_CACHE}" "${OLD_HF_DATASETS}" "${OLD_HF_HUB}" 2>/dev/null || true

echo "--- Removing old cache directories ---"
rm -rf "${OLD_LOCAL_CACHE}"
rm -rf "${OLD_HF_DATASETS}"
rm -rf "${OLD_HF_HUB}"

echo "--- Size after cleanup ---"
du -sh data .cache/huggingface 2>/dev/null || true

quota -vs stiva001 || true

echo "Cleanup finished: $(date)"
