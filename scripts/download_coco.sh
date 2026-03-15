#!/bin/bash
#SBATCH --job-name=download_coco
#SBATCH --output=logs/slurm/download_coco_%j.out
#SBATCH --error=logs/slurm/download_coco_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "${PROJECT_ROOT}"

echo "=== Starting COCO 2017 Instance Download ==="
date
echo "Project Root: ${PROJECT_ROOT}"

# 1. Environment
source venv/bin/activate

# 2. Check current quota
echo "--- Quota Status ---"
quota -vs stiva001 || true

# 3. Prepare directories
COCO_ROOT="${PROJECT_ROOT}/data/coco"
mkdir -p "${COCO_ROOT}" logs/slurm

# 4. COCO official URLs (real instance segmentation)
TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

download_if_missing() {
    local url="$1"
    local out="$2"
    if [ -f "$out" ]; then
        echo "Already exists: $out"
        return 0
    fi
    echo "Downloading: $url"
    wget -c "$url" -O "$out"
}

extract_if_needed() {
    local zip_file="$1"
    local marker="$2"
    if [ -e "$marker" ]; then
        echo "Already extracted: $marker"
        return 0
    fi
    echo "Extracting: $zip_file"
    unzip -q "$zip_file" -d "${COCO_ROOT}"
}

echo "--- Downloading archives ---"
download_if_missing "$TRAIN_URL" "${COCO_ROOT}/train2017.zip"
download_if_missing "$VAL_URL" "${COCO_ROOT}/val2017.zip"
download_if_missing "$ANN_URL" "${COCO_ROOT}/annotations_trainval2017.zip"

echo "--- Extracting archives ---"
extract_if_needed "${COCO_ROOT}/train2017.zip" "${COCO_ROOT}/train2017"
extract_if_needed "${COCO_ROOT}/val2017.zip" "${COCO_ROOT}/val2017"
extract_if_needed "${COCO_ROOT}/annotations_trainval2017.zip" "${COCO_ROOT}/annotations/instances_train2017.json"

echo "--- Verifying expected files ---"
test -f "${COCO_ROOT}/annotations/instances_train2017.json"
test -f "${COCO_ROOT}/annotations/instances_val2017.json"
test -d "${COCO_ROOT}/train2017"
test -d "${COCO_ROOT}/val2017"

echo "COCO instance dataset is ready at: ${COCO_ROOT}"

echo "--- Download Complete ---"
date
quota -vs stiva001 || true
