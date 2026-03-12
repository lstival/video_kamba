#!/bin/bash
#SBATCH --job-name=download_coco
#SBATCH --output=logs/slurm/download_coco_%j.out
#SBATCH --error=logs/slurm/download_coco_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -e

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "${PROJECT_ROOT}"

echo "=== Starting COCO Download ==="
date
echo "Project Root: ${PROJECT_ROOT}"

# 1. Environment
source venv/bin/activate

# 2. Check current quota
echo "--- Quota Status ---"
quota -vs stiva001 || true

# 3. Create cache directory in project root (to avoid home quota)
mkdir -p data/coco_cache

# 4. Run download script
echo "--- Downloading COCO ---"
python -c "
import os
from datasets import load_dataset
cache_dir = 'data/coco_cache'
print(f'Using cache_dir: {cache_dir}')
# Download and prepare (non-streaming)
try:
    load_dataset(
        'detection-datasets/coco',
        split='train',
        cache_dir=cache_dir,
        streaming=False
    )
    print('COCO train split downloaded successfully.')
    load_dataset(
        'detection-datasets/coco',
        split='val',
        cache_dir=cache_dir,
        streaming=False
    )
    print('COCO val split downloaded successfully.')
except Exception as e:
    print(f'Error during download: {e}')
    raise e
"

echo "--- Download Complete ---"
date
quota -vs stiva001 || true
