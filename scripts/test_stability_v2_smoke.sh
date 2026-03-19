#!/bin/bash
#SBATCH --job-name=smoke_test_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/smoke_test_v2_%j.out
#SBATCH --error=logs/slurm/smoke_test_v2_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"
source venv/bin/activate
export PYTHONPATH=.

echo "Running stability_fix_v2 smoke tests on $(hostname) (GPU: $CUDA_VISIBLE_DEVICES)"
python scripts/test_stability_v2_smoke.py

echo ""
echo "Running COCO visualization (coherent motion)..."
python scripts/vis_coco_pretrain.py

echo ""
echo "All tests completed at: $(date)"
