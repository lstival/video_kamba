#!/bin/bash
#SBATCH --job-name=smoke_kan_key
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/smoke_kan_key_%j.out
#SBATCH --error=logs/slurm/smoke_kan_key_%j.err

# ============================================================
# Smoke test for:
#   1. KANKeyAdapter shapes and residual correctness
#   2. MemoryBank with key_adapter enabled 
#   3. COCOPretrainDataModule batch shapes (using max_samples=20)
#   4. Full VideoMambaSystem forward pass with use_kan_key_adapter=True
#
# Usage:
#   sbatch scripts/smoke_kan_key_adapter.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Smoke Test: KAN Key Adapter + COCO DataModule"
echo "  Job ID : ${SLURM_JOB_ID:-manual}"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "=========================================="

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
mkdir -p logs/slurm

echo "--- Running pytest: test_kan_key_adapter.py ---"
pytest tests/test_kan_key_adapter.py -v --tb=short

echo ""
echo "--- Running pytest: test_vos_shapes.py (with adapter path) ---"
pytest tests/test_vos_shapes.py::TestVideoMambaSystemVOS -v --tb=short

echo ""
echo "--- COCO DataModule smoke (max_samples=20, 1 epoch) ---"
python train.py \
    model=default \
    datamodule=coco_pretrain \
    ++model.encoder_type=dino \
    ++model.target_size=224 \
    ++model.max_epochs=1 \
    ++model.learning_rate=2e-4 \
    ++model.scheduled_sampling_rate=0.1 \
    ++model.use_kan_key_adapter=true \
    ++datamodule.img_size=224 \
    ++datamodule.max_samples=20 \
    ++datamodule.batch_size=2 \
    ++datamodule.num_workers=0 \
    ++trainer.max_epochs=1 \
    ++trainer.limit_train_batches=3 \
    ++trainer.limit_val_batches=2 \
    ++trainer.precision="32" \
    ++logger.name="smoke_coco_kan_key" \
    ++callbacks.monitor=val_loss \
    ++callbacks.mode=min

echo ""
echo "Smoke test PASSED at: $(date)"
