#!/bin/bash
# scripts/smoke_curriculum_ssm_mem_v2.sh

set -euo pipefail

mkdir -p logs/slurm/smoke_v2
export PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv not found." && exit 1
fi

echo "=========================================="
echo "  SSM-Memory V2 (AOT) Curriculum Smoke Test"
echo "=========================================="

echo "1) Submitting Phase 1 (COCO)..."
JOB1=$(sbatch --parsable << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_v2_p1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/smoke_v2/phase1_%j.out

source venv/bin/activate
export PYTHONPATH=.
python train.py \
    datamodule=coco_pretrain \
    +experiment=mv2_ssm_mem_phase1_coco \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    +trainer.limit_val_batches=5 \
    ++logger.name="smoke_v2_phase1" \
    ++callbacks.model_checkpoint.dirpath="checkpoints/smoke_v2" \
    ++callbacks.model_checkpoint.filename="best_mv2_ssmmem_phase1_coco_v2"
EOF
)
echo "   -> Phase 1 Job ID: $JOB1"

echo "2) Submitting Phase 2 (DAVIS) depending on $JOB1..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_v2_p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/smoke_v2/phase2_%j.out

source venv/bin/activate
export PYTHONPATH=.
python train.py \
    datamodule=davis_vos \
    +experiment=mv2_ssm_mem_phase2_davis \
    +checkpoint="checkpoints/smoke_v2/best_mv2_ssmmem_phase1_coco_v2.ckpt" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    +trainer.limit_val_batches=5 \
    ++logger.name="smoke_v2_phase2" \
    ++callbacks.model_checkpoint.dirpath="checkpoints/smoke_v2" \
    ++callbacks.model_checkpoint.filename="best_mv2_ssmmem_phase2_davis"
EOF
)
echo "   -> Phase 2 Job ID: $JOB2"

echo "3) Submitting Phase 3 (Joint) depending on $JOB2..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_v2_p3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/smoke_v2/phase3_%j.out

source venv/bin/activate
export PYTHONPATH=.
python train.py \
    datamodule=ytv_dav_joint \
    +experiment=mv2_ssm_mem_phase3_ytbdav \
    +checkpoint="checkpoints/smoke_v2/best_mv2_ssmmem_phase2_davis.ckpt" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    +trainer.limit_val_batches=5 \
    ++logger.name="smoke_v2_phase3" \
    ++callbacks.model_checkpoint.dirpath="checkpoints/smoke_v2" \
    ++callbacks.model_checkpoint.filename="best_mv2_ssmmem_phase3_ytbdav"
EOF
)
echo "   -> Phase 3 Job ID: $JOB3"

echo "=========================================="
echo "V2 Smoke jobs submitted. Check progress with 'squeue --me'."
