#!/bin/bash
# scripts/train_full_curriculum_ssm_mem.sh

set -euo pipefail

mkdir -p logs/slurm/curriculum
export PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/ssm_mem_v1"
mkdir -p "$CHECKPOINT_DIR"

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv not found." && exit 1
fi

echo "=========================================="
echo "  SSM-Memory Full Curriculum Submission"
echo "  Target: $CHECKPOINT_DIR"
echo "=========================================="

echo "1) Submitting Phase 1 (COCO)..."
JOB1=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=ssmm_p1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/curriculum/phase1_%j.out

source venv/bin/activate
export PYTHONPATH=.
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi
python train.py \
    datamodule=coco_pretrain \
    +experiment=mv2_ssm_mem_phase1_coco \
    ++callbacks.model_checkpoint.dirpath="$CHECKPOINT_DIR" \
    ++logger.name="ssm_mem_p1_coco"
EOF
)
echo "   -> Phase 1 Job ID: $JOB1"

echo "2) Submitting Phase 2 (DAVIS) depending on $JOB1..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 << EOF
#!/bin/bash
#SBATCH --job-name=ssmm_p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/curriculum/phase2_%j.out

source venv/bin/activate
export PYTHONPATH=.
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi
python train.py \
    datamodule=davis_vos \
    +experiment=mv2_ssm_mem_phase2_davis \
    +pretrained_weights="$CHECKPOINT_DIR/best_mv2_ssmmem_phase1_coco.ckpt" \
    ++callbacks.model_checkpoint.dirpath="$CHECKPOINT_DIR" \
    ++logger.name="ssm_mem_p2_davis"
EOF
)
echo "   -> Phase 2 Job ID: $JOB2"

echo "3) Submitting Phase 3 (YouTube-VOS + DAVIS) depending on $JOB2..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 << EOF
#!/bin/bash
#SBATCH --job-name=ssmm_p3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=30:00:00
#SBATCH --output=logs/slurm/curriculum/phase3_%j.out

source venv/bin/activate
export PYTHONPATH=.
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi
python train.py \
    datamodule=ytv_dav_joint \
    +experiment=mv2_ssm_mem_phase3_ytbdav \
    +pretrained_weights="$CHECKPOINT_DIR/best_mv2_ssmmem_phase2_davis.ckpt" \
    ++callbacks.model_checkpoint.dirpath="$CHECKPOINT_DIR" \
    ++logger.name="ssm_mem_p3_ytbdav"
EOF
)
echo "   -> Phase 3 Job ID: $JOB3"

echo "=========================================="
echo "All curriculum jobs submitted successfully!"
echo "Sequence: $JOB1 -> $JOB2 -> $JOB3"
echo "Check progress with 'squeue --me'."
