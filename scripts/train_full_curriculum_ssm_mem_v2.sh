#!/bin/bash

# Configuration
EXP_NAME="ssm_mem_v2_aot"
LOG_DIR="logs/slurm/curriculum_v2"
mkdir -p $LOG_DIR

# 1. Phase 1: COCO Pre-training
JOB_P1=$(sbatch <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=v2p1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=${LOG_DIR}/phase1_%j.out

source venv/bin/activate
export PYTHONPATH=.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

python train.py \
    datamodule=coco_pretrain \
    +experiment=mv2_ssm_mem_phase1_coco \
    trainer.max_epochs=20
EOT
)

echo "Submitted Phase 1 (COCO): $JOB_P1"

# 2. Phase 2: DAVIS Fine-tuning (depends on Phase 1)
JOB_P2=$(sbatch --dependency=afterok:$JOB_P1 <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=v2p2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${LOG_DIR}/phase2_%j.out

source venv/bin/activate
export PYTHONPATH=.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

python train.py \
    datamodule=davis \
    +experiment=mv2_ssm_mem_phase2_davis \
    trainer.max_epochs=100
EOT
)

echo "Submitted Phase 2 (DAVIS) with dependency on $JOB_P1: $JOB_P2"

# 3. Phase 3: Joint Fine-tuning (depends on Phase 2)
JOB_P3=$(sbatch --dependency=afterok:$JOB_P2 <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=v2p3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=${LOG_DIR}/phase3_%j.out

source venv/bin/activate
export PYTHONPATH=.
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

python train.py \
    datamodule=ytv_dav_joint \
    +experiment=mv2_ssm_mem_phase3_ytbdav \
    trainer.max_epochs=50
EOT
)

echo "Submitted Phase 3 (Joint) with dependency on $JOB_P2: $JOB_P3"

echo "V2 Curriculum Launch Complete."
