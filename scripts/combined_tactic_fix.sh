#!/bin/bash
#SBATCH --job-name=vos_combined_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/combined_fix_%j.out
#SBATCH --error=logs/slurm/combined_fix_%j.err

echo "=========================================="
echo "  VOS Combined Tactic Fix Started"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

python train.py \
    datamodule=davis \
    ++trainer.max_epochs=5 \
    ++model.ssm_d_state=64 \
    ++model.ssm_layers=2 \
    ++model.identity_mode="concat" \
    ++model.vos_loss_beta=0.2 \
    ++model.consistency_weight=0.1 \
    ++logger.name="tactic_combined_fixed"

echo -e "\n=========================================="
echo "Job Finished at $(date)"
echo "=========================================="
