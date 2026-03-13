#!/bin/bash
#SBATCH --job-name=vos_improvement_suite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/improvement_%j.out
#SBATCH --error=logs/slurm/improvement_%j.err

# ==========================================
# Video Mamba VOS Improvement Suite
# Tests 5 tactics for 5 epochs each
# ==========================================

echo "=========================================="
echo "  VOS Improvement Suite Started"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Start time: $(date)"
echo "=========================================="

source venv/bin/activate
export PYTHONPATH=.

EPOCHS=5

# 1. Tactic B: Increased SSM Capacity (d_state=64, layers=2)
echo -e "\n[1/5] Running Tactic B: High Capacity SSM..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.ssm_d_state=64 \
    ++model.ssm_layers=2 \
    ++logger.name="tactic_b_capacity"

# ... (omitted similar updates for brevity, will apply carefully)

# 2. Tactic A: Identity Injection (Concat Mode)
echo -e "\n[2/5] Running Tactic A: Identity Concat..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.identity_mode="concat" \
    ++logger.name="tactic_a_concat"

# 3. Tactic A v2: Identity Modulation (Gated Mode)
echo -e "\n[3/5] Running Tactic A v2: Identity Modulation..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.identity_mode="modulate" \
    ++logger.name="tactic_a_modulate"

# 4. Tactic D & E: Optimized Loss (Dice focus + Consistency)
echo -e "\n[4/5] Running Tactic D+E: Loss Optimization..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.vos_loss_beta=0.2 \
    ++model.consistency_weight=0.1 \
    ++logger.name="tactic_d_e_loss"

# 5. Combined: The "Ultimate" VOS Configuration
echo -e "\n[5/5] Running All-In: Combined Tactics..."
python train.py \
    datamodule=davis \
    ++trainer.max_epochs=$EPOCHS \
    ++model.ssm_d_state=64 \
    ++model.ssm_layers=2 \
    ++model.identity_mode="concat" \
    ++model.vos_loss_beta=0.2 \
    ++model.consistency_weight=0.1 \
    ++logger.name="tactic_combined"

echo -e "\n=========================================="
echo "Improvement Suite Finished at $(date)"
echo "=========================================="
