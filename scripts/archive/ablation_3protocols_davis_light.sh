#!/bin/bash
#SBATCH --job-name=abl_davis_3prot_lw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/abl_davis_3prot_lw_%j.out
#SBATCH --error=logs/slurm/abl_davis_3prot_lw_%j.err

# 3-protocol lightweight ablation on DAVIS only (sequential runs).
# Purpose: fast ranking of protocol variants before expensive YouTube-VOS runs.

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found." && exit 1
fi

export PYTHONPATH=.
# Job-specific TMPDIR helps avoid NFS temp-file contention.
export TMPDIR="${PROJECT_ROOT}/tmp/abl_davis_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" logs/slurm

CKPT="/home/WUR/stiva001/WUR/video_kamba/checkpoints/ade20k_ft_base/ade20k_epoch27_step140000.ckpt"
EPOCHS="${EPOCHS:-20}"

echo "=========================================="
echo "  DAVIS 3-Protocol Ablation (Lightweight)"
echo "  Checkpoint: $CKPT"
echo "  Epochs/protocol: $EPOCHS"
echo "  Job ID: ${SLURM_JOB_ID:-manual}"
echo "=========================================="

run_protocol() {
    local NAME="$1"
    shift

    echo ""
    echo "------------------------------------------"
    echo "Running protocol: $NAME"
    echo "------------------------------------------"

    python train.py \
        model=vision_mamba_tiny_sota_light \
        datamodule=davis \
        +checkpoint="$CKPT" \
        ++trainer.max_epochs="$EPOCHS" \
        ++trainer.accumulate_grad_batches=2 \
        ++datamodule.img_size=448 \
        ++datamodule.augment_train=True \
        ++datamodule.num_workers=8 \
        ++model.max_epochs="$EPOCHS" \
        ++callbacks.monitor=val_J_and_F \
        ++callbacks.mode=max \
        ++logger.name="$NAME" \
        "$@"
}

# Protocol 1: Baseline (shallower temporal stack)
run_protocol \
    "abl_davis_p1_l1_mem5_fixedss" \
    ++model.ssm_layers=1 \
    ++model.max_mem_frames=5 \
    ++model.scheduled_sampling_rate=0.10 \
    ++model.scheduled_sampling_start=-1 \
    ++model.scheduled_sampling_end=-1

# Protocol 2: Deeper temporal stack
run_protocol \
    "abl_davis_p2_l2_mem5_fixedss" \
    ++model.ssm_layers=2 \
    ++model.max_mem_frames=5 \
    ++model.scheduled_sampling_rate=0.10 \
    ++model.scheduled_sampling_start=-1 \
    ++model.scheduled_sampling_end=-1

# Protocol 3: Deeper temporal + larger memory + SS curriculum
run_protocol \
    "abl_davis_p3_l2_mem7_currss" \
    ++model.ssm_layers=2 \
    ++model.max_mem_frames=7 \
    ++model.scheduled_sampling_rate=0.25 \
    ++model.scheduled_sampling_start=0.0 \
    ++model.scheduled_sampling_end=0.25 \
    ++model.scheduled_sampling_warmup_epochs=5

echo ""
echo "Finished DAVIS 3-protocol ablation at: $(date)"
