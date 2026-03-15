#!/bin/bash
#SBATCH --job-name=smoke_diag_kan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/smoke_diag_kan_%j.out
#SBATCH --error=logs/slurm/smoke_diag_kan_%j.err

# ============================================================
# Smoke test — MobileNetV2 + DiagonalKANSSMCore
# Branch: feat/mobilenet-diagonal-kan-ssm
#
# Validates:
#   1. DiagonalKANSSMCore unit (shapes, gradients, T=1 + T>1)
#   2. KangaSSM diagonal backend (state carry-over)
#   3. Full VideoMambaSystem VOS forward pass (MV2 encoder)
#   4. Gradient flow through the complete model
#   5. All 10 ablation configs (A/B/C on|off + KAN vs MLP)
#   6. Parameter count check (must stay < 6M)
#   7. FPS benchmark at 480p (target: > 40 fps)
#
# Runtime: ~10 min on a single GPU
# ============================================================

set -euo pipefail

echo "=========================================="
echo " Smoke: MobileNetV2 + DiagonalKANSSM"
echo " Job ID : ${SLURM_JOB_ID:-local}"
echo " Node   : $(hostname)"
echo " Date   : $(date)"
echo "=========================================="

# ── Environment ────────────────────────────────────────────────
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

# ── Working directory ────────────────────────────────────────────
# SLURM copies the script to its spool dir, so BASH_SOURCE[0] is wrong.
# SLURM_SUBMIT_DIR is the directory where sbatch was called — use that.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

# Activate Python environment (prefer repo-local venv used for development).
if [ -f "$REPO_DIR/venv/bin/activate" ]; then
    source "$REPO_DIR/venv/bin/activate"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate video_kamba 2>/dev/null || conda activate base
elif [ -d "$HOME/.venv" ]; then
    source "$HOME/.venv/bin/activate"
fi

if ! python -c 'import torch' >/dev/null 2>&1; then
    echo "[ERROR] Python environment does not provide torch."
    echo "        Checked: $REPO_DIR/venv, conda(video_kamba/base), $HOME/.venv"
    exit 1
fi

echo "Repo    : $REPO_DIR"
echo "Python  : $(which python)"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

# ── Log directory ────────────────────────────────────────────────
mkdir -p logs/slurm

# ── Run smoke test ───────────────────────────────────────────────
echo "--- Starting smoke test ---"
python scripts/smoke_mobilenet_diagonal_kan.py \
    --device cuda \
    --fps-frames 40

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " RESULT: ALL TESTS PASSED ✓"
else
    echo " RESULT: SMOKE TEST FAILED (exit $EXIT_CODE)"
fi
echo " $(date)"
echo "=========================================="

exit $EXIT_CODE
