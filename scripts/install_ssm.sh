#!/bin/bash
#SBATCH --job-name=install_ssm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/install_ssm_%j.out
#SBATCH --error=logs/slurm/install_ssm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Installing SSM Kernels (Mamba)"
echo "  Node: $(hostname)"
echo "=========================================="

# Load GPU module (based on JacKAN setup)
module load GPU
module load CUDA/12.6.0

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found."
    exit 1
fi

echo "Python: $(which python)"
echo "NVCC: $(which nvcc || echo 'Not Found')"

# Install causal-conv1d and mamba-ssm
# We use --no-cache-dir to ensure we build from source if needed
echo "Installing causal-conv1d..."
pip install --no-cache-dir causal-conv1d>=1.4.0

echo "Installing mamba-ssm..."
pip install --no-cache-dir mamba-ssm

echo "=========================================="
echo "  Verification"
echo "=========================================="
python -c "import causal_conv1d; import mamba_ssm; print('Success: SSM Kernels installed and importable!')"

echo "Installation complete at $(date)."
