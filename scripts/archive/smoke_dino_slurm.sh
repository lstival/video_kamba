#!/bin/bash
#SBATCH --job-name=dino_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/dino_smoke_%j.out
#SBATCH --error=logs/slurm/dino_smoke_%j.err

source venv/bin/activate
export PYTHONPATH=.

python -c "
import torch
from models.video_mamba import VideoMambaSystem

print('Initializing VideoMambaSystem with DINO backbone...')
model = VideoMambaSystem(dim_in=768)
print('SUCCESS: Initialization complete.')

B, T, C, H, W = 1, 2, 3, 224, 224
x = torch.randn(B, T, C, H, W)
ref_frame = torch.randn(B, C, H, W)
ref_mask = torch.randint(0, 2, (B, H, W))

print(f'Running forward pass with input {x.shape}...')
with torch.no_grad():
    logits_clf, pred_boxes, pred_box_logits, logits_seg = model(x, ref_frame=ref_frame, ref_mask=ref_mask)

print('SUCCESS: Forward pass complete.')
print(f'Logits Seg Shape: {logits_seg.shape}')
assert logits_seg.shape == (B, T, 11, H, W)
"
