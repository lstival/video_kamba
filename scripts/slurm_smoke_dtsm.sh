#!/bin/bash
#SBATCH --job-name=smoke_dtsm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/smoke_dtsm_%j.out

# ============================================================
#  DTSM smoke test — verifies instantiation, param counts,
#  and a single forward pass with use_dual_timescale_memory=True.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

python - <<'EOF'
import math
import torch
from models.video_mamba import VideoMambaSystem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = VideoMambaSystem(
    encoder_type="mobilenetv2",
    dim_in=256, dim_in_fine=96, dim_in_s2=32, dim_in_s1=24,
    use_dual_timescale_memory=True,
    slow_ssm_d_state=64,
    slow_ssm_log_a_init=-3.0,
    num_seg_classes=11,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
slow_params  = sum(p.numel() for p in model.temporal_model_slow.parameters())
proj_params  = sum(p.numel() for p in model.slow_residual_proj.parameters())

print(f"Total params : {total_params / 1e6:.2f}M")
print(f"Slow SSM     : {slow_params  / 1e3:.1f}k")
print(f"Residual proj: {proj_params  / 1e3:.1f}k")
print(f"DTSM overhead: {(slow_params + proj_params) / 1e3:.1f}k")

# Verify log_A initialisation → A_bar should be close to 1 (slow decay)
for layer in model.temporal_model_slow.layers:
    if hasattr(layer, "log_A"):
        a_bar = torch.exp(-torch.exp(layer.log_A) * 0.1).mean().item()
        assert a_bar > 0.98, f"A_bar={a_bar:.4f} — slow decay not achieved"
        print(f"Slow A_bar (delta=0.1): {a_bar:.4f}  ✓  (target > 0.98)")

# Single forward pass with T=12 frames (new clip_len)
B, T, C, H, W = 1, 12, 3, 480, 480
ref_img   = torch.randn(B, C, H, W, device=device)
ref_mask  = torch.zeros(B, H, W, dtype=torch.long, device=device)
q_imgs    = torch.randn(B, T, C, H, W, device=device)
q_masks   = torch.zeros(B, T, H, W, dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    _, _, _, logits_seg = model(q_imgs, ref_frame=ref_img, ref_mask=ref_mask)

assert logits_seg.shape == (B, T, 11, H, W), \
    f"Unexpected logits shape: {logits_seg.shape}"
print(f"Forward pass output: {logits_seg.shape}  ✓")
print("DTSM smoke test passed.")
EOF
