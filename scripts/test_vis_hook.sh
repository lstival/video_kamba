#!/bin/bash
#SBATCH --job-name=test_vis_hook
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/test_vis_hook_%j.out
#SBATCH --error=logs/slurm/test_vis_hook_%j.err

set -euo pipefail
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"
source venv/bin/activate
export PYTHONPATH=.
mkdir -p logs/slurm

python - <<'EOF'
import torch, math
from omegaconf import DictConfig, ListConfig
torch.serialization.add_safe_globals([DictConfig, ListConfig])
from models.video_mamba import VideoMambaSystem

device = torch.device("cuda")
m = VideoMambaSystem.load_from_checkpoint(
    "checkpoints/best_mv2_phase1_coco.ckpt", map_location=device
)
m.eval().to(device)

print("Hook target:", m.propagation_attention.attn_drop)

x = torch.zeros(1, 1, 3, 480, 480, device=device)
with torch.no_grad():
    _, raw = m.feature_extractor(x)
ms = m._map_ms_features(raw)
p = ms["stage_3"].shape[-1]
hw = ms.get("stage_3_hw")
print(f"P={p}, stage_3_hw={hw}, isqrt(P)={math.isqrt(p)}")

# Full forward pass with hook
captures = []
def hook(mod, inp, out):
    captures.append(out.detach().cpu())
handle = m.propagation_attention.attn_drop.register_forward_hook(hook)

from scripts.vis_mca_memory import load_frame, load_annotation
from pathlib import Path
seq = "blackswan"
davis = Path("data/DAVIS/DAVIS")
frames = sorted((davis / "JPEGImages/480p" / seq).glob("*.jpg"))
anns   = sorted((davis / "Annotations/480p" / seq).glob("*.png"))

ref_f = load_frame(frames[0], 480).unsqueeze(0).to(device)
ref_m = load_annotation(anns[0], 480).unsqueeze(0).to(device)
qframes = torch.stack([load_frame(frames[i], 480) for i in range(1, 4)]).unsqueeze(0).to(device)

with torch.no_grad():
    m(qframes, ref_frame=ref_f, ref_mask=ref_m)
handle.remove()

print(f"Captured {len(captures)} attention maps")
print(f"Shape of attn[0]: {captures[0].shape}  (B, n_heads, P_q, P_mem)")
print("Sanity check PASSED")
EOF
