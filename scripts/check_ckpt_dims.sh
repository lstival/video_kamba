#!/bin/bash
#SBATCH --job-name=check_ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm/check_ckpt_%j.out
#SBATCH --error=logs/slurm/check_ckpt_%j.err

set -euo pipefail
PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"
source venv/bin/activate
export PYTHONPATH=.

python3 - <<'EOF'
import torch, os, glob

ckpts = {
    "best_coco":        "checkpoints/best_coco.ckpt",
    "best_coco_vimtiny":"checkpoints/best_coco_vimtiny.ckpt",
}

# Also find the most recent checkpoint in logs/
recent = sorted(glob.glob("logs/**/*.ckpt", recursive=True), key=os.path.getmtime)
for p in recent[-3:]:
    ckpts[os.path.basename(p)] = p

for name, path in ckpts.items():
    if not os.path.exists(path):
        print(f"[SKIP] {name}: not found")
        continue
    size_mb = os.path.getsize(path) / 1e6
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", {})

    # Check temporal model hidden dim
    b_key = next((k for k in sd if "temporal_model.layers.0.B" in k and "modulator" not in k), None)
    enc_key = next((k for k in sd if "encoder" in k and "weight" in k), None)

    b_shape = tuple(sd[b_key].shape) if b_key else "N/A"
    enc_shape = tuple(sd[enc_key].shape) if enc_key else "N/A"

    hyper = ckpt.get("hyper_parameters", {})
    print(f"\n{'='*60}")
    print(f"  {name}  ({size_mb:.1f} MB)")
    print(f"  temporal B shape : {b_shape}  (dim = {b_shape[1] if isinstance(b_shape,tuple) else '?'})")
    print(f"  first encoder key: {enc_key}")
    print(f"  first enc shape  : {enc_shape}")
    print(f"  hyper_parameters : {list(hyper.keys())[:8]}")
EOF
