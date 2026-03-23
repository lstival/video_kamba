#!/usr/bin/env bash
# ============================================================
# train_kan_decoder_local.sh
#
# Local smoke-train: KAN-Modulated Spatial Alignment Decoder
# runs 5 epochs on DAVIS using the conda environment KAR.
#
# Usage:
#   bash scripts/train_kan_decoder_local.sh
# ============================================================
set -euo pipefail

# ---------- resolve repo root (works whether called from any cwd) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ---------- sanity-check: must be on the feature branch ----------
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "kan-spatial-decoder" ]]; then
    echo "[WARN] Not on branch 'kan-spatial-decoder' (currently: $CURRENT_BRANCH)."
    echo "       Run:  git checkout kan-spatial-decoder"
fi

# ---------- activate conda environment ----------
# conda activate does not work inside non-interactive shells;
# we source conda.sh to make it available.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate KAR

# Resolve Python: KAR packages may live in site-packages without their own
# interpreter.  Fall back to conda-base Python, then system python3/python.
PYTHON=$(command -v python 2>/dev/null \
    || command -v python3 2>/dev/null \
    || echo "$CONDA_BASE/python.exe")

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] No Python interpreter found."
    echo "        Install Python into the KAR env first:"
    echo "          conda install -n KAR python"
    exit 1
fi

# Ensure KAR site-packages are visible when running the base interpreter
KAR_SITE="$CONDA_BASE/envs/KAR/Lib/site-packages"
if [[ -d "$KAR_SITE" ]]; then
    export PYTHONPATH="$KAR_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "========================================"
echo " KAN Spatial Decoder — DAVIS 5-epoch run"
echo " Branch  : $CURRENT_BRANCH"
echo " Env     : KAR  (site-packages: $KAR_SITE)"
echo " Root    : $REPO_ROOT"
echo " Python  : $($PYTHON --version)"
echo "========================================"

# ---------- quick decoder import check ----------
$PYTHON - <<'EOF'
import torch
from models.components.segmentation_decoder import KANSpatialGatingUpBlock, SegmentationDecoder

# Shape test: mimics up1 inside SegmentationDecoder (dim_ssm=768 → 256)
block = KANSpatialGatingUpBlock(in_channels=768, skip_channels=768, out_channels=256)
x    = torch.randn(2, 768, 16, 16)   # [BT, D_ssm, h, w]
skip = torch.randn(2, 768, 32, 32)   # [BT, D_dino, 2h, 2w]
out  = block(x, skip)
assert out.shape == (2, 256, 32, 32), f"Unexpected shape: {out.shape}"
print(f"[OK] KANSpatialGatingUpBlock output: {out.shape}")

# Full decoder test (fusion_mode must default to kan_spatial)
decoder = SegmentationDecoder(
    dim_ssm=768, dim_dinov2=768,
    num_classes=11, target_size=224,
    fusion_mode="kan_spatial",
)
ssm_out  = torch.randn(1, 2, 768, 256)   # [B, T, D, P]  P=16x16
dino_feats = {k: torch.randn(1, 2, 768, 256) for k in ["layer_3","layer_6","layer_9","layer_11"]}
logits = decoder(ssm_out, dino_feats)
print(f"[OK] SegmentationDecoder output   : {logits.shape}")   # [1, 2, 11, 224, 224]
EOF

echo ""
echo "[INFO] Import check passed. Starting training..."
echo ""

# ---------- train 5 epochs on DAVIS ----------
$PYTHON train.py \
    datamodule=davis \
    model=default \
    trainer=default \
    logger=none \
    trainer.max_epochs=5 \
    trainer.min_epochs=1 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.precision="16-mixed" \
    trainer.fast_dev_run=false \
    datamodule.batch_size=2 \
    datamodule.num_workers=2 \
    model.fusion_mode=kan_spatial \
    model.ssm_layers=1 \
    model.learning_rate=1e-4 \
    seed=42

echo ""
echo "[DONE] 5-epoch DAVIS run complete."
