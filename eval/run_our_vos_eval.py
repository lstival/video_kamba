"""Evaluate VideoMambaSystem (SSM-KAN VOS) on DAVIS 2017 validation set.

Adapts the aot-benchmark evaluation protocol (yoxu515/aot-benchmark) to run
our SSM-KAN model and compare results against the AOTT baseline.

Evaluation Protocol (mirrors sota/run_aott_davis_eval.py):
  1. For each sequence in DAVIS 2017 val (30 sequences, 480p):
     a. Frame 0: extract DINO features, initialise MemoryBank with GT mask.
     b. Frames 1..T: run PropagationAttention + KAN-SSM → predict mask.
     c. Save predictions as palette-indexed PNG files.
  2. Compute J (Jaccard IoU) and F (boundary F-measure) per sequence.
  3. Print comparison table vs AOTT baseline (83.3% J&F).

Requirements:
  - models/video_mamba.py on fix/claude_decoder branch (or compatible).
  - data/DAVIS/DAVIS directory (Annotations + JPEGImages).
  - Optional: a trained checkpoint via --ckpt.

Usage:
  # From project root — with random weights (architecture sanity check)
  python eval/run_our_vos_eval.py

  # With a trained checkpoint
  python eval/run_our_vos_eval.py --ckpt path/to/model.ckpt

  # Limit to N sequences for a quick smoke test
  python eval/run_our_vos_eval.py --max_seqs 3

  # Use a different input resolution (higher = better boundary accuracy)
  python eval/run_our_vos_eval.py --img_size 448
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

# ── Project paths ─────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
# Insert project root FIRST so our local 'utils' package is found before
# sota/aot-benchmark/utils (which has __init__.py and would shadow ours).
sys.path.insert(0, str(WORKSPACE_ROOT))
# Append aot-benchmark after project root so its 'utils' doesn't shadow ours.
_AOT_PATH = str(WORKSPACE_ROOT / "sota" / "aot-benchmark")
if _AOT_PATH not in sys.path:
    sys.path.append(_AOT_PATH)

DAVIS_ROOT   = WORKSPACE_ROOT / "data" / "DAVIS" / "DAVIS"
RESULT_ROOT  = WORKSPACE_ROOT / "eval_results" / "our_vos_davis2017_val"
METRICS_FILE = RESULT_ROOT / "metrics.json"

# ── AOTT baseline for comparison (measured in previous session) ───────────────
AOTT_RESULTS = {
    "bike-packing":       {"J": 0.823, "F": 0.804},
    "blackswan":          {"J": 0.959, "F": 0.986},
    "bmx-trees":          {"J": 0.588, "F": 0.833},
    "breakdance":         {"J": 0.883, "F": 0.907},
    "camel":              {"J": 0.961, "F": 0.983},
    "car-roundabout":     {"J": 0.973, "F": 0.964},
    "car-shadow":         {"J": 0.956, "F": 0.991},
    "cows":               {"J": 0.956, "F": 0.965},
    "dance-twirl":        {"J": 0.882, "F": 0.879},
    "dog":                {"J": 0.958, "F": 0.974},
    "dogs-jump":          {"J": 0.917, "F": 0.966},
    "drift-chicane":      {"J": 0.735, "F": 0.799},
    "drift-straight":     {"J": 0.935, "F": 0.924},
    "goat":               {"J": 0.901, "F": 0.896},
    "gold-fish":          {"J": 0.868, "F": 0.890},
    "horsejump-high":     {"J": 0.808, "F": 0.932},
    "india":              {"J": 0.768, "F": 0.593},
    "judo":               {"J": 0.867, "F": 0.878},
    "kite-surf":          {"J": 0.468, "F": 0.616},
    "lab-coat":           {"J": 0.553, "F": 0.519},
    "libby":              {"J": 0.882, "F": 0.969},
    "loading":            {"J": 0.854, "F": 0.881},
    "mbike-trick":        {"J": 0.726, "F": 0.733},
    "motocross-jump":     {"J": 0.693, "F": 0.693},
    "paragliding-launch": {"J": 0.498, "F": 0.587},
    "parkour":            {"J": 0.942, "F": 0.970},
    "pigs":               {"J": 0.838, "F": 0.856},
    "scooter-black":      {"J": 0.809, "F": 0.832},
    "shooting":           {"J": 0.773, "F": 0.810},
    "soapbox":            {"J": 0.769, "F": 0.794},
}

# ── DAVIS palette (same 256-entry palette used by all DAVIS tools) ─────────────
_PALETTE = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80\x40\x00\x00\xc0\x00\x00\x40\x80\x00\xc0\x80\x00\x40\x00\x80\xc0\x00\x80\x40\x80\x80\xc0\x80\x80\x00\x40\x00\x80\x40\x00\x00\xc0\x00\x80\xc0\x00\x00\x40\x80\x80\x40\x80\x00\xc0\x80\x80\xc0\x80\x40\x40\x00\xc0\x40\x00\x40\xc0\x00\xc0\xc0\x00\x40\x40\x80\xc0\x40\x80\x40\xc0\x80\xc0\xc0\x80\x00\x00\x40\x80\x00\x40\x00\x80\x40\x80\x80\x40\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0\x40\x00\x40\xc0\x00\x40\x40\x80\x40\xc0\x80\x40\x40\x00\xc0\xc0\x00\xc0\x40\x80\xc0\xc0\x80\xc0\x00\x40\x40\x80\x40\x40\x00\xc0\x40\x80\xc0\x40\x00\x40\xc0\x80\x40\xc0\x00\xc0\xc0\x80\xc0\xc0\x40\x40\x40\xc0\x40\x40\x40\xc0\x40\xc0\xc0\x40\x40\x40\xc0\xc0\x40\xc0\x40\xc0\xc0\xc0\xc0\xc0\x20\x00\x00\xa0\x00\x00\x20\x80\x00\xa0\x80\x00\x20\x00\x80\xa0\x00\x80\x20\x80\x80\xa0\x80\x80\x60\x00\x00\xe0\x00\x00\x60\x80\x00\xe0\x80\x00\x60\x00\x80\xe0\x00\x80\x60\x80\x80\xe0\x80\x80\x20\x40\x00\xa0\x40\x00\x20\xc0\x00\xa0\xc0\x00\x20\x40\x80\xa0\x40\x80\x20\xc0\x80\xa0\xc0\x80\x60\x40\x00\xe0\x40\x00\x60\xc0\x00\xe0\xc0\x00\x60\x40\x80\xe0\x40\x80\x60\xc0\x80\xe0\xc0\x80\x20\x00\x40\xa0\x00\x40\x20\x80\x40\xa0\x80\x40\x20\x00\xc0\xa0\x00\xc0\x20\x80\xc0\xa0\x80\xc0\x60\x00\x40\xe0\x00\x40\x60\x80\x40\xe0\x80\x40\x60\x00\xc0\xe0\x00\xc0\x60\x80\xc0\xe0\x80\xc0\x20\x40\x40\xa0\x40\x40\x20\xc0\x40\xa0\xc0\x40\x20\x40\xc0\xa0\x40\xc0\x20\xc0\xc0\xa0\xc0\xc0\x60\x40\x40\xe0\x40\x40\x60\xc0\x40\xe0\xc0\x40\x60\x40\xc0\xe0\x40\xc0\x60\xc0\xc0\xe0\xc0\xc0\x00\x20\x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80\x40 \x00\xc0 \x00\x40\xa0\x00\xc0\xa0\x00\x40 \x80\xc0 \x80\x40\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80\x40`\x00\xc0`\x00\x40\xe0\x00\xc0\xe0\x00\x40`\x80\xc0`\x80\x40\xe0\x80\xc0\xe0\x80\x00 \x40\x80 \x40\x00\xa0\x40\x80\xa0\x40\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0\x40 \x40\xc0 \x40\x40\xa0\x40\xc0\xa0\x40\x40 \xc0\xc0 \xc0\x40\xa0\xc0\xc0\xa0\xc0\x00`\x40\x80`\x40\x00\xe0\x40\x80\xe0\x40\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0\x40`\x40\xc0`\x40\x40\xe0\x40\xc0\xe0\x40\x40`\xc0\xc0`\xc0\x40\xe0\xc0\xc0\xe0\xc0\x20 \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00`\xa0\x00\xe0\xa0\x00 \xa0\x00\xa0\xa0\x00`\xa0\x00\xe0\xa0\x00 \xa0\x80\xa0\xa0\x80 \xa0\x00\x80\xa0\x00\x00\x00\x00'


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers (self-contained, matches sota/run_aott_davis_eval.py)
# ══════════════════════════════════════════════════════════════════════════════

def db_eval_iou(annotation: np.ndarray, segmentation: np.ndarray) -> float:
    """Jaccard similarity (J / IoU) for two binary masks."""
    a = annotation.astype(bool)
    s = segmentation.astype(bool)
    if a.sum() == 0 and s.sum() == 0:
        return 1.0
    if a.sum() == 0 or s.sum() == 0:
        return 0.0
    return float(np.logical_and(a, s).sum()) / float(np.logical_or(a, s).sum())


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(bool).copy()
    seg[[0, -1], :] = False
    seg[:, [0, -1]] = False
    e  = np.zeros_like(seg); e[:,  :-1] = seg[:, 1:]
    s  = np.zeros_like(seg); s[ :-1, :] = seg[1:, :]
    se = np.zeros_like(seg); se[:-1, :-1] = seg[1:, 1:]
    return (seg ^ e) | (seg ^ s) | (seg ^ se)


def db_eval_boundary(
    annotation: np.ndarray, segmentation: np.ndarray, bound_th: float = 0.008
) -> float:
    """Boundary F-measure between two binary masks."""
    pix = max(1, round(bound_th * np.linalg.norm(segmentation.shape)))
    fb  = _seg2bmap(segmentation.astype(bool))
    gb  = _seg2bmap(annotation.astype(bool))
    fd  = binary_dilation(fb, iterations=pix)
    gd  = binary_dilation(gb, iterations=pix)
    tp  = np.logical_and(fb, gd).sum()
    fp  = np.logical_and(fb, ~gd).sum()
    fn  = np.logical_and(gb, ~fd).sum()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def compute_jf_for_sequence(pred_dir: Path, gt_dir: Path) -> Dict[str, float]:
    """Compute mean J and F for a single multi-object sequence.

    Args:
        pred_dir:  directory containing predicted palette-indexed PNGs.
        gt_dir:    directory containing ground-truth palette-indexed PNGs.

    Returns:
        {"J": float, "F": float, "JF": float}
    """
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        return {"J": 0.0, "F": 0.0, "JF": 0.0}

    # Discover all object IDs from the first GT frame
    first_gt = sorted(gt_dir.glob("*.png"))[0]
    gt0 = np.array(Image.open(first_gt), dtype=np.uint8)
    obj_ids = sorted(set(np.unique(gt0).tolist()) - {0, 255})
    if not obj_ids:
        return {"J": 0.0, "F": 0.0, "JF": 0.0}

    j_per_obj: Dict[int, List[float]] = {oid: [] for oid in obj_ids}
    f_per_obj: Dict[int, List[float]] = {oid: [] for oid in obj_ids}

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred = np.array(Image.open(pred_path), dtype=np.uint8)
        gt   = np.array(Image.open(gt_path),   dtype=np.uint8)

        for oid in obj_ids:
            pred_bin = (pred == oid)
            gt_bin   = (gt   == oid)
            if gt_bin.sum() == 0:
                continue       # object not visible in this frame; skip
            j_per_obj[oid].append(db_eval_iou(gt_bin, pred_bin))
            f_per_obj[oid].append(db_eval_boundary(gt_bin, pred_bin))

    mean_j = float(np.mean([
        np.mean(scores) for scores in j_per_obj.values() if scores
    ]))
    mean_f = float(np.mean([
        np.mean(scores) for scores in f_per_obj.values() if scores
    ]))
    return {"J": mean_j, "F": mean_f, "JF": (mean_j + mean_f) / 2.0}


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(ckpt_path: Optional[str], device: torch.device) -> torch.nn.Module:
    """Load VideoMambaSystem (with VOS capability) onto ``device``.

    When a checkpoint is provided, uses ``load_from_checkpoint`` so that ALL
    saved hyper-parameters are restored automatically (avoids mismatches
    with post-training additions such as ``prop_use_dual_scale``).

    Falls back to random initialisation when no checkpoint is given — useful
    for sanity-checking that the inference pipeline runs end-to-end.
    """
    try:
        from models.video_mamba import VideoMambaSystem
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import VideoMambaSystem: {exc}\n"
            "Make sure you are running from the project root and the "
            "fix/claude_decoder branch is checked out (or merged)."
        )

    # Inspect the signature to handle both the simple model (main branch) and
    # the full VOS model (fix/claude_decoder branch).
    import inspect
    sig = inspect.signature(VideoMambaSystem.__init__)
    full_vos = "prop_d_key" in sig.parameters

    if ckpt_path is not None:
        print(f"[build_model] Loading checkpoint: {ckpt_path}")
        if full_vos:
            # prop_use_dual_scale was added AFTER the checkpoint was trained.
            # Forcing it to False matches the exact training configuration stored
            # in the checkpoint (epoch=49, step=11050).
            model = VideoMambaSystem.load_from_checkpoint(
                ckpt_path,
                strict=False,
                map_location=device,
                prop_use_dual_scale=False,
            )
            print("[build_model] Full VOS model loaded (prop_use_dual_scale=False).")
        else:
            model = VideoMambaSystem.load_from_checkpoint(
                ckpt_path, strict=False, map_location=device
            )
            print("[build_model] Simple model loaded from checkpoint.")
    else:
        print("[build_model] No checkpoint — using random initialisation.")
        if full_vos:
            model = VideoMambaSystem(
                num_seg_classes     = 11,
                prop_d_key          = 256,
                prop_d_value        = 256,
                prop_n_heads        = 8,
                max_mem_frames      = 5,
                memory_update_freq  = 1,
                fusion_mode         = "kan_spatial",
                modulator_type      = "kan",
                prop_use_dual_scale = False,
            )
        else:
            model = VideoMambaSystem()

    model.eval()
    model.to(device)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame inference helper
# ══════════════════════════════════════════════════════════════════════════════

# DINOv2 normalisation constants
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _preprocess_frame(
    img: Image.Image,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize and normalise a PIL image → [1, 3, size, size] float tensor."""
    img = img.resize((size, size), Image.BILINEAR)
    t   = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    t   = t.permute(2, 0, 1).unsqueeze(0)           # [1, 3, H, W]
    t   = (t - _DINO_MEAN.to(device)) / _DINO_STD.to(device)
    return t


def _save_pred_mask(pred_mask: np.ndarray, out_path: Path) -> None:
    """Save a uint8 mask as a palette-indexed PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(pred_mask.astype(np.uint8), mode="P")
    img.putpalette(_PALETTE)
    img.save(str(out_path))


# ══════════════════════════════════════════════════════════════════════════════
# Sequence-level inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sequence_inference(
    model: torch.nn.Module,
    seq_name: str,
    img_dir: Path,
    ann_dir: Path,
    pred_dir: Path,
    img_size: int,
    device: torch.device,
    full_vos: bool,
    chunk_size: int = 40,
) -> None:
    """Run VOS inference on a single DAVIS sequence and save predictions.

    Frame 0: initialise memory with GT mask.
    Frames 1..T: propagate memory → predict → save.

    Implementation note — memory bank reset behaviour:
        ``model.memory_bank.encode_reference()`` (called whenever ``ref_frame``
        is passed to ``forward()``) first calls ``reset()`` which discards ALL
        accumulated frames.  Therefore we must process all query frames in ONE
        forward call (or in contiguous chunks where the reference encoding is
        suppressed after the first chunk).

    Args:
        model:      VideoMambaSystem.
        seq_name:   Sequence name (e.g. ``"dog"``).
        img_dir:    Path to DAVIS JPEGImages/480p/<seq_name>/.
        ann_dir:    Path to DAVIS Annotations/480p/<seq_name>/.
        pred_dir:   Directory where predicted PNGs are written.
        img_size:   Input spatial resolution for the model.
        device:     ``torch.device`` to run on.
        full_vos:   Whether the model has the full VOS API.
        chunk_size: Max query frames per forward call.  Larger = more accurate
                    SSM continuity; smaller = lower peak memory.  Default 40.
    """
    frame_names = sorted(f.name for f in img_dir.glob("*.jpg"))
    if not frame_names:
        print(f"  [warn] No frames found in {img_dir}")
        return

    # ── Frame 0: ground-truth mask as reference ────────────────────────────
    frame0_img  = Image.open(img_dir  / frame_names[0]).convert("RGB")
    frame0_ann  = ann_dir / frame_names[0].replace(".jpg", ".png")
    frame0_mask = np.array(Image.open(frame0_ann), dtype=np.uint8)

    orig_h, orig_w = frame0_mask.shape[:2]

    # Save the reference frame prediction (copy GT mask as prediction)
    _save_pred_mask(frame0_mask, pred_dir / frame_names[0].replace(".jpg", ".png"))

    query_names = frame_names[1:]   # frames to predict
    if not query_names:
        return

    if not full_vos or not hasattr(model, "memory_bank"):
        # Fallback: simple model without VOS — run forward on each frame
        # independently for architecture validation only.
        print(f"  [info] Simple model: running frame-by-frame without propagation")
        for fname in query_names:
            img_t = _preprocess_frame(
                Image.open(img_dir / fname).convert("RGB"), img_size, device
            )
            dummy_ref  = img_t
            dummy_mask = torch.zeros(1, img_size, img_size, dtype=torch.long, device=device)
            out = model(img_t.unsqueeze(1), ref_frame=dummy_ref, ref_mask=dummy_mask)
            if isinstance(out, (list, tuple)) and len(out) == 4:
                logits_seg = out[3]                          # [1,1,C,H,W]
                pred_ids   = logits_seg[0, 0].argmax(0).cpu().numpy().astype(np.uint8)
            else:
                pred_ids = np.zeros((img_size, img_size), dtype=np.uint8)
            pred_pil = Image.fromarray(pred_ids).resize((orig_w, orig_h), Image.NEAREST)
            _save_pred_mask(np.array(pred_pil), pred_dir / fname.replace(".jpg", ".png"))
        return

    # ── Full VOS model ─────────────────────────────────────────────────────
    ref_img_t  = _preprocess_frame(frame0_img, img_size, device)      # [1,3,H,W]
    ref_mask_t = torch.from_numpy(frame0_mask.astype(np.int64)).unsqueeze(0)  # [1,H,W]
    ref_mask_ss = F.interpolate(
        ref_mask_t.float().unsqueeze(1), size=(img_size, img_size), mode="nearest"
    ).long().squeeze(1)  # [1, img_size, img_size]

    # Split query frames into chunks to bound peak memory.
    # CRITICAL: Only the FIRST chunk passes ref_frame (initialises the memory
    # bank).  Subsequent chunks temporarily replace encode_reference with a
    # no-op so the memory bank is NOT reset between chunks — preserving all
    # frames added by previous chunks.
    chunks = [query_names[i : i + chunk_size] for i in range(0, len(query_names), chunk_size)]

    for chunk_idx, chunk in enumerate(chunks):
        # Preload all frames in this chunk
        imgs = [
            _preprocess_frame(Image.open(img_dir / fn).convert("RGB"), img_size, device)
            for fn in chunk
        ]
        query_tensor = torch.stack(imgs, dim=1)  # [1, T_chunk, 3, H, W]

        if chunk_idx == 0:
            # Provide ref_frame so encode_reference initialises the bank.
            _, _, _, logits_seg = model(
                query_tensor,
                ref_frame   = ref_img_t,
                ref_mask    = ref_mask_ss,
                query_masks = None,
            )  # [1, T_chunk, C, H, W]
        else:
            # Suppress the reset inside encode_reference by temporarily
            # replacing it with a lightweight no-op that just re-encodes the
            # reference entry in-place (keeps bank state intact).
            orig_encode = model.memory_bank.encode_reference

            def _reinsert_ref(feats, mask, feats_fine=None):
                """Re-encode ref entry without calling reset()."""
                K, V = model.memory_bank._encode_frame(feats, mask, feats_fine)
                model.memory_bank._keys[0]         = K
                model.memory_bank._values[0]       = V
                model.memory_bank._is_reference[0] = True

            model.memory_bank.encode_reference = _reinsert_ref
            try:
                _, _, _, logits_seg = model(
                    query_tensor,
                    ref_frame   = ref_img_t,
                    ref_mask    = ref_mask_ss,
                    query_masks = None,
                )  # [1, T_chunk, C, H, W]
            finally:
                model.memory_bank.encode_reference = orig_encode

        # Save predictions for this chunk
        for i, fname in enumerate(chunk):
            pred_ids_ss = logits_seg[0, i].argmax(0).cpu().numpy().astype(np.uint8)
            pred_full   = np.array(
                Image.fromarray(pred_ids_ss).resize((orig_w, orig_h), Image.NEAREST)
            )
            _save_pred_mask(pred_full, pred_dir / fname.replace(".jpg", ".png"))


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSM-KAN VOS on DAVIS 2017 val")
    parser.add_argument("--ckpt",     type=str, default=None,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Model input resolution (use 448 for higher accuracy)")
    parser.add_argument("--max_seqs", type=int, default=None,
                        help="Limit to first N sequences (for smoke tests)")
    parser.add_argument("--resolution", type=str, default="480p",
                        help="DAVIS resolution tier: 480p or Full-Resolution")
    parser.add_argument("--davis_root", type=str, default=str(DAVIS_ROOT),
                        help="Override DAVIS root path")
    parser.add_argument("--out_dir", type=str, default=str(RESULT_ROOT),
                        help="Directory to write predictions and metrics")
    parser.add_argument("--chunk_size", type=int, default=40,
                        help="Max query frames per forward call (lower = less RAM)")
    args = parser.parse_args()

    davis_root = Path(args.davis_root)
    out_dir    = Path(args.out_dir)
    pred_root  = out_dir / "preds" / args.resolution

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  img_size: {args.img_size}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = _build_model(args.ckpt, device)

    import inspect
    full_vos = "prop_d_key" in inspect.signature(
        model.__class__.__init__
    ).parameters

    # ── Load val sequences ────────────────────────────────────────────────────
    split_file = davis_root / "ImageSets" / "2017" / "val.txt"
    with open(split_file) as f:
        sequences = [l.strip() for l in f if l.strip()]
    if args.max_seqs is not None:
        sequences = sequences[: args.max_seqs]

    img_base = davis_root / "JPEGImages"  / args.resolution
    ann_base = davis_root / "Annotations" / args.resolution

    # ── Per-sequence inference + metric computation ───────────────────────────
    all_results: Dict[str, Dict] = {}
    total_time = 0.0

    for si, seq in enumerate(sequences, start=1):
        seq_img_dir  = img_base / seq
        seq_ann_dir  = ann_base / seq
        seq_pred_dir = pred_root / seq
        seq_pred_dir.mkdir(parents=True, exist_ok=True)

        n_frames = len(list(seq_img_dir.glob("*.jpg")))
        t0 = time.time()

        run_sequence_inference(
            model      = model,
            seq_name   = seq,
            img_dir    = seq_img_dir,
            ann_dir    = seq_ann_dir,
            pred_dir   = seq_pred_dir,
            img_size   = args.img_size,
            device     = device,
            full_vos   = full_vos,
            chunk_size = args.chunk_size,
        )

        elapsed  = time.time() - t0
        total_time += elapsed
        fps = n_frames / elapsed

        jf = compute_jf_for_sequence(seq_pred_dir, seq_ann_dir)
        all_results[seq] = {**jf, "frames": n_frames, "time_s": elapsed}

        # Reference AOTT numbers for this sequence
        aott = AOTT_RESULTS.get(seq, {"J": 0.0, "F": 0.0})
        aott_jf = (aott["J"] + aott["F"]) / 2.0

        print(
            f"[{si:2d}/{len(sequences)}]  {seq:<20s}  "
            f"J={jf['J']:.3f}  F={jf['F']:.3f}  J&F={jf['JF']:.3f}  "
            f"| AOTT J&F={aott_jf:.3f}  "
            f"({fps:.1f} fps)"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mean_j  = float(np.mean([v["J"]  for v in all_results.values()]))
    mean_f  = float(np.mean([v["F"]  for v in all_results.values()]))
    mean_jf = float(np.mean([v["JF"] for v in all_results.values()]))

    aott_mean_j  = float(np.mean([AOTT_RESULTS[s]["J"] for s in sequences if s in AOTT_RESULTS]))
    aott_mean_f  = float(np.mean([AOTT_RESULTS[s]["F"] for s in sequences if s in AOTT_RESULTS]))
    aott_mean_jf = (aott_mean_j + aott_mean_f) / 2.0

    # ── Save metrics ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = out_dir / "metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({
            "model":        "VideoMambaSystem (SSM-KAN VOS)",
            "ckpt":         args.ckpt or "random_init",
            "img_size":     args.img_size,
            "resolution":   args.resolution,
            "num_seqs":     len(sequences),
            "mean_J":       mean_j,
            "mean_F":       mean_f,
            "mean_JF":      mean_jf,
            "total_time_s": total_time,
            "per_sequence": all_results,
        }, f, indent=2)

    # ── Comparison table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DAVIS 2017 Val — SSM-KAN vs AOTT Comparison")
    print("=" * 60)
    print(f"  {'Model':<24s}  {'J&F':>6s}  {'J':>6s}  {'F':>6s}")
    print(f"  {'-'*24}  {'------':>6s}  {'------':>6s}  {'------':>6s}")
    print(f"  {'SSM-KAN (ours)':<24s}  {mean_jf*100:>6.1f}  {mean_j*100:>6.1f}  {mean_f*100:>6.1f}")
    print(f"  {'AOTT (paper)':<24s}  {79.2:>6.1f}  {76.5:>6.1f}  {81.9:>6.1f}")
    print(f"  {'AOTT (our run)':<24s}  {aott_mean_jf*100:>6.1f}  {aott_mean_j*100:>6.1f}  {aott_mean_f*100:>6.1f}")
    print(f"  {'AOTS (paper)':<24s}  {82.1:>6.1f}  {79.3:>6.1f}  {84.8:>6.1f}")
    print(f"  {'AOTB (paper)':<24s}  {83.3:>6.1f}  {80.6:>6.1f}  {85.9:>6.1f}")
    print("=" * 60)
    delta_jf = mean_jf * 100 - aott_mean_jf * 100
    print(f"  Gap vs AOTT (our run): {delta_jf:+.1f}%  J&F")
    print(f"  Metrics saved to: {metrics_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
