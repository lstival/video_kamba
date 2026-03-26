"""LVOS evaluation results visualizer.

Reads the JSON produced by ``eval_lvos_train.py``, selects the best-N,
worst-N and middle-N sequences by J&F score, generates animated GIFs with
predicted mask overlays, and prints a SOTA comparison table.

Usage
-----
python scripts/visualize_lvos_results.py \\
    --results  artifacts/lvos_eval/results_YYYYMMDD_HHMMSS.json \\
    --lvos-root data/LVOS \\
    --output-dir artifacts/lvos_gifs \\
    [--top-n 5] [--fps 6] [--max-frames 120] [--alpha 0.55]

The predictions directory is inferred automatically as
``<results_parent>/Annotations/`` (where eval_lvos_train.py saves PNGs).
Pass ``--predictions <path>`` to override.

Output
------
artifacts/lvos_gifs/
    best/   best-N GIFs
    worst/  worst-N GIFs
    middle/ middle-N GIFs
    sota_comparison.txt

References
----------
LVOS V2 — Hong et al., T-PAMI 2024.
SOTA numbers from SOTA.md (VideoKamba project, 2026-03-25).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger(__name__)

# ── DAVIS / LVOS colour palette ───────────────────────────────────────────────
# Object IDs 0..10; background is (0,0,0).
_PALETTE_RGB: list[tuple[int, int, int]] = [
    (  0,   0,   0),   # 0 background
    (128,   0,   0),   # 1
    (  0, 128,   0),   # 2
    (128, 128,   0),   # 3
    (  0,   0, 128),   # 4
    (128,   0, 128),   # 5
    (  0, 128, 128),   # 6
    (128, 128, 128),   # 7
    ( 64,   0,   0),   # 8
    (192,   0,   0),   # 9
    ( 64, 128,   0),   # 10
]


# ── SOTA table data ───────────────────────────────────────────────────────────

class SOTAEntry(NamedTuple):
    method: str
    params: str
    temporal: str
    bl30k: bool
    dav17: str
    ytv19: str
    lvos_v2: str
    note: str


_SOTA_LIGHTWEIGHT: list[SOTAEntry] = [
    SOTAEntry("TrickVOS (PT)",    "~5M",   "STM Attn",    True,  "82.7", "80.5", "—",  ""),
    SOTAEntry("WarpFormer-S",     "~8M",   "Flow+Attn",   True,  "81.0", "80.1", "—",  ""),
    SOTAEntry("AOTT",             "~8M",   "LSTT Attn",   True,  "79.2", "80.0", "—",  ""),
    SOTAEntry("MobileVOS",        "~5M",   "Transformer", False, "77.8", "81.8", "—",  ""),
]

_SOTA_REFERENCE: list[SOTAEntry] = [
    SOTAEntry("SAM 2 (Large)",    "~224M", "Mem. Attn",   True,  "92.3", "89.3", "79.8", ""),
    SOTAEntry("Cutie",            "~42M",  "Transformer", True,  "84.3", "82.5", "~71–72", ""),
    SOTAEntry("LiVOS (RN-50)",    "~35M",  "Linear Attn", False, "85.1", "83.6", "44.6", "LVOS-v2"),
]


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _overlay_mask_on_frame(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Alpha-blend coloured object masks onto a BGR frame.

    Args:
        frame_rgb: uint8 ``[H, W, 3]`` RGB image.
        mask:      uint8 ``[H, W]`` integer label mask (0=background).
        alpha:     Opacity of the mask overlay (0=transparent, 1=opaque).

    Returns:
        uint8 ``[H, W, 3]`` RGB composite.
    """
    out = frame_rgb.astype(np.float32)
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue
        colour = _PALETTE_RGB[min(int(obj_id), len(_PALETTE_RGB) - 1)]
        region = mask == obj_id
        for c, val in enumerate(colour):
            out[region, c] = (1 - alpha) * out[region, c] + alpha * val
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_caption(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int] = (6, 6),
    font_size: int = 16,
) -> np.ndarray:
    """Draw a one-line caption with a semi-transparent background.

    Args:
        frame:     uint8 ``[H, W, 3]`` RGB image (modified in-place).
        text:      Caption string.
        pos:       Top-left pixel (x, y).
        font_size: Approximate font size in points (PIL default fallback used
                   if a truetype font is unavailable).

    Returns:
        Annotated copy of the frame.
    """
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox(pos, text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0, 160),
    )
    draw.text(pos, text, fill=(255, 255, 255, 255), font=font)
    return np.array(pil.convert("RGB"))


# ── GIF writer ────────────────────────────────────────────────────────────────

def _make_gif(
    video_id: str,
    scores: dict,
    lvos_root: Path,
    pred_ann_dir: Path,
    output_path: Path,
    fps: int,
    max_frames: int,
    alpha: float,
) -> None:
    """Generate an animated GIF for a single LVOS sequence.

    Each frame shows the original JPEG with the predicted mask overlaid.
    Annotated frames get an extra outline around the caption to signal that
    a GT evaluation was performed on that frame.

    Args:
        video_id:     LVOS sequence folder name.
        scores:       Dict with keys ``J``, ``F``, ``J&F``, ``annotated_frames``.
        lvos_root:    Root of the LVOS dataset (contains ``train/JPEGImages/``).
        pred_ann_dir: Directory that holds ``<video_id>/*.png`` prediction masks.
        output_path:  Destination ``.gif`` file path.
        fps:          Animation frame rate.
        max_frames:   Maximum number of frames to include (subsampled if needed).
        alpha:        Mask overlay opacity.
    """
    img_dir  = lvos_root / "train" / "JPEGImages"  / video_id
    ann_dir  = lvos_root / "train" / "Annotations" / video_id
    pred_dir = pred_ann_dir / video_id

    if not img_dir.is_dir():
        LOGGER.warning("[%s] image dir not found — skipping GIF.", video_id)
        return
    if not pred_dir.is_dir():
        LOGGER.warning("[%s] prediction dir not found at %s — skipping GIF.", video_id, pred_dir)
        return

    frame_paths = sorted(img_dir.glob("*.jpg"), key=lambda p: p.stem)
    if not frame_paths:
        LOGGER.warning("[%s] no JPEG frames — skipping GIF.", video_id)
        return

    # GT annotation stems (for marking annotated frames in the caption).
    gt_stems = {p.stem for p in ann_dir.glob("*.png")} if ann_dir.is_dir() else set()

    # Subsample frames so GIF stays under max_frames length.
    step = max(1, len(frame_paths) // max_frames)
    frame_paths = frame_paths[::step]

    duration_ms = max(40, round(1000 / fps))
    pil_frames: list[Image.Image] = []

    jf_str = f"J&F={scores['J&F']:.3f}  J={scores['J']:.3f}  F={scores['F']:.3f}"

    for frame_path in frame_paths:
        frame_rgb = np.array(Image.open(frame_path).convert("RGB"), dtype=np.uint8)

        # Load prediction mask if available.
        pred_name = frame_path.stem + ".png"
        pred_path = pred_dir / pred_name
        if pred_path.exists():
            pred_pil = Image.open(pred_path).convert("P")
            pred_mask = np.array(pred_pil, dtype=np.uint8)
            if pred_mask.shape != frame_rgb.shape[:2]:
                pred_pil = pred_pil.resize(
                    (frame_rgb.shape[1], frame_rgb.shape[0]), Image.NEAREST
                )
                pred_mask = np.array(pred_pil, dtype=np.uint8)
            composite = _overlay_mask_on_frame(frame_rgb, pred_mask, alpha)
        else:
            composite = frame_rgb

        is_annotated = frame_path.stem in gt_stems
        ann_marker   = " [GT]" if is_annotated else ""
        caption      = f"{video_id}  {jf_str}{ann_marker}"
        composite    = _add_caption(composite, caption)

        pil_frames.append(Image.fromarray(composite))

    if not pil_frames:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    LOGGER.info("Saved GIF → %s  (%d frames)", output_path, len(pil_frames))


# ── SOTA table printer ────────────────────────────────────────────────────────

def _print_sota_table(ours_jf: float | None, ours_dav: float | None, ours_ytv: float | None) -> None:
    """Print a SOTA comparison table to stdout.

    Args:
        ours_jf:  Our mean J&F on LVOS V2 train set (or None if not available).
        ours_dav: Our DAVIS 2017 val J&F (or None).
        ours_ytv: Our YouTube-VOS 2019 val J&F (or None).
    """
    ours_dav_str  = f"{ours_dav:.1f}"  if ours_dav  is not None else "—"
    ours_ytv_str  = f"{ours_ytv:.1f}"  if ours_ytv  is not None else "—"
    ours_lvos_str = f"{ours_jf*100:.1f}" if ours_jf is not None else "—"

    ours = SOTAEntry(
        "VideoKamba (Ours)", "~5M", "KAN-SSM", True,
        ours_dav_str, ours_ytv_str, ours_lvos_str, "★ DTSM enabled",
    )

    header = f"{'Method':<22} {'Params':>7}  {'BL30K':>5}  {'DAV-17':>7}  {'YTV-19':>7}  {'LVOS-v2':>8}  Note"
    sep    = "─" * len(header)

    print()
    print("=" * len(header))
    print("  VideoKamba — SOTA Comparison (LVOS V2 train-set evaluation)")
    print("=" * len(header))
    print()
    print("  Ultra-lightweight (≤10M params, MobileNetV2 backbone)")
    print(sep)
    print(header)
    print(sep)

    for entry in _SOTA_LIGHTWEIGHT:
        bl = "✓" if entry.bl30k else "✗"
        print(
            f"  {entry.method:<20} {entry.params:>7}  {bl:>5}  {entry.dav17:>7}  "
            f"{entry.ytv19:>7}  {entry.lvos_v2:>8}  {entry.note}"
        )
    bl = "✓" if ours.bl30k else "✗"
    print(sep)
    print(
        f"  {ours.method:<20} {ours.params:>7}  {bl:>5}  {ours.dav17:>7}  "
        f"{ours.ytv19:>7}  {ours.lvos_v2:>8}  {ours.note}"
    )
    print(sep)

    print()
    print("  Reference (large models, not target regime)")
    print(sep)
    print(header)
    print(sep)
    for entry in _SOTA_REFERENCE:
        bl = "✓" if entry.bl30k else "✗"
        print(
            f"  {entry.method:<20} {entry.params:>7}  {bl:>5}  {entry.dav17:>7}  "
            f"{entry.ytv19:>7}  {entry.lvos_v2:>8}  {entry.note}"
        )
    print(sep)
    print()
    print("  Metrics: J&F score on respective val sets (higher is better).")
    print("  LVOS-v2 column: J&F on validation split; '—' = not reported.")
    print("  Our LVOS-v2 column shows train-set J&F (proxy before val submission).")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_latest_results(artifacts_dir: Path) -> Path | None:
    """Return the most recently modified results JSON in a directory tree."""
    candidates = sorted(artifacts_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> None:
    """Entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Visualize LVOS eval results: GIFs + SOTA table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results",     type=Path, default=None,
                        help="Path to results JSON from eval_lvos_train.py. "
                             "Auto-detected from --output-dir if omitted.")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Path to saved prediction PNGs directory. "
                             "Defaults to <results_parent>/Annotations/.")
    parser.add_argument("--lvos-root",   type=Path, default=Path("data/LVOS"),
                        help="LVOS dataset root (contains train/JPEGImages/).")
    parser.add_argument("--output-dir",  type=Path, default=Path("artifacts/lvos_gifs"),
                        help="Where to write GIFs and the comparison text.")
    parser.add_argument("--top-n",       type=int,  default=5,
                        help="Number of sequences for best/worst/middle each.")
    parser.add_argument("--fps",         type=int,  default=6,
                        help="GIF animation frame rate.")
    parser.add_argument("--max-frames",  type=int,  default=120,
                        help="Maximum frames per GIF (subsampled if sequence is longer).")
    parser.add_argument("--alpha",       type=float, default=0.55,
                        help="Mask overlay opacity [0, 1].")
    parser.add_argument("--davis-jf",    type=float, default=None,
                        help="Our DAVIS 2017 val J&F (if known) for SOTA table.")
    parser.add_argument("--ytv-jf",      type=float, default=None,
                        help="Our YouTube-VOS 2019 val J&F (if known) for SOTA table.")
    args = parser.parse_args(argv)

    # ── Resolve results JSON ──────────────────────────────────────────────────
    results_path: Path | None = args.results
    if results_path is None:
        results_path = _find_latest_results(args.output_dir.parent / "lvos_eval")
        if results_path is None:
            LOGGER.error(
                "No results JSON found. Run eval_lvos_train.py first, "
                "or pass --results <path>."
            )
            sys.exit(1)
        LOGGER.info("Auto-detected results: %s", results_path)

    with open(results_path) as fh:
        data = json.load(fh)

    per_seq: dict[str, dict] = data["per_sequence"]
    if not per_seq:
        LOGGER.error("results JSON contains no per-sequence data.")
        sys.exit(1)

    LOGGER.info(
        "Loaded %d sequences — mean J=%.4f  F=%.4f  J&F=%.4f",
        data["num_sequences"], data["mean_J"], data["mean_F"], data["mean_J&F"],
    )

    # ── Resolve predictions directory ─────────────────────────────────────────
    pred_ann_dir: Path = (
        args.predictions if args.predictions is not None
        else results_path.parent / "Annotations"
    )
    if not pred_ann_dir.is_dir():
        LOGGER.warning(
            "Predictions directory not found at %s. GIFs will show frames only.",
            pred_ann_dir,
        )

    # ── Rank sequences by J&F ─────────────────────────────────────────────────
    ranked: list[tuple[str, float]] = sorted(
        [(vid, info["J&F"]) for vid, info in per_seq.items()],
        key=lambda x: x[1],
    )
    n = len(ranked)
    top_n  = args.top_n
    mid_n  = args.top_n

    best_seqs   = [vid for vid, _ in ranked[-(top_n):]][::-1]       # descending
    worst_seqs  = [vid for vid, _ in ranked[: top_n]]               # ascending
    mid_start   = max(0, n // 2 - mid_n // 2)
    middle_seqs = [vid for vid, _ in ranked[mid_start: mid_start + mid_n]]

    LOGGER.info(
        "Best  (%d): %s", top_n, ", ".join(f"{v}({per_seq[v]['J&F']:.3f})" for v in best_seqs)
    )
    LOGGER.info(
        "Worst (%d): %s", top_n, ", ".join(f"{v}({per_seq[v]['J&F']:.3f})" for v in worst_seqs)
    )
    LOGGER.info(
        "Middle(%d): %s", mid_n, ", ".join(f"{v}({per_seq[v]['J&F']:.3f})" for v in middle_seqs)
    )

    # ── Generate GIFs ─────────────────────────────────────────────────────────
    groups = [
        ("best",   best_seqs),
        ("worst",  worst_seqs),
        ("middle", middle_seqs),
    ]

    for group_name, seq_ids in groups:
        out_subdir = args.output_dir / group_name
        for rank_idx, video_id in enumerate(seq_ids, start=1):
            scores  = per_seq[video_id]
            jf_val  = scores["J&F"]
            gif_name = f"{rank_idx:02d}_{video_id}_JF{jf_val:.3f}.gif"
            _make_gif(
                video_id=video_id,
                scores=scores,
                lvos_root=args.lvos_root,
                pred_ann_dir=pred_ann_dir,
                output_path=out_subdir / gif_name,
                fps=args.fps,
                max_frames=args.max_frames,
                alpha=args.alpha,
            )

    # ── Print per-sequence ranking ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Per-sequence J&F ranking (all sequences)")
    print("=" * 70)
    print(f"  {'Rank':>4}  {'Video ID':<20}  {'J&F':>6}  {'J':>6}  {'F':>6}  {'ann_frames':>10}")
    print("  " + "─" * 66)
    for rank, (vid, jf) in enumerate(reversed(ranked), start=1):
        info = per_seq[vid]
        print(
            f"  {rank:>4}  {vid:<20}  {info['J&F']:>6.3f}  "
            f"{info['J']:>6.3f}  {info['F']:>6.3f}  {info['annotated_frames']:>10d}"
        )
    print()

    # ── Print SOTA table ───────────────────────────────────────────────────────
    _print_sota_table(
        ours_jf=data["mean_J&F"],
        ours_dav=args.davis_jf,
        ours_ytv=args.ytv_jf,
    )

    # ── Save comparison as text ────────────────────────────────────────────────
    sota_txt = args.output_dir / "sota_comparison.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_sota_table(
            ours_jf=data["mean_J&F"],
            ours_dav=args.davis_jf,
            ours_ytv=args.ytv_jf,
        )
    sota_txt.write_text(buf.getvalue())
    LOGGER.info("SOTA table saved → %s", sota_txt)


if __name__ == "__main__":
    main()
