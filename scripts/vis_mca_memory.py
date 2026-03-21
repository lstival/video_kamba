"""Visualise the MCA memory dynamic frame alongside input frames.

Replicates Fig. 3 from the paper:

    "Interpreting the dynamic frame of the MCA memory reveals its ability
    to encode temporal smoothness over time."

For each query frame the script captures the cross-attention weight tensor
from ``PropagationAttention`` via a non-invasive forward hook, computes a
spatial saliency map by averaging over heads and summing over memory slots,
and renders a two-row grid (RGB frames / attention heatmaps) annotated with
a time arrow — matching the paper figure exactly.

Hook point
----------
``model.propagation_attention.attn_drop`` receives ``attn`` of shape
``[B, n_heads, P_query, P_mem]`` (fp32, post-softmax).  In eval mode
``nn.Dropout`` is identity, so hook output == attention weights exactly.

Usage (interactive / sbatch)
-----------------------------
    python scripts/vis_mca_memory.py \\
        --checkpoint checkpoints/best_mv2_phase2_davis.ckpt \\
        --sequences blackswan camel \\
        --n_frames 5 \\
        --output vis_mca_memory/

Note: frame indices are 1-based query frames (frame 0 is always the reference).
"""

from __future__ import annotations

import argparse
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from jaxtyping import Float
from PIL import Image
from torch import Tensor

# ── Local ────────────────────────────────────────────────────────────────────
from models.video_mamba import VideoMambaSystem

# ── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_frame(path: Path, img_size: int) -> Float[Tensor, "3 H W"]:
    """Load a JPEG frame, resize, and apply ImageNet normalisation."""
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [img_size, img_size])
    tensor = TF.to_tensor(img)  # [3, H, W] in [0, 1]
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def load_annotation(path: Path, img_size: int) -> Float[Tensor, "H W"]:
    """Load a DAVIS palette-PNG annotation and resize (nearest) to ``[H, W]``.

    DAVIS annotations are palette PNGs where pixel values are object IDs
    (0 = background, 1 = object 1, …).  PIL returns palette indices directly
    when the image is kept in mode ``P``.
    """
    mask = Image.open(path)  # mode P — palette indices == object IDs
    mask = TF.resize(
        mask, [img_size, img_size], interpolation=TF.InterpolationMode.NEAREST
    )
    return torch.from_numpy(np.array(mask)).long()


def load_sequence(
    davis_root: Path,
    sequence: str,
    n_query_frames: int,
    img_size: int,
) -> tuple[
    Float[Tensor, "3 H W"],       # ref_frame
    Float[Tensor, "H W"],         # ref_mask
    Float[Tensor, "T 3 H W"],     # query_frames
    list[Path],                    # query_frame_paths (for display)
]:
    """Load one DAVIS sequence: reference frame + *n_query_frames* query frames.

    Args:
        davis_root:     Root of the DAVIS dataset (contains JPEGImages/).
        sequence:       Sequence name, e.g. ``"blackswan"``.
        n_query_frames: Number of query frames to load (after frame 0).
        img_size:       Spatial resolution to resize all frames to.

    Returns:
        Tuple of (ref_frame, ref_mask, query_frames, query_frame_paths).

    Raises:
        FileNotFoundError: If the sequence directory does not exist.
        AssertionError:    If the sequence has fewer frames than requested.
    """
    img_dir = davis_root / "JPEGImages" / "480p" / sequence
    ann_dir = davis_root / "Annotations" / "480p" / sequence

    if not img_dir.exists():
        raise FileNotFoundError(f"DAVIS sequence not found: {img_dir}")

    frame_paths = sorted(img_dir.glob("*.jpg"))
    ann_paths = sorted(ann_dir.glob("*.png"))

    assert len(frame_paths) >= n_query_frames + 1, (
        f"Sequence '{sequence}' has {len(frame_paths)} frames; "
        f"need at least {n_query_frames + 1} (1 ref + {n_query_frames} query)."
    )

    ref_frame = load_frame(frame_paths[0], img_size)
    ref_mask = load_annotation(ann_paths[0], img_size)

    query_frame_paths = frame_paths[1 : n_query_frames + 1]
    query_frames = torch.stack([load_frame(p, img_size) for p in query_frame_paths])

    return ref_frame, ref_mask, query_frames, query_frame_paths


# ─────────────────────────────────────────────────────────────────────────────
# Attention capture
# ─────────────────────────────────────────────────────────────────────────────

class AttentionCapture:
    """Accumulates attention tensors emitted by ``PropagationAttention.attn_drop``.

    In eval mode ``nn.Dropout`` is identity: the module's output equals its
    input, which is the post-softmax attention weight ``attn`` of shape
    ``[B, n_heads, P_query, P_mem]``.

    Attributes:
        weights: List of captured tensors, one per forward call (= one per
                 query frame in the VOS loop).
    """

    def __init__(self) -> None:
        self.weights: list[Float[Tensor, "B n_heads P_q P_mem"]] = []

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: tuple[Tensor, ...],
        output: Float[Tensor, "B n_heads P_q P_mem"],
    ) -> None:
        self.weights.append(output.detach().float().cpu())

    def clear(self) -> None:
        self.weights.clear()


@contextmanager
def capture_attention(
    model: VideoMambaSystem,
) -> Generator[AttentionCapture, None, None]:
    """Context manager that registers/removes the attention hook safely.

    Yields:
        An :class:`AttentionCapture` instance populated after the forward pass.

    Example::

        with capture_attention(model) as cap:
            model(frames, ref_frame=ref, ref_mask=mask)
        heatmaps = build_heatmaps(cap.weights, feat_h, feat_w, frame_h, frame_w)
    """
    capture = AttentionCapture()
    handle = model.propagation_attention.attn_drop.register_forward_hook(capture)
    try:
        yield capture
    finally:
        handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap computation
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmaps(
    attention_weights: list[Float[Tensor, "1 n_heads P_q P_mem"]],
    target_h: int,
    target_w: int,
) -> list[np.ndarray]:
    """Convert per-frame attention tensors to spatial heatmaps.

    For each frame:

    1. Average over attention heads → ``[P_q, P_mem]``
    2. Sum over memory slots → ``[P_q]``  (total attention drawn per query patch)
    3. Infer spatial grid from ``P_q`` (assumes square patch layout)
    4. Upsample bilinearly to ``(target_h, target_w)``
    5. Min-max normalise to ``[0, 1]``

    Args:
        attention_weights: List of T tensors ``[1, n_heads, P_q, P_mem]``.
        target_h:          Target heatmap height (= frame height).
        target_w:          Target heatmap width  (= frame width).

    Returns:
        List of T numpy arrays of shape ``[target_h, target_w]`` in ``[0, 1]``.
    """
    heatmaps: list[np.ndarray] = []

    for attn in attention_weights:
        # attn: [1, n_heads, P_q, P_mem]
        saliency = attn[0].mean(dim=0).sum(dim=-1)  # [P_q]

        p_q = saliency.shape[0]
        feat_h = feat_w = int(math.isqrt(p_q))
        assert feat_h * feat_w == p_q, (
            f"Non-square patch grid: P_q={p_q} is not a perfect square. "
            f"Rectangular grids are not yet supported."
        )

        spatial = saliency.reshape(1, 1, feat_h, feat_w)  # [1, 1, fH, fW]
        upsampled = F.interpolate(
            spatial, size=(target_h, target_w), mode="bilinear", align_corners=False
        ).squeeze()  # [target_h, target_w]

        hi, lo = upsampled.max(), upsampled.min()
        normalised = ((upsampled - lo) / (hi - lo + 1e-8)).numpy()
        heatmaps.append(normalised)

    return heatmaps


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _denormalise(
    tensor: Float[Tensor, "3 H W"],
) -> np.ndarray:
    """Reverse ImageNet normalisation → ``[H, W, 3]`` numpy array in ``[0, 1]``."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def render_figure(
    query_frames: Float[Tensor, "T 3 H W"],
    heatmaps: list[np.ndarray],
    sequence: str,
    output_path: Path,
    col_width: float = 3.0,
) -> None:
    """Render the Fig. 3-style two-row grid and save to disk.

    Layout::

        ←──────────────── T ──────────────────►
        [ frame_0 ]  [ frame_1 ]  …  [ frame_n ]   ← Image row
        [ heat_0  ]  [ heat_1  ]  …  [ heat_n  ]   ← MCA Memory row

    Args:
        query_frames: Normalised query frames ``[T, 3, H, W]``.
        heatmaps:     Per-frame heatmaps, list of ``[H, W]`` arrays in ``[0,1]``.
        sequence:     Sequence name, used in the figure title and file name.
        output_path:  Full path (including filename) for the saved PNG.
        col_width:    Width of each column in inches.
    """
    n_frames = query_frames.shape[0]
    assert len(heatmaps) == n_frames, (
        f"Frame / heatmap count mismatch: {n_frames} vs {len(heatmaps)}"
    )

    fig_w = col_width * n_frames + 0.8  # +0.8 for row labels
    fig_h = col_width * 2 + 0.6        # 2 rows + space for time arrow
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # ── Time arrow spanning the full top ────────────────────────────────────
    ax_arrow = fig.add_axes([0.08, 0.93, 0.88, 0.04])
    ax_arrow.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
    )
    ax_arrow.text(
        0.005, 0.55, r"$T$",
        transform=ax_arrow.transAxes,
        fontsize=14, va="bottom", ha="left",
    )
    ax_arrow.axis("off")

    # ── Image + heatmap grid ─────────────────────────────────────────────────
    gs = gridspec.GridSpec(
        2, n_frames,
        figure=fig,
        top=0.89, bottom=0.02,
        left=0.10, right=0.98,
        hspace=0.04, wspace=0.04,
    )

    row_labels = ["Image", "MCA\nMemory"]
    cmap = plt.cm.jet

    for row, label in enumerate(row_labels):
        for col in range(n_frames):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

            if row == 0:
                ax.imshow(_denormalise(query_frames[col].cpu()), aspect="auto")
            else:
                ax.imshow(
                    heatmaps[col],
                    cmap=cmap,
                    vmin=0.0, vmax=1.0,
                    interpolation="bilinear",
                    aspect="auto",
                )

            if col == 0:
                ax.set_ylabel(
                    label,
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=42,
                    va="center",
                    ha="right",
                )
                ax.yaxis.set_label_coords(-0.18, 0.5)
                ax.set_ylabel(label, fontsize=10, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def visualise_sequence(
    model: VideoMambaSystem,
    davis_root: Path,
    sequence: str,
    n_frames: int,
    img_size: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Run the full pipeline for a single DAVIS sequence.

    Args:
        model:      Loaded, eval-mode ``VideoMambaSystem``.
        davis_root: DAVIS dataset root.
        sequence:   Sequence name (e.g. ``"blackswan"``).
        n_frames:   Number of query frames to process.
        img_size:   Spatial resolution for frames and masks.
        output_dir: Directory to write the output PNG.
        device:     Torch device for inference.
    """
    log.info("Processing sequence: %s (%d query frames)", sequence, n_frames)

    ref_frame, ref_mask, query_frames, _ = load_sequence(
        davis_root, sequence, n_frames, img_size
    )

    ref_frame_b = ref_frame.unsqueeze(0).to(device)    # [1, 3, H, W]
    ref_mask_b = ref_mask.unsqueeze(0).to(device)       # [1, H, W]
    query_b = query_frames.unsqueeze(0).to(device)      # [1, T, 3, H, W]

    with capture_attention(model) as cap:
        with torch.no_grad():
            model(
                query_b,
                ref_frame=ref_frame_b,
                ref_mask=ref_mask_b,
            )

    assert len(cap.weights) == n_frames, (
        f"Hook captured {len(cap.weights)} maps, expected {n_frames}. "
        "Check that propagation_attention.attn_drop exists in the model."
    )

    heatmaps = build_heatmaps(cap.weights, target_h=img_size, target_w=img_size)

    output_path = output_dir / f"mca_memory_{sequence}.png"
    render_figure(query_frames, heatmaps, sequence, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise MCA memory dynamic frame (Fig. 3 style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to Lightning .ckpt file.",
    )
    parser.add_argument(
        "--sequences", nargs="+", default=["blackswan"],
        help="One or more DAVIS sequence names to process.",
    )
    parser.add_argument(
        "--n_frames", type=int, default=5,
        help="Number of query frames to visualise per sequence.",
    )
    parser.add_argument(
        "--img_size", type=int, default=480,
        help="Spatial resolution to resize frames and masks to.",
    )
    parser.add_argument(
        "--output", default="vis_mca_memory",
        help="Output directory for PNG files.",
    )
    parser.add_argument(
        "--davis_root", default="data/DAVIS/DAVIS",
        help="Path to the DAVIS dataset root directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading checkpoint: %s", args.checkpoint)
    # weights_only=False: Lightning checkpoints embed OmegaConf DictConfig /
    # ContainerMetadata objects that PyTorch ≥ 2.6 rejects under the new
    # weights_only=True default.  We own these checkpoints, so False is safe.
    model: VideoMambaSystem = VideoMambaSystem.load_from_checkpoint(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.eval().to(device)

    davis_root = Path(args.davis_root)
    output_dir = Path(args.output)

    for sequence in args.sequences:
        try:
            visualise_sequence(
                model=model,
                davis_root=davis_root,
                sequence=sequence,
                n_frames=args.n_frames,
                img_size=args.img_size,
                output_dir=output_dir,
                device=device,
            )
        except (FileNotFoundError, AssertionError) as exc:
            log.error("Skipping '%s': %s", sequence, exc)


if __name__ == "__main__":
    main()
