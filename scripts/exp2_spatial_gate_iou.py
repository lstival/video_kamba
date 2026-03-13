"""Experiment 2 — Spatial Gate Fidelity  (IoU_gate, Geometric Proof).

Measures how well the internal spatial gate G_k from ``KANSpatialGatingUpBlock``
aligns with the ground-truth object mask, compared to averaged cross-attention
maps from a ``CrossAttentionUpBlock`` baseline.

Mathematical formulation
------------------------
Let G_bar_k be the channel-averaged gate::

    G_bar_k = (1/C) * sum_c G_{k,c}    in [0, 1]^{H x W}

Binarise with threshold tau::

    B_k = {G_bar_k > tau}

Intersection over Union against the GT mask M in {0,1}^{H x W}::

    IoU_gate = |B_k ∩ M| / |B_k ∪ M|

The final score is the arithmetic mean over all three pyramid levels
(up1: 14x14, up2: 28x28, up3: 56x56) and all validation frames.

Usage
-----
From the project root::

    python scripts/exp2_spatial_gate_iou.py \\
        kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \\
        xattn_ckpt=lightning_logs/version_xattn/checkpoints/best.ckpt \\
        data_dir=data/DAVIS/DAVIS \\
        output_dir=results/exp2

Authors: KANGA Project
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Ensure project root is on sys.path when executed as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.davis import DAVISDataModule
from models.components.segmentation_decoder import (
    CrossAttentionUpBlock,
    KANSpatialGatingUpBlock,
    SegmentationDecoder,
)
from models.video_mamba import VideoMambaSystem

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hydra structured config
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class Exp2Config:
    """Configuration for Experiment 2 — Spatial Gate Fidelity."""

    kan_ckpt: str = "???"
    xattn_ckpt: str = "???"
    data_dir: str = "data/DAVIS/DAVIS"
    output_dir: str = "results/exp2"
    tau: float = 0.5
    pyramid_levels: List[int] = field(default_factory=lambda: [0, 1, 2])
    save_visualizations: bool = True
    vis_max_seqs: int = 10
    batch_size: int = 1
    num_workers: int = 4
    img_size: int = 224
    seq_len: int = 8
    max_batches: int = -1
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="exp2_config", node=Exp2Config)


# ---------------------------------------------------------------------------
# Gate extraction utilities
# ---------------------------------------------------------------------------

def enable_gate_caching(decoder: SegmentationDecoder) -> None:
    """Enable ``return_gate=True`` on all up-blocks that support it.

    Args:
        decoder: The ``SegmentationDecoder`` from a loaded model.
    """
    for block in [decoder.up1, decoder.up2, decoder.up3]:
        if isinstance(block, (KANSpatialGatingUpBlock, CrossAttentionUpBlock)):
            block.return_gate = True
    log.info("Gate caching enabled on decoder up-blocks.")


def collect_gate_maps(
    decoder: SegmentationDecoder,
) -> List[Optional[torch.Tensor]]:
    """Read cached gate / attention tensors from each decoder level.

    Returns a 3-element list (one per pyramid level).  Each element is

    * ``[BT, C, H, W]`` for KAN (``_last_gate``),
    * ``[B, tgt_len, src_len]`` for cross-attention (``_last_attn``),
    * ``None`` if the block type has no cached gate.
    """
    gates: List[Optional[torch.Tensor]] = []
    for block in [decoder.up1, decoder.up2, decoder.up3]:
        if isinstance(block, KANSpatialGatingUpBlock):
            gates.append(block._last_gate)
        elif isinstance(block, CrossAttentionUpBlock):
            gates.append(block._last_attn)
        else:
            gates.append(None)
    return gates


def gate_to_spatial_map(
    gate: torch.Tensor,
    target_hw: Tuple[int, int],
    model_type: str,
    bt: int,
) -> torch.Tensor:
    """Convert raw gate/attention tensor to a 2D spatial map in [0, 1].

    For KAN gates (``[BT, C, H, W]``): channel-mean, then bilinear resize.
    For cross-attention maps (``[B, N_q, N_kv]``): average over query tokens,
    reshape to grid, then bilinear resize.

    Args:
        gate: Raw gate tensor.
        target_hw: Target (H, W) for the output map.
        model_type: ``'kan'`` or ``'xattn'``.
        bt: Expected leading batch dimension after flattening B*T.

    Returns:
        ``[BT, 1, H_target, W_target]`` spatial map in ``[0, 1]``.
    """
    if model_type == "kan":
        # gate: [BT, C, H, W] — channel-average
        spatial = gate.mean(dim=1, keepdim=True)  # [BT, 1, H, W]
    elif model_type == "xattn":
        # gate: [B, N_q, N_kv] — average over query positions
        B_attn, N_q, N_kv = gate.shape
        spatial_flat = gate.mean(dim=1)  # [B, N_kv]
        h_s = w_s = int(N_kv ** 0.5)
        spatial = spatial_flat.view(B_attn, 1, h_s, w_s)  # [B, 1, h, w]
        # Expand to BT by repeating (assumes same map for all T frames per batch)
        T = bt // B_attn if B_attn > 0 else 1
        spatial = spatial.repeat_interleave(T, dim=0)  # [BT, 1, h, w]
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Normalise to [0, 1]
    s_min = spatial.flatten(2).min(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    s_max = spatial.flatten(2).max(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    spatial = (spatial - s_min) / (s_max - s_min + 1e-8)

    # Resize to target
    spatial = F.interpolate(
        spatial.float(), size=target_hw, mode="bilinear", align_corners=False
    )  # [BT, 1, H_t, W_t]
    return spatial


def compute_iou_gate(
    gate_map: torch.Tensor,
    gt_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute IoU_gate for a batch of spatial maps against GT masks.

    Args:
        gate_map: ``[N, 1, H, W]`` — normalised gate map.
        gt_mask: ``[N, H, W]`` — binary ground-truth foreground mask.
        tau: Binarisation threshold.

    Returns:
        ``[N]`` per-sample IoU_gate values.
    """
    binary_gate = (gate_map.squeeze(1) > tau)      # [N, H, W] bool
    binary_mask = (gt_mask > 0)                    # [N, H, W] bool

    intersection = (binary_gate & binary_mask).float().sum(dim=(1, 2))   # [N]
    union = (binary_gate | binary_mask).float().sum(dim=(1, 2))           # [N]

    iou = intersection / (union + 1e-8)  # [N]
    return iou


def save_overlay_visualization(
    frame: torch.Tensor,
    gate_map: torch.Tensor,
    gt_mask: torch.Tensor,
    save_path: Path,
) -> None:
    """Save an overlay image of the gate heatmap on top of the video frame.

    Args:
        frame: ``[3, H, W]`` float tensor in [0, 1].
        gate_map: ``[1, H, W]`` gate heatmap in [0, 1].
        gt_mask: ``[H, W]`` integer segmentation mask.
        save_path: Destination PNG file path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    img_np = frame.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    gate_np = gate_map.squeeze(0).cpu().numpy()
    mask_np = (gt_mask.cpu().numpy() > 0).astype(float)

    axes[0].imshow(img_np)
    axes[0].set_title("Input Frame")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(gate_np, alpha=0.5, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Gate Heatmap G_k overlay")
    axes[1].axis("off")

    axes[2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground-truth Mask")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main inference + metric loop
# ---------------------------------------------------------------------------

def run_iou_evaluation(
    model: VideoMambaSystem,
    dataloader: torch.utils.data.DataLoader,
    model_type: str,
    cfg: DictConfig,
    device: torch.device,
    vis_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run inference and collect per-level IoU_gate for one model.

    Args:
        model: Loaded VideoMambaSystem.
        dataloader: DAVIS validation dataloader.
        model_type: ``'kan'`` or ``'xattn'``.
        cfg: Experiment config.
        device: Computation device.
        vis_dir: If not None, save per-sequence overlay images here.

    Returns:
        DataFrame with columns ``[frame_idx, level, iou_gate]``.
    """
    decoder: SegmentationDecoder = model.seg_decoder
    enable_gate_caching(decoder)

    records: List[Dict] = []
    global_frame_idx = 0
    vis_seq_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if cfg.max_batches > 0 and batch_idx >= cfg.max_batches:
                break

            ref_img, ref_mask, query_images, query_masks = (
                t.to(device) for t in batch[:4]
            )
            B, T, _, H, W = query_images.shape
            BT = B * T

            # Forward pass — gates are populated into _last_gate / _last_attn
            model(query_images, ref_frame=ref_img, ref_mask=ref_mask)

            gates = collect_gate_maps(decoder)
            gt_masks_flat = query_masks.reshape(BT, H, W)  # [BT, H, W]

            # ---- per-level IoU computation ----
            level_ious: Dict[int, torch.Tensor] = {}
            for lvl_idx in cfg.pyramid_levels:
                gate = gates[lvl_idx]
                if gate is None:
                    continue

                # Resize GT mask to match pyramid resolution
                # Typical DINO patch=14: up1→14x14, up2→28x28, up3→56x56
                py_resolutions = [14, 28, 56]
                hw_lvl = (py_resolutions[lvl_idx], py_resolutions[lvl_idx])

                gt_resized = F.interpolate(
                    gt_masks_flat.unsqueeze(1).float(),
                    size=hw_lvl,
                    mode="nearest",
                ).squeeze(1)  # [BT, h, w]

                spatial_map = gate_to_spatial_map(gate, hw_lvl, model_type, BT)
                iou_vals = compute_iou_gate(spatial_map, gt_resized, tau=cfg.tau)
                level_ious[lvl_idx] = iou_vals

                for i, v in enumerate(iou_vals.cpu().numpy()):
                    records.append({
                        "frame_idx": global_frame_idx + i,
                        "level": lvl_idx,
                        "iou_gate": float(v),
                        "model": model_type,
                    })

            # ---- optional visualisation (first frame of batch) ----
            if vis_dir is not None and vis_seq_count < cfg.vis_max_seqs:
                lvl = cfg.pyramid_levels[-1]  # use finest level
                if gates[lvl] is not None:
                    spatial_map = gate_to_spatial_map(gates[lvl], (H, W), model_type, BT)
                    save_overlay_visualization(
                        frame=query_images[0, 0],          # first seq, first frame
                        gate_map=spatial_map[0],
                        gt_mask=gt_masks_flat[0],
                        save_path=vis_dir / f"seq{vis_seq_count:04d}_{model_type}.png",
                    )
                    vis_seq_count += 1

            global_frame_idx += BT

            if (batch_idx + 1) % 20 == 0:
                log.info("  [%s] processed %d batches", model_type, batch_idx + 1)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/analysis", config_name="exp2")
def main(cfg: DictConfig) -> None:  # noqa: D103
    OmegaConf.to_yaml(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if cfg.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm = DAVISDataModule(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seq_len=cfg.seq_len,
        img_size=cfg.img_size,
        test_split="val",
    )
    dm.setup(stage="test")
    val_loader = dm.test_dataloader()

    # ------------------------------------------------------------------
    # Evaluate both models
    # ------------------------------------------------------------------
    all_records = []
    model_configs = [
        ("kan", cfg.kan_ckpt, None),
        ("xattn", cfg.xattn_ckpt, vis_dir if cfg.save_visualizations else None),
    ]

    for model_type, ckpt, vis_d in model_configs:
        log.info("--- Evaluating %s model ---", model_type)
        model = VideoMambaSystem.load_from_checkpoint(ckpt, map_location=device)
        model.to(device).eval()
        df_model = run_iou_evaluation(model, val_loader, model_type, cfg, device, vis_d)
        all_records.append(df_model)
        del model
        torch.cuda.empty_cache()

    df_all = pd.concat(all_records, ignore_index=True)
    csv_path = output_dir / "exp2_iou_gate.csv"
    df_all.to_csv(csv_path, index=False)
    log.info("Saved per-frame IoU CSV to %s", csv_path)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    summary_rows = []
    for model_type in ["kan", "xattn"]:
        subset = df_all[df_all["model"] == model_type]["iou_gate"]
        summary_rows.append({
            "model": model_type,
            "n": len(subset),
            "mean_iou_gate": subset.mean(),
            "std_iou_gate": subset.std(),
            "median_iou_gate": subset.median(),
        })
    df_summary = pd.DataFrame(summary_rows)

    from scipy.stats import mannwhitneyu
    kan_vals = df_all[df_all["model"] == "kan"]["iou_gate"].values
    xattn_vals = df_all[df_all["model"] == "xattn"]["iou_gate"].values
    stat_u, p_u = mannwhitneyu(kan_vals, xattn_vals, alternative="greater")  # KAN > xattn

    summary_lines = [
        "=" * 60,
        "Experiment 2 — Spatial Gate Fidelity Summary",
        "=" * 60,
        df_summary.to_string(index=False),
        "",
        f"Mann-Whitney U (KAN > xattn):  U={stat_u:.1f}, p={p_u:.6f}",
        "",
        "Interpretation:",
        "  KAN IoU_gate > xattn IoU_gate  →  KAN gate aligns better",
        "  with object boundaries (significant p-value < 0.05).",
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (output_dir / "exp2_summary.txt").write_text(summary_text)

    # ------------------------------------------------------------------
    # Box-plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    data_to_plot = [kan_vals, xattn_vals]
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=["KAN (ours)", "Cross-Attention"],
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    colours = ["#2196F3", "#F44336"]
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax.set_ylabel("IoU_gate", fontsize=12)
    ax.set_title("Spatial Gate Fidelity: KAN Decoder vs Cross-Attention", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_iou_gate.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    log.info("Experiment 2 complete.  Results written to: %s", output_dir)


if __name__ == "__main__":
    main()
