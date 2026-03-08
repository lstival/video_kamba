"""Experiment 3 — 1D RBF Activation Profile  (Qualitative Interpretability Proof).

Visualises the learned univariate basis functions of ``FastKANLayer`` inside
the temporal gate modulator (``B_modulator.kan``).

Mathematical formulation
------------------------
Each edge (in_feature i, out_feature o) in ``FastKANLayer`` implements::

    f_{o,i}(x) = sum_{g=1}^{G} w_{o,i,g} * exp(-((x - mu_g) / h)^2)
               + b_i * SiLU(x)

where:
    * w_{o,i,g}  = ``rbf_weight[o, i, g]``  (learnable, shape [out, in, G])
    * mu_g       = ``grid[g]``               (fixed centroids in [-2, 2])
    * h          = step = (2 - (-2)) / G     (fixed bandwidth = 0.5 for G=8)
    * b_i        = ``base_weight[o, i]``     (residual linear term)

Channels are auto-discovered by ranking by mean absolute activation across
the DAVIS validation set — the most decisive channels are profiled first.

Usage
-----
From the project root::

    python scripts/exp3_rbf_profile.py \\
        kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \\
        data_dir=data/DAVIS/DAVIS \\
        output_dir=results/exp3

Authors: KANGA Project
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Ensure project root is on sys.path when executed as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.davis import DAVISDataModule
from models.components.fast_kan_layer import FastKANLayer
from models.video_mamba import VideoMambaSystem

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hydra structured config
# ---------------------------------------------------------------------------

@dataclass
class Exp3Config:
    """Configuration for Experiment 3 — RBF Activation Profile."""

    kan_ckpt: str = "???"
    data_dir: str = "data/DAVIS/DAVIS"
    output_dir: str = "results/exp3"
    n_top_channels: int = 5
    x_range: Tuple[float, float] = (-3.0, 3.0)
    x_points: int = 500
    ssm_layer_idx: int = 0
    batch_size: int = 1
    num_workers: int = 4
    img_size: int = 224
    seq_len: int = 8
    max_batches: int = 30
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="exp3_config", node=Exp3Config)


# ---------------------------------------------------------------------------
# Channel discovery
# ---------------------------------------------------------------------------

def find_top_channels(
    model: VideoMambaSystem,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    max_batches: int,
    device: torch.device,
    n_top: int,
) -> Tuple[List[int], np.ndarray]:
    """Rank input channels of ``B_modulator.kan`` by mean |activation output|.

    We hook the *input* (pre-modulator) tensor of ``B_modulator``
    (shape ``[BT, D]``) to measure which input features drive the largest
    absolute responses in the gate.

    Args:
        model: Loaded VideoMambaSystem with KAN modulators.
        dataloader: DAVIS validation dataloader.
        layer_idx: SSM layer index.
        max_batches: Maximum batches to process.
        device: Computation device.
        n_top: Number of top channels to return.

    Returns:
        Tuple of (channel_indices sorted by importance, per-channel mean |act|).
    """
    target = model.temporal_model.layers[layer_idx].B_modulator
    collected_inputs: List[torch.Tensor] = []

    def hook_fn(_m: torch.nn.Module, inp: tuple, _out: torch.Tensor) -> None:
        # inp[0]: [BT, D] — raw input to B_modulator
        collected_inputs.append(inp[0].detach().cpu())

    handle = target.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                ref_img, ref_mask, query_images, query_masks = (
                    t.to(device) for t in batch[:4]
                )
                model(query_images, ref_frame=ref_img, ref_mask=ref_mask)
    finally:
        handle.remove()

    if not collected_inputs:
        raise RuntimeError("No inputs collected — check model and dataloader.")

    all_inputs = torch.cat(collected_inputs, dim=0)  # [N, D]
    channel_importance = all_inputs.abs().mean(dim=0).numpy()  # [D]
    top_indices = list(np.argsort(channel_importance)[::-1][:n_top])
    log.info(
        "Top %d channels (by mean |input|): %s", n_top, top_indices
    )
    return top_indices, channel_importance


# ---------------------------------------------------------------------------
# RBF function reconstruction
# ---------------------------------------------------------------------------

def get_fastkan_layer(model: VideoMambaSystem, layer_idx: int) -> FastKANLayer:
    """Retrieve the ``FastKANLayer`` from ``B_modulator.kan`` at *layer_idx*.

    Args:
        model: Loaded VideoMambaSystem.
        layer_idx: Index into ``temporal_model.layers``.

    Returns:
        The ``FastKANLayer`` module.

    Raises:
        AttributeError: If the modulator is not a ``FastKANModulator``
            (e.g., the model was trained with ``modulator_type='mlp'``).
    """
    b_mod = model.temporal_model.layers[layer_idx].B_modulator
    if not hasattr(b_mod, "kan"):
        raise AttributeError(
            f"B_modulator at layer {layer_idx} has no 'kan' attribute. "
            "This experiment requires a KAN-modulated model "
            "(modulator_type='kan')."
        )
    return b_mod.kan


def reconstruct_1d_function(
    kan_layer: FastKANLayer,
    channel_in: int,
    channel_out: int,
    x_range: Tuple[float, float],
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct the learned 1D edge function f_{out,in}(x).

    Evaluates::

        f_{o,i}(x) = sum_g w_{o,i,g} * exp(-((x - mu_g)/h)^2)
                   + base_weight[o,i] * SiLU(x)

    over a dense grid, returning (x_grid, y_values).

    Args:
        kan_layer: The ``FastKANLayer`` whose weights to inspect.
        channel_in: Input channel index ``i``.
        channel_out: Output channel index ``o``.
        x_range: (x_min, x_max) for the evaluation grid.
        n_points: Number of grid points.

    Returns:
        ``(x_grid, y_values)`` as 1-D numpy arrays of length *n_points*.
    """
    x = np.linspace(x_range[0], x_range[1], n_points).astype(np.float32)
    x_t = torch.from_numpy(x)                               # [N]

    grid = kan_layer.grid.detach().cpu()                    # [G]
    step = kan_layer.step
    rbf_w = kan_layer.rbf_weight.detach().cpu()             # [out, in, G]
    base_w = kan_layer.base_weight.detach().cpu()           # [out, in]

    # RBF basis for this input channel
    # x_t: [N], grid: [G]  → basis: [N, G]
    x_expanded = x_t.unsqueeze(-1)                          # [N, 1]
    basis = torch.exp(-((x_expanded - grid) / step) ** 2)  # [N, G]

    rbf_contrib = (basis * rbf_w[channel_out, channel_in]).sum(dim=-1)  # [N]

    silu_x = torch.nn.functional.silu(x_t)
    linear_contrib = base_w[channel_out, channel_in] * silu_x           # [N]

    y = (rbf_contrib + linear_contrib).numpy()
    return x, y


def leaky_relu_reference(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Leaky ReLU activation as a reference for MLP-style gating.

    Args:
        x: Input array.
        alpha: Slope for negative values.

    Returns:
        LeakyRelu(x) = max(alpha * x, x).
    """
    return np.where(x > 0, x, alpha * x)


def gelu_reference(x: np.ndarray) -> np.ndarray:
    """GELU activation as a reference for MLP-style gating."""
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rbf_profiles(
    profiles: List[Tuple[int, int, np.ndarray, np.ndarray]],
    x_range: Tuple[float, float],
    output_path: Path,
) -> None:
    """Multi-panel figure of learned KAN edge functions vs ReLU/GELU.

    Uses dual y-axes (twinx) to ensure the KAN RBF shape is visible even if
    its magnitude is much smaller than the MLP references.
    """
    n = len(profiles)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, (ch_in, ch_out, x_grid, y_kan) in enumerate(profiles):
        ax = axes[idx // cols][idx % cols]

        # 1. Plot KAN Profile (Left Axis)
        l1, = ax.plot(x_grid, y_kan, color="#1565C0", linewidth=2.5,
                      label="KAN edge (learned RBF)")
        ax.set_ylabel("KAN Gate Contribution", color="#1565C0", fontsize=9)
        ax.tick_params(axis='y', labelcolor="#1565C0")

        # 2. Plot MLP References (Right Axis)
        ax_ref = ax.twinx()
        y_lrelu = leaky_relu_reference(x_grid)
        y_gelu = gelu_reference(x_grid)

        l2, = ax_ref.plot(x_grid, y_lrelu, color="#B71C1C", linewidth=1.5,
                          linestyle="--", alpha=0.6, label="Leaky ReLU ref")
        l3, = ax_ref.plot(x_grid, y_gelu, color="#2E7D32", linewidth=1.5,
                          linestyle=":", alpha=0.6, label="GELU ref")
        
        ax_ref.set_ylabel("MLP Ref Activation", color="gray", fontsize=8)
        ax_ref.tick_params(axis='y', labelcolor="gray")

        # --- Align Zeros ---
        # Get limits
        y1_min, y1_max = ax.get_ylim()
        y2_min, y2_max = ax_ref.get_ylim()
        
        # Calculate scaling to align zeros
        # We want y1=0 and y2=0 to be at the same vertical position.
        # This keeps the horizontal grid line consistent.
        if y1_min < 0 < y1_max and y2_min < 0 < y2_max:
             # Align them by adjusting limits to maintain common ratio
             ratio1 = y1_max / (y1_max - y1_min)
             ratio2 = y2_max / (y2_max - y2_min)
             # Adjust whichever one is "narrower" around 0
             if ratio1 > ratio2: # max1 is relatively larger
                  new_y2_max = y2_min * ratio1 / (ratio1 - 1)
                  ax_ref.set_ylim(y2_min, new_y2_max)
             else:
                  new_y1_max = y1_min * ratio2 / (ratio2 - 1)
                  ax.set_ylim(y1_min, new_y1_max)

        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)

        ax.set_title(f"Channel in={ch_in}, out={ch_out}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Normalised DINOv2 feature value", fontsize=9)
        
        # Combine legends from both axes
        lns = [l1, l2, l3]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=8, loc="upper left", frameon=True, framealpha=0.8)
        
        ax.set_xlim(x_range)
        ax.grid(True, linestyle=':', alpha=0.4)

    # Hide unused axes
    for extra_idx in range(n, rows * cols):
        axes[extra_idx // cols][extra_idx % cols].set_visible(False)

    fig.suptitle(
        "Learned FastKAN RBF Edge Functions vs MLP Baselines (Leaky ReLU/GELU)\n"
        r"$f_{o,i}(x)=\sum_g w_{o,i,g}\,e^{-((x-\mu_g)/h)^2}+b_i\,\mathrm{SiLU}(x)$",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved RBF profile figure to %s", output_path)


def plot_weight_heatmap(
    kan_layer: FastKANLayer,
    channel_in: int,
    output_path: Path,
) -> None:
    """Heatmap of ``rbf_weight[:, channel_in, :]`` for a given input channel.

    Rows = output features, columns = grid bins.

    Args:
        kan_layer: The ``FastKANLayer`` to inspect.
        channel_in: Input channel to visualise.
        output_path: Destination PDF path.
    """
    w = kan_layer.rbf_weight[:, channel_in, :].detach().cpu().numpy()  # [out, G]

    fig, ax = plt.subplots(figsize=(8, max(3, w.shape[0] // 40)))
    im = ax.imshow(w, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(w).max(), vmax=np.abs(w).max())
    ax.set_xlabel("Grid bin  g", fontsize=10)
    ax.set_ylabel("Output channel  o", fontsize=10)
    ax.set_title(
        f"RBF weight heatmap  rbf_weight[:, channel_in={channel_in}, :]",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="Weight value")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved weight heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/analysis", config_name="exp3")
def main(cfg: DictConfig) -> None:  # noqa: D103
    OmegaConf.to_yaml(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = VideoMambaSystem.load_from_checkpoint(cfg.kan_ckpt, map_location=device)
    model.to(device).eval()
    log.info("Loaded KAN checkpoint: %s", cfg.kan_ckpt)

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
    # Channel discovery
    # ------------------------------------------------------------------
    top_channels, importance = find_top_channels(
        model=model,
        dataloader=val_loader,
        layer_idx=cfg.ssm_layer_idx,
        max_batches=cfg.max_batches,
        device=device,
        n_top=cfg.n_top_channels,
    )

    channel_df = pd.DataFrame({
        "channel_idx": np.arange(len(importance)),
        "mean_abs_activation": importance,
    }).sort_values("mean_abs_activation", ascending=False)
    channel_df.to_csv(output_dir / "exp3_top_channels.csv", index=False)
    log.info("Top channels saved to exp3_top_channels.csv")

    # ------------------------------------------------------------------
    # RBF reconstruction
    # ------------------------------------------------------------------
    kan_layer = get_fastkan_layer(model, cfg.ssm_layer_idx)
    x_range: Tuple[float, float] = tuple(cfg.x_range)  # type: ignore[assignment]

    profiles: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for ch_in in top_channels:
        # For clarity, profile the mapping from input channel ch_in to the same
        # output channel (diagonal edge in the full weight tensor).
        ch_out = ch_in
        x_grid, y_vals = reconstruct_1d_function(
            kan_layer, ch_in, ch_out, x_range, cfg.x_points
        )
        profiles.append((ch_in, ch_out, x_grid, y_vals))
        log.info(
            "  Channel %d: y_range=[%.4f, %.4f]", ch_in, y_vals.min(), y_vals.max()
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_rbf_profiles(profiles, x_range, output_dir / "exp3_rbf_profiles.pdf")

    # Weight heatmap for the most important channel
    if top_channels:
        plot_weight_heatmap(
            kan_layer,
            channel_in=top_channels[0],
            output_path=output_dir / "exp3_weight_heatmap.pdf",
        )

    log.info("Experiment 3 complete.  Results written to: %s", output_dir)


if __name__ == "__main__":
    main()
