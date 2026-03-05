"""Experiment 1 — Activation Entropy (Disentanglement Proof).

Compares the Shannon entropy of the temporal gate vector alpha_t between a
KAN-modulated model and an MLP-modulated baseline.

Mathematical formulation
------------------------
For a gate vector alpha_t in R^D, compute the normalised magnitude distribution::

    alpha_hat_{t,i} = |alpha_{t,i}| / sum_j |alpha_{t,j}|

Shannon entropy of the representation::

    H(alpha_t) = - sum_{i=1}^{D} alpha_hat_{t,i} * log(alpha_hat_{t,i})

Low entropy  →  sparse, disentangled channels (KAN expected behaviour)
High entropy →  dense, entangled activations (MLP expected behaviour)

The gate vector is captured from ``IntricateKANSSMCore.B_modulator`` via
a forward hook; shape is ``[B*T, inner_dim]`` per batch.

Usage
-----
From the project root::

    python scripts/exp1_activation_entropy.py \\
        kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \\
        mlp_ckpt=lightning_logs/version_mlp/checkpoints/best.ckpt \\
        data_dir=data/DAVIS/DAVIS \\
        output_dir=results/exp1

Authors: KANGA Project
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
from models.video_mamba import VideoMambaSystem

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hydra structured config
# ---------------------------------------------------------------------------

@dataclass
class Exp1Config:
    """Configuration for Experiment 1 — Activation Entropy."""

    kan_ckpt: str = "???"
    mlp_ckpt: str = "???"
    data_dir: str = "data/DAVIS/DAVIS"
    output_dir: str = "results/exp1"
    max_batches: int = 50
    batch_size: int = 1
    num_workers: int = 4
    img_size: int = 224
    seq_len: int = 8
    b_mod_layer_idx: int = 0
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="exp1_config", node=Exp1Config)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> VideoMambaSystem:
    """Load a ``VideoMambaSystem`` from checkpoint.

    Args:
        ckpt_path: Absolute or relative path to the ``.ckpt`` file.
        device: Target device.

    Returns:
        Model in eval mode on *device*.
    """
    model = VideoMambaSystem.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    log.info("Loaded checkpoint: %s", ckpt_path)
    return model


def extract_b_mod_activations(
    model: VideoMambaSystem,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    max_batches: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect B_modulator output activations using a forward hook.

    The hook captures the *output* of ``B_modulator`` inside the chosen
    ``IntricateKANSSMCore`` layer.  Shape per invocation is ``[B*T, D]``
    (after the softplus activation inside the modulator).

    Args:
        model: Loaded VideoMambaSystem.
        dataloader: DAVIS validation dataloader.
        layer_idx: Index into ``model.temporal_model.layers``.
        max_batches: Stop after this many batches (``-1`` = full set).
        device: Computation device.

    Returns:
        Tensor of shape ``[N_samples, D]`` collecting all gate vectors.
    """
    target_module = model.temporal_model.layers[layer_idx].B_modulator
    collected: List[torch.Tensor] = []

    def hook_fn(_module: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        # out: [B*T, D]  — positive-valued gate vector post-activation
        collected.append(out.detach().cpu())

    handle = target_module.register_forward_hook(hook_fn)
    log.info(
        "Registered forward hook on: temporal_model.layers[%d].B_modulator", layer_idx
    )

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                ref_img, ref_mask, query_images, query_masks = (
                    t.to(device) for t in batch[:4]
                )
                model(query_images, ref_frame=ref_img, ref_mask=ref_mask)

                if (batch_idx + 1) % 10 == 0:
                    log.info("  processed %d / %d batches", batch_idx + 1, max_batches)
    finally:
        handle.remove()

    if not collected:
        raise RuntimeError("No activations collected — check dataloader and hook target.")

    activations = torch.cat(collected, dim=0)  # [N, D]
    log.info("Collected activations: %s", tuple(activations.shape))
    return activations


def compute_shannon_entropy(activations: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Shannon entropy of the normalised gate distribution.

    For each sample vector ``a`` in ``activations``::

        a_hat_i = |a_i| / sum_j |a_j|
        H(a) = - sum_i a_hat_i * log(a_hat_i)       (nats, clip at 0)

    Args:
        activations: ``[N, D]`` gate vectors.

    Returns:
        ``[N]`` entropy values in nats.
    """
    abs_act = activations.abs()  # [N, D]
    norm_sum = abs_act.sum(dim=1, keepdim=True).clamp(min=1e-12)
    a_hat = abs_act / norm_sum  # [N, D]

    # Stable entropy: −(a_hat * log(a_hat + eps)).  Pixels with a_hat≈0
    # contribute ≈0 because x * log(x) → 0 as x → 0.
    entropy = -(a_hat * (a_hat + 1e-12).log()).sum(dim=1)  # [N]

    assert entropy.shape[0] == activations.shape[0], "Entropy shape mismatch"
    assert (entropy >= 0).all(), "Entropy must be non-negative"
    return entropy


def plot_entropy_histogram(
    h_kan: np.ndarray,
    h_mlp: np.ndarray,
    output_path: Path,
) -> None:
    """Dual histogram with KDE overlay comparing KAN vs MLP entropy.

    Args:
        h_kan: Entropy values for the KAN model.
        h_mlp: Entropy values for the MLP model.
        output_path: File path for the saved figure (PDF).
    """
    from scipy.stats import gaussian_kde  # local import — optional dependency

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(h_kan.min(), h_mlp.min()), max(h_kan.max(), h_mlp.max()), 60
    )

    ax.hist(h_kan, bins=bins, alpha=0.4, color="#2196F3", label="KAN (ours)", density=True)
    ax.hist(h_mlp, bins=bins, alpha=0.4, color="#F44336", label="MLP baseline", density=True)

    for values, colour in [(h_kan, "#1565C0"), (h_mlp, "#B71C1C")]:
        kde = gaussian_kde(values, bw_method="scott")
        x_grid = np.linspace(bins[0], bins[-1], 300)
        ax.plot(x_grid, kde(x_grid), color=colour, linewidth=2)

    ax.axvline(np.mean(h_kan), color="#1565C0", linestyle="--", linewidth=1.2,
               label=f"KAN mean = {np.mean(h_kan):.3f}")
    ax.axvline(np.mean(h_mlp), color="#B71C1C", linestyle="--", linewidth=1.2,
               label=f"MLP mean = {np.mean(h_mlp):.3f}")

    ax.set_xlabel("Shannon Entropy H(α_t)  [nats]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Activation Entropy Distribution: KAN vs MLP Temporal Gate", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved entropy histogram to %s", output_path)


def summarise_entropy(
    h_kan: np.ndarray,
    h_mlp: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """Write summary statistics and CSV files.

    Args:
        h_kan: KAN entropy values.
        h_mlp: MLP entropy values.
        output_dir: Directory for output files.

    Returns:
        DataFrame with summary statistics.
    """
    # Per-sample CSV
    df = pd.DataFrame({
        "entropy": np.concatenate([h_kan, h_mlp]),
        "model": ["KAN"] * len(h_kan) + ["MLP"] * len(h_mlp),
    })
    csv_path = output_dir / "exp1_entropy.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved per-sample entropy CSV to %s", csv_path)

    # Summary
    from scipy.stats import mannwhitneyu, ttest_ind

    stat_t, p_t = ttest_ind(h_kan, h_mlp, equal_var=False)
    stat_u, p_u = mannwhitneyu(h_kan, h_mlp, alternative="less")  # KAN < MLP expected

    summary_lines = [
        "=" * 60,
        "Experiment 1 — Activation Entropy Summary",
        "=" * 60,
        f"KAN    n={len(h_kan):6d}  mean={h_kan.mean():.4f}  std={h_kan.std():.4f}"
        f"  median={np.median(h_kan):.4f}",
        f"MLP    n={len(h_mlp):6d}  mean={h_mlp.mean():.4f}  std={h_mlp.std():.4f}"
        f"  median={np.median(h_mlp):.4f}",
        "",
        f"Welch t-test:       t={stat_t:.4f}, p={p_t:.6f}",
        f"Mann-Whitney U:     U={stat_u:.1f}, p={p_u:.6f}  (H1: KAN < MLP)",
        "",
        "Interpretation:",
        "  KAN entropy < MLP entropy  →  sparser, more disentangled KAN gate",
        "  Significant p-value (<0.05) supports Principle of Simplicity claim.",
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    txt_path = output_dir / "exp1_summary.txt"
    txt_path.write_text(summary_text)
    log.info("Saved summary to %s", txt_path)

    summary_df = pd.DataFrame({
        "model": ["KAN", "MLP"],
        "n": [len(h_kan), len(h_mlp)],
        "mean_entropy": [h_kan.mean(), h_mlp.mean()],
        "std_entropy": [h_kan.std(), h_mlp.std()],
        "median_entropy": [np.median(h_kan), np.median(h_mlp)],
        "welch_t": [stat_t, float("nan")],
        "welch_p": [p_t, float("nan")],
        "mannwhitney_u": [stat_u, float("nan")],
        "mannwhitney_p": [p_u, float("nan")],
    })
    return summary_df


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/analysis", config_name="exp1")
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
    # Collect activations
    # ------------------------------------------------------------------
    results: Dict[str, np.ndarray] = {}
    for label, ckpt in [("KAN", cfg.kan_ckpt), ("MLP", cfg.mlp_ckpt)]:
        log.info("--- Processing %s model ---", label)
        model = load_model(ckpt, device)
        acts = extract_b_mod_activations(
            model,
            val_loader,
            layer_idx=cfg.b_mod_layer_idx,
            max_batches=cfg.max_batches,
            device=device,
        )
        entropy = compute_shannon_entropy(acts).numpy()
        results[label] = entropy
        log.info("%s entropy: mean=%.4f  std=%.4f", label, entropy.mean(), entropy.std())
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    h_kan, h_mlp = results["KAN"], results["MLP"]

    plot_entropy_histogram(h_kan, h_mlp, output_dir / "exp1_entropy.pdf")
    summarise_entropy(h_kan, h_mlp, output_dir)

    log.info("Experiment 1 complete.  Results written to: %s", output_dir)


if __name__ == "__main__":
    main()
