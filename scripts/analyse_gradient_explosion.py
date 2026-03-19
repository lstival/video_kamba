"""Analyse gradient explosion patterns from saved .npz artifacts.

Focuses on the first N epochs (from a specific SLURM run) and identifies:
  - Dominant layer types causing explosions
  - out_proj progression across epochs
  - Module-level breakdown at each epoch
  - Global norm trend

Usage:
    python scripts/analyse_gradient_explosion.py +num_epochs=4
    python scripts/analyse_gradient_explosion.py +num_epochs=4 +artifacts_dir=artifacts/gradient_norms
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

ARTIFACTS_GRAD_NORMS = "artifacts/gradient_norms"
EXPLODING_THRESHOLD = 1_000.0


def load_epoch(artifacts_dir: Path, epoch: int) -> dict | None:
    """Load gradient norm npz for a given epoch."""
    path = artifacts_dir / f"layer_grad_norms_epoch_{epoch:04d}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "layer_names": data["layer_names"].tolist(),
        "mean_norms": data["mean_norms"].astype(np.float64),
        "std_norms": data["std_norms"].astype(np.float64),
        "p95_norms": data["p95_norms"].astype(np.float64),
        "exploding_counts": data["exploding_counts"].astype(np.int64),
        "vanishing_counts": data["vanishing_counts"].astype(np.int64),
    }


def group_by_suffix(
    names: list[str], means: np.ndarray, expl: np.ndarray
) -> dict[str, dict]:
    """Group layer stats by parameter type suffix (weight, bias, rbf_weight, etc.)."""
    groups: dict[str, dict] = defaultdict(
        lambda: {"layers": [], "total_mean": 0.0, "total_expl": 0}
    )
    for name, mean, e in zip(names, means, expl):
        suffix = name.split(".")[-1]
        groups[suffix]["layers"].append(name)
        groups[suffix]["total_mean"] += float(mean)
        groups[suffix]["total_expl"] += int(e)
    return dict(groups)


def group_by_top_module(
    names: list[str], means: np.ndarray, expl: np.ndarray
) -> dict[str, dict]:
    """Group layer stats by top-level module name."""
    groups: dict[str, dict] = defaultdict(
        lambda: {"layers": [], "total_mean": 0.0, "total_expl": 0}
    )
    for name, mean, e in zip(names, means, expl):
        module = name.split(".")[0]
        groups[module]["layers"].append(name)
        groups[module]["total_mean"] += float(mean)
        groups[module]["total_expl"] += int(e)
    return dict(groups)


SEP = "─" * 80
THICK = "═" * 80


@hydra.main(version_base="1.3", config_path=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    artifacts_dir = Path(cfg.get("artifacts_dir", ARTIFACTS_GRAD_NORMS))
    num_epochs: int = int(cfg.get("num_epochs", 4))

    print(THICK)
    print("  GRADIENT EXPLOSION DEEP ANALYSIS")
    print(THICK)

    epochs_data = []
    for ep in range(num_epochs):
        d = load_epoch(artifacts_dir, ep)
        if d is None:
            log.warning("No artifact for epoch %d", ep)
            continue
        epochs_data.append((ep, d))

    if not epochs_data:
        log.error("No artifacts found in %s", artifacts_dir)
        raise SystemExit(1)

    # ── Per-epoch overview ────────────────────────────────────────────────
    print(f"\n## Per-Epoch Overview ({num_epochs} epochs)\n")
    print(f"  {'Epoch':>5}  {'Global Mean':>12}  {'Exploding':>10}  {'Vanishing':>10}  {'Max norm':>12}  {'Max layer'}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*40}")
    for ep, d in epochs_data:
        means = d["mean_norms"]
        expl = d["exploding_counts"]
        vani = d["vanishing_counts"]
        max_idx = int(np.argmax(means))
        print(
            f"  {ep:>5}  {means.mean():>12.3e}  {expl.sum():>10,}  {vani.sum():>10,}  "
            f"{means[max_idx]:>12.3e}  {d['layer_names'][max_idx]}"
        )

    # ── Layer-type breakdown (last epoch) ────────────────────────────────
    last_ep, last_d = epochs_data[-1]
    print(f"\n## Layer-Type Breakdown (epoch {last_ep})\n")
    suffix_groups = group_by_suffix(
        last_d["layer_names"], last_d["mean_norms"], last_d["exploding_counts"]
    )
    sorted_suffixes = sorted(
        suffix_groups.items(), key=lambda kv: kv[1]["total_mean"], reverse=True
    )
    print(f"  {'Param type':<30}  {'# layers':>8}  {'Sum mean norm':>14}  {'Total expl':>10}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*14}  {'─'*10}")
    for suffix, stats in sorted_suffixes[:15]:
        print(
            f"  {suffix:<30}  {len(stats['layers']):>8}  "
            f"{stats['total_mean']:>14.3e}  {stats['total_expl']:>10,}"
        )

    # ── Top-module breakdown across all epochs ────────────────────────────
    print(f"\n## Module Breakdown Across Epochs\n")
    all_modules: set[str] = set()
    epoch_module_data: dict[int, dict[str, float]] = {}
    for ep, d in epochs_data:
        groups = group_by_top_module(
            d["layer_names"], d["mean_norms"], d["exploding_counts"]
        )
        all_modules.update(groups.keys())
        epoch_module_data[ep] = {
            mod: stats["total_mean"] for mod, stats in groups.items()
        }
    modules_sorted = sorted(
        all_modules,
        key=lambda m: epoch_module_data[epochs_data[-1][0]].get(m, 0.0),
        reverse=True,
    )
    header = "  ".join(f"ep{ep:02d}" for ep, _ in epochs_data)
    print(f"  {'Module':<40}  {header}")
    print(f"  {'─'*40}  {'─'*len(header)}")
    for mod in modules_sorted:
        vals = "  ".join(
            f"{epoch_module_data[ep].get(mod, 0.0):>6.1e}" for ep, _ in epochs_data
        )
        print(f"  {mod:<40}  {vals}")

    # ── out_proj layers specifically ──────────────────────────────────────
    print(f"\n## out_proj Layer Progression\n")
    print(f"  {'Layer':<65}", end="")
    for ep, _ in epochs_data:
        print(f"  ep{ep:02d}", end="")
    print()
    print(f"  {'─'*65}  {'─'*(6*len(epochs_data))}")

    # Collect all out_proj layer names
    out_proj_names = [
        n for n in epochs_data[0][1]["layer_names"] if "out_proj" in n
    ]
    for layer in out_proj_names:
        print(f"  {layer:<65}", end="")
        for ep, d in epochs_data:
            if layer in d["layer_names"]:
                idx = d["layer_names"].index(layer)
                norm = d["mean_norms"][idx]
                print(f"  {norm:>6.1e}", end="")
            else:
                print(f"  {'n/a':>6}", end="")
        print()

    # ── KAN modulator layers ──────────────────────────────────────────────
    print(f"\n## KAN Modulator Layer Progression\n")
    kan_names = [
        n for n in epochs_data[0][1]["layer_names"]
        if "kan" in n.lower() and "weight" in n
    ]
    print(f"  {'Layer':<70}", end="")
    for ep, _ in epochs_data:
        print(f"  ep{ep:02d}", end="")
    print()
    print(f"  {'─'*70}  {'─'*(6*len(epochs_data))}")
    for layer in kan_names[:20]:
        print(f"  {layer:<70}", end="")
        for ep, d in epochs_data:
            if layer in d["layer_names"]:
                idx = d["layer_names"].index(layer)
                norm = d["mean_norms"][idx]
                flag = "⚠" if norm > EXPLODING_THRESHOLD else " "
                print(f"  {norm:>5.1e}{flag}", end="")
            else:
                print(f"  {'n/a':>6}", end="")
        print()

    # ── Top 20 worst layers (last epoch) ─────────────────────────────────
    print(f"\n## Top 20 Worst Layers — Epoch {last_ep}\n")
    sorted_idx = np.argsort(last_d["mean_norms"])[::-1][:20]
    print(f"  {'Layer':<70}  {'Mean':>10}  {'P95':>10}  {'Expl count':>10}")
    print(f"  {'─'*70}  {'─'*10}  {'─'*10}  {'─'*10}")
    for i in sorted_idx:
        name = last_d["layer_names"][i]
        mean = last_d["mean_norms"][i]
        p95 = last_d["p95_norms"][i]
        ec = last_d["exploding_counts"][i]
        flag = " ⚠" if mean > EXPLODING_THRESHOLD else ""
        print(f"  {name:<70}  {mean:>10.3e}  {p95:>10.3e}  {ec:>10,}{flag}")

    print(f"\n{THICK}\n")


if __name__ == "__main__":
    main()
