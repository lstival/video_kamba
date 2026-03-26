"""Analyze training convergence: gradient norms + val metrics from checkpoints.

Reads the latest layer_grad_norms_epoch_*.npz artifacts and the Phase 3 joint
checkpoints, then prints a concise convergence report to stdout (captured by
the SLURM .out file).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch


CKPT_DIR = "/home/WUR/stiva001/WUR/video_kamba/checkpoints/ssm_mem_v2_aot"
GRAD_DIR = "/home/WUR/stiva001/WUR/video_kamba/artifacts/gradient_norms"


def _best_score(ckpt: dict) -> float | None:
    for cb_key, cb_val in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in str(cb_key):
            return cb_val.get("best_model_score")
    return None


def _last_lr(ckpt: dict) -> list[float] | None:
    schedulers = ckpt.get("lr_schedulers", [])
    if schedulers:
        return schedulers[0].get("_last_lr")
    return None


def print_checkpoint_summary() -> None:
    paths = sorted(
        glob.glob(f"{CKPT_DIR}/best_slim_v2_phase3_joint*.ckpt")
        + glob.glob(f"{CKPT_DIR}/last-v1[0-9].ckpt")
    )
    print("=" * 70)
    print("CHECKPOINT SUMMARY — Phase 3 Joint")
    print("=" * 70)
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("global_step", "?")
        score = _best_score(ckpt)
        lr = _last_lr(ckpt)
        score_str = f"{float(score):.4f}" if score is not None else "N/A"
        lr_str = ", ".join(f"{v:.2e}" for v in lr) if lr else "N/A"
        fname = os.path.basename(path)
        print(f"  {fname}")
        print(f"    epoch={epoch}  step={step}  val_J_and_F={score_str}  lr={lr_str}")
    print()


def print_gradient_summary() -> None:
    all_files = sorted(
        f for f in os.listdir(GRAD_DIR)
        if f.startswith("layer_grad_norms_epoch_000")
        and int(f.split("_epoch_")[1].split(".")[0]) <= 5
    )
    print("=" * 70)
    print("GRADIENT NORM TRENDS — epochs 0–5 (job 66043497)")
    print("=" * 70)
    global_means: list[float] = []
    for fname in all_files:
        data = np.load(os.path.join(GRAD_DIR, fname), allow_pickle=True)
        ep = fname.split("_epoch_")[1].split(".")[0]
        mean_norms: np.ndarray = data["mean_norms"]
        vanishing = int(data["vanishing_counts"].sum())
        exploding = int(data["exploding_counts"].sum())
        layer_names: np.ndarray = data["layer_names"]
        p95_norms: np.ndarray = data["p95_norms"]
        global_mean = float(mean_norms.mean())
        global_means.append(global_mean)

        top3_idx = np.argsort(mean_norms)[-3:][::-1]
        bottom3_idx = np.argsort(mean_norms)[:3]

        print(f"\n  Epoch {ep}:")
        print(
            f"    global mean={global_mean:.4e}  max={mean_norms.max():.4e}"
            f"  min={mean_norms.min():.4e}  vanishing={vanishing}  exploding={exploding}"
        )
        print("    Top-3 norms:")
        for i in top3_idx:
            short = str(layer_names[i])[-55:]
            print(f"      ...{short}: {mean_norms[i]:.4e}  p95={p95_norms[i]:.4e}")
        print("    Bottom-3 (vanishing risk):")
        for i in bottom3_idx:
            short = str(layer_names[i])[-55:]
            v = int(data["vanishing_counts"][i])
            print(f"      ...{short}: {mean_norms[i]:.4e}  vanishing_steps={v}")

    if len(global_means) >= 2:
        trend = "DECREASING" if global_means[-1] < global_means[0] else "INCREASING"
        ratio = global_means[-1] / global_means[0]
        print(f"\n  Trend: {trend}  (epoch0={global_means[0]:.4e} → epoch{len(global_means)-1}={global_means[-1]:.4e}, ratio={ratio:.3f})")
    print()


if __name__ == "__main__":
    print_checkpoint_summary()
    print_gradient_summary()
