"""Analyze Phase 2 BL30K full training convergence (job 66040635).

Reads the best_slim_v2_phase2_bl30k_full checkpoint and reports
val metrics, epoch, LR, and the spike diagnostics log.
"""

from __future__ import annotations

import json
import os

import torch


CKPT_DIR = "/home/WUR/stiva001/WUR/video_kamba/checkpoints/ssm_mem_v2_aot"
SPIKE_LOG = "/home/WUR/stiva001/WUR/video_kamba/logs/spike_diagnostics.jsonl"


def print_checkpoint_summary() -> None:
    path = os.path.join(CKPT_DIR, "best_slim_v2_phase2_bl30k_full.ckpt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    epoch = ckpt.get("epoch", "?")
    step  = ckpt.get("global_step", "?")

    score = None
    best_path = ""
    for cb_key, cb_val in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in str(cb_key):
            score    = cb_val.get("best_model_score")
            best_path = cb_val.get("best_model_path", "")
            all_best  = cb_val.get("best_k_models", {})

    lr = None
    if "lr_schedulers" in ckpt and ckpt["lr_schedulers"]:
        sched = ckpt["lr_schedulers"][0]
        lr = sched.get("_last_lr")
        last_epoch_sched = sched.get("last_epoch")

    score_str = f"{float(score):.4f}" if score is not None else "N/A"
    lr_str    = ", ".join(f"{v:.2e}" for v in lr) if lr else "N/A"

    print("=" * 70)
    print("BL30K FULL PHASE 2 — best checkpoint")
    print("=" * 70)
    print(f"  epoch          : {epoch}")
    print(f"  global_step    : {step}")
    print(f"  val_J_and_F    : {score_str}")
    print(f"  current LR     : {lr_str}")
    print(f"  best_model_path: {os.path.basename(best_path)}")
    if all_best:
        print("  all best_k models:")
        for p, s in all_best.items():
            print(f"    {os.path.basename(p)}: {float(s):.4f}")

    # also check the last checkpoint to see latest epoch
    import glob
    last_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "last-v*.ckpt")))
    # The BL30K full job's last ckpt — saved at 13:32 (last-v9) and 14:05 (last-v10)?
    # We print all recent ones to compare
    print("\n  Recent last-v*.ckpt epochs:")
    for lp in last_ckpts[-4:]:
        lckpt = torch.load(lp, map_location="cpu", weights_only=False)
        le = lckpt.get("epoch", "?")
        ls = lckpt.get("global_step", "?")
        llr = None
        if "lr_schedulers" in lckpt and lckpt["lr_schedulers"]:
            llr = lckpt["lr_schedulers"][0].get("_last_lr")
        llr_str = ", ".join(f"{v:.2e}" for v in llr) if llr else "N/A"
        # identify by checkpoint callback monitor
        monitor = "?"
        for cb_key in lckpt.get("callbacks", {}):
            if "ModelCheckpoint" in str(cb_key):
                monitor = str(cb_key)
        print(f"    {os.path.basename(lp)}: epoch={le}, step={ls}, lr={llr_str}")
    print()


def print_spike_diagnostics() -> None:
    if not os.path.exists(SPIKE_LOG):
        print("No spike diagnostics log found.")
        return

    print("=" * 70)
    print("SPIKE DIAGNOSTICS (last 20 entries)")
    print("=" * 70)
    with open(SPIKE_LOG) as f:
        lines = f.readlines()
    for line in lines[-20:]:
        entry = json.loads(line.strip())
        print(f"  {entry}")
    print()


if __name__ == "__main__":
    print_checkpoint_summary()
    print_spike_diagnostics()
