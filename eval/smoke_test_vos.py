"""Smoke test for the YouTube-VOS / MOSE multi-object VOS pipeline.

Mirrors ``eval/smoke_test_davis.py``: tries real data on disk first,
falls back to synthetic tensors if the dataset is not found.

Usage (direct – not via sbatch):
    python eval/smoke_test_vos.py
    python eval/smoke_test_vos.py --dataset mose

Run via sbatch for GPU smoke-test::
    sbatch scripts/slurm_youtubevos_seg.sh trainer.fast_dev_run=true
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.video_mamba import VideoMambaSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_synthetic_batch(
    B: int = 2,
    T: int = 5,
    H: int = 224,
    n_id: int = 10,
    num_seg_classes: int = 11,
):
    """Return a synthetic 6-element VOS batch."""
    ref_img = torch.randn(B, 3, H, H)
    ref_mask = torch.randint(0, num_seg_classes, (B, H, H))
    query_images = torch.randn(B, T, 3, H, H)
    query_masks = torch.randint(0, num_seg_classes, (B, T, H, H))
    obj_present = torch.ones(B, T, n_id, dtype=torch.bool)
    meta = [
        {"video_id": f"synthetic_{i:04d}", "seen_obj_ids": list(range(1, 4)), "unseen_obj_ids": []}
        for i in range(B)
    ]
    return ref_img, ref_mask, query_images, query_masks, obj_present, meta


def _try_real_data(data_dir: str, dataset_type: str, batch_size: int = 2):
    """Try to load one real batch; return None if data not present."""
    img_dir = os.path.join(data_dir, "train", "JPEGImages")
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) == 0:
        return None

    from data.multi_object_vos import MultiObjectVOSDataModule

    dm = MultiObjectVOSDataModule(
        data_dir=data_dir,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=0,  # single-process for smoke test
        seq_len=5,
        img_size=224,
        n_id=10,
        augment_train=False,
    )
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VOS pipeline smoke test")
    parser.add_argument(
        "--dataset",
        choices=["youtubevos", "mose"],
        default="youtubevos",
        help="Dataset type to test against real data (if present).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data root directory.  Defaults to data/<YouTubeVOS|MOSE>.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir_map = {
        "youtubevos": os.path.join(project_root, "data", "YouTubeVOS"),
        "mose": os.path.join(project_root, "data", "MOSE"),
    }
    data_dir = args.data_root or dataset_dir_map[args.dataset]

    n_id = 10
    num_seg_classes = n_id + 1  # background + n_id objects

    # ── 1. Load batch ────────────────────────────────────────────────────────
    batch = _try_real_data(data_dir, args.dataset)

    if batch is None:
        print(
            f"[INFO] {args.dataset.upper()} data not found at {data_dir}.\n"
            "       Falling back to synthetic tensors for the model test."
        )
        batch = _build_synthetic_batch(B=2, T=5, H=224, n_id=n_id, num_seg_classes=num_seg_classes)
    else:
        ref_img, ref_mask, query_images, query_masks, obj_present, meta = batch
        print(
            f"[INFO] Loaded real {args.dataset.upper()} batch:\n"
            f"       ref_img       {tuple(ref_img.shape)}\n"
            f"       ref_mask      {tuple(ref_mask.shape)}\n"
            f"       query_images  {tuple(query_images.shape)}\n"
            f"       query_masks   {tuple(query_masks.shape)}\n"
            f"       obj_present   {tuple(obj_present.shape)}\n"
            f"       meta[0]       {meta[0]}"
        )

    # ── 2. Instantiate model ─────────────────────────────────────────────────
    print("\n[INFO] Initialising VideoMambaSystem …")
    model = VideoMambaSystem(
        dim_in=768,
        num_clf_classes=51,
        num_seg_classes=num_seg_classes,
        target_size=224,
    )
    model.eval()

    # ── 3. Forward pass ──────────────────────────────────────────────────────
    ref_img, ref_mask, query_images, query_masks, obj_present, meta = batch
    print("[INFO] Running forward pass …")
    with torch.no_grad():
        logits_clf, pred_boxes, pred_box_logits, logits_seg = model(
            query_images, ref_frame=ref_img, ref_mask=ref_mask
        )

    print(
        f"       logits_seg shape : {tuple(logits_seg.shape)}"
        f"  (expected [B, T, {num_seg_classes}, H, W])"
    )
    assert logits_seg.shape[2] == num_seg_classes, (
        f"Expected {num_seg_classes} seg channels, got {logits_seg.shape[2]}"
    )

    # ── 4. Shared step (loss + metric) ───────────────────────────────────────
    print("[INFO] Running _shared_step (val) …")
    with torch.no_grad():
        loss = model._shared_step(batch, batch_idx=0, prefix="val")

    print(f"       Loss: {loss.item():.6f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # ── 5. Metric compute ────────────────────────────────────────────────────
    metrics = model.vos_val_metric.compute()
    print(
        f"\n[INFO] VOS Metrics (J&F on synthetic/first batch):\n"
        f"       J   = {metrics['J'].item():.4f}\n"
        f"       F   = {metrics['F'].item():.4f}\n"
        f"       J&F = {metrics['J&F'].item():.4f}\n"
    )

    print("✓  Smoke test passed.")


if __name__ == "__main__":
    main()
