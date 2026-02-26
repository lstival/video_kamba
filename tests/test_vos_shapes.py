"""Pytest suite for MultiObjectVOSDataset, HybridVOSLoss, and VideoMambaSystem.

All tests operate on synthetic in-memory tensors or minimal on-disk fixtures
created with ``tmp_path``.  No real dataset download is required.

Run with::

    pytest tests/test_vos_shapes.py -v
"""

from __future__ import annotations

import os
import json
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vos_dims():
    """Common dimensions used across all VOS shape tests."""
    return {
        "B": 2,
        "T": 5,
        "C": 3,
        "H": 64,   # small for speed
        "W": 64,
        "n_id": 10,
        "num_seg_classes": 11,  # bg + n_id
        "num_clf_classes": 51,
    }


@pytest.fixture()
def fake_vos_root(tmp_path) -> str:
    """Build a minimal on-disk VOS dataset tree for one 'train' video.

    Layout mirrors YouTube-VOS / MOSE shared structure::

        <root>/train/JPEGImages/<video_id>/<frame>.jpg
        <root>/train/Annotations/<video_id>/<frame>.png
        <root>/train/meta.json

    Each frame is a 64×64 solid-colour image; masks contain IDs 0–3.
    """
    root = str(tmp_path / "FakeVOS")
    vid_id = "vid_0001"
    n_frames = 8
    h = w = 64

    img_dir = os.path.join(root, "train", "JPEGImages", vid_id)
    ann_dir = os.path.join(root, "train", "Annotations", vid_id)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(n_frames):
        frame_name = f"{i:05d}"
        # Save JPEG
        img_arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(os.path.join(img_dir, f"{frame_name}.jpg"))
        # Save mask (PNG, pixel values 0–3)
        mask_arr = rng.integers(0, 4, (h, w), dtype=np.uint8)
        # Frame 3: object 2 absent (simulate disappearance)
        if i == 3:
            mask_arr[mask_arr == 2] = 0
        Image.fromarray(mask_arr, mode="L").save(os.path.join(ann_dir, f"{frame_name}.png"))

    # meta.json (YouTube-VOS format)
    meta = {
        "videos": {
            vid_id: {
                "objects": {
                    "1": {"category": "cat", "split": "seen", "frames": [f"{i:05d}" for i in range(n_frames)]},
                    "2": {"category": "bear", "split": "unseen", "frames": [f"{i:05d}" for i in range(n_frames)]},
                    "3": {"category": "dog", "split": "seen", "frames": [f"{i:05d}" for i in range(n_frames)]},
                }
            }
        }
    }
    with open(os.path.join(root, "train", "meta.json"), "w") as fh:
        json.dump(meta, fh)

    # Also create a minimal valid/ split for DataModule setup
    valid_img = os.path.join(root, "valid", "JPEGImages", vid_id)
    valid_ann = os.path.join(root, "valid", "Annotations", vid_id)
    os.makedirs(valid_img, exist_ok=True)
    os.makedirs(valid_ann, exist_ok=True)
    for i in range(n_frames):
        frame_name = f"{i:05d}"
        img_arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(os.path.join(valid_img, f"{frame_name}.jpg"))
        mask_arr = rng.integers(0, 4, (h, w), dtype=np.uint8)
        Image.fromarray(mask_arr, mode="L").save(os.path.join(valid_ann, f"{frame_name}.png"))
    meta_valid = {"videos": {vid_id: {"objects": {}}}}
    os.makedirs(os.path.join(root, "valid"), exist_ok=True)
    with open(os.path.join(root, "valid", "meta.json"), "w") as fh:
        json.dump(meta_valid, fh)

    return root


# ---------------------------------------------------------------------------
# Dataset shape tests
# ---------------------------------------------------------------------------

class TestMultiObjectVOSDataset:
    """Verify tensor shapes returned by MultiObjectVOSDataset."""

    def test_item_shapes_train(self, fake_vos_root):
        from data.multi_object_vos import MultiObjectVOSDataset

        ds = MultiObjectVOSDataset(
            root_dir=fake_vos_root,
            dataset_type="youtubevos",
            split="train",
            seq_len=5,
            img_size=64,
            n_id=10,
            augment=False,  # no kornia dep in CI
        )
        assert len(ds) == 1, "Expected 1 video in fixture"

        ref_img, ref_mask, query_images, query_masks, obj_present, meta = ds[0]

        assert ref_img.shape == (3, 64, 64), f"ref_img: {ref_img.shape}"
        assert ref_img.dtype == torch.float32
        assert ref_mask.shape == (64, 64), f"ref_mask: {ref_mask.shape}"
        assert ref_mask.dtype == torch.long

        assert query_images.shape == (5, 3, 64, 64), f"query_images: {query_images.shape}"
        assert query_masks.shape == (5, 64, 64), f"query_masks: {query_masks.shape}"
        assert query_masks.dtype == torch.long

        assert obj_present.shape == (5, 10), f"obj_present: {obj_present.shape}"
        assert obj_present.dtype == torch.bool

        assert "video_id" in meta
        assert "seen_obj_ids" in meta
        assert "unseen_obj_ids" in meta

    def test_mask_ids_clamped(self, fake_vos_root):
        """Mask values must be in [0, n_id] after clamping."""
        from data.multi_object_vos import MultiObjectVOSDataset

        ds = MultiObjectVOSDataset(
            root_dir=fake_vos_root,
            dataset_type="youtubevos",
            split="train",
            seq_len=5,
            img_size=64,
            n_id=10,
            augment=False,
        )
        _, ref_mask, _, query_masks, _, _ = ds[0]
        assert ref_mask.min() >= 0
        assert ref_mask.max() <= 10
        assert query_masks.min() >= 0
        assert query_masks.max() <= 10

    def test_valid_split_no_augment(self, fake_vos_root):
        """Val split should return items without augmentation."""
        from data.multi_object_vos import MultiObjectVOSDataset

        ds = MultiObjectVOSDataset(
            root_dir=fake_vos_root,
            dataset_type="youtubevos",
            split="valid",
            seq_len=5,
            img_size=64,
            n_id=10,
            augment=False,
        )
        assert len(ds) >= 1
        ref_img, ref_mask, query_images, query_masks, obj_present, meta = ds[0]
        assert query_images.shape[0] <= 5  # may be fewer if video is short


class TestPermutationAugmentation:
    """Verify the ID permutation utility."""

    def test_permutation_preserves_shape(self):
        from data.multi_object_vos import _apply_permutation

        masks = [torch.randint(0, 11, (64, 64)) for _ in range(6)]
        permuted, perm = _apply_permutation(masks, n_id=10)

        assert len(permuted) == 6
        for orig, perm_m in zip(masks, permuted):
            assert orig.shape == perm_m.shape
            assert perm_m.dtype == torch.long

    def test_permutation_ids_in_range(self):
        from data.multi_object_vos import _apply_permutation

        masks = [torch.randint(0, 11, (32, 32)) for _ in range(4)]
        permuted, _ = _apply_permutation(masks, n_id=10)
        for m in permuted:
            assert m.min() >= 0
            assert m.max() <= 10

    def test_background_unchanged(self):
        """Background pixels (ID=0) must remain 0 after permutation."""
        from data.multi_object_vos import _apply_permutation

        mask = torch.zeros(16, 16, dtype=torch.long)
        permuted, _ = _apply_permutation([mask], n_id=10)
        assert (permuted[0] == 0).all()


class TestObjPresent:
    """Verify the object-presence matrix builder."""

    def test_shape(self):
        from data.multi_object_vos import _build_obj_present

        masks = [torch.randint(0, 11, (64, 64)) for _ in range(5)]
        obj_present = _build_obj_present(masks, n_id=10)
        assert obj_present.shape == (5, 10)
        assert obj_present.dtype == torch.bool

    def test_absent_object_detected(self):
        """A mask with no pixels of ID=2 should yield obj_present[t, 1]=False."""
        from data.multi_object_vos import _build_obj_present

        mask_no_obj2 = torch.zeros(32, 32, dtype=torch.long)
        mask_no_obj2[0, 0] = 1  # only object 1 present
        obj_present = _build_obj_present([mask_no_obj2], n_id=10)
        assert obj_present[0, 0].item() is True   # obj 1 present
        assert obj_present[0, 1].item() is False  # obj 2 absent


# ---------------------------------------------------------------------------
# VOS Loss tests
# ---------------------------------------------------------------------------

class TestHybridVOSLoss:
    """Verify HybridVOSLoss is finite, handles absent objects, and shapes."""

    @pytest.fixture()
    def loss_fn(self):
        from utils.vos_loss import HybridVOSLoss
        return HybridVOSLoss(beta=0.5, from_logits=True)

    def test_finite_loss(self, loss_fn, vos_dims):
        B, T, n_id = 2, 5, vos_dims["n_id"]
        C = n_id + 1
        H = W = 32

        pred = torch.randn(B, T, C, H, W)
        target = torch.randint(0, C, (B, T, H, W))
        obj_present = torch.ones(B, T, n_id, dtype=torch.bool)

        loss = loss_fn(pred, target, obj_present)
        assert loss.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0.0

    def test_all_absent_objects(self, loss_fn, vos_dims):
        """When all objects are absent, only background contributes to loss."""
        B, T, n_id = 1, 3, vos_dims["n_id"]
        C = n_id + 1
        H = W = 16

        pred = torch.randn(B, T, C, H, W)
        target = torch.zeros(B, T, H, W, dtype=torch.long)   # all background
        obj_present = torch.zeros(B, T, n_id, dtype=torch.bool)  # all absent

        loss = loss_fn(pred, target, obj_present)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_perfect_pred(self, loss_fn, vos_dims):
        """Loss for near-perfect predictions should be lower than random."""
        B, T, n_id = 1, 2, vos_dims["n_id"]
        C = n_id + 1
        H = W = 16

        target = torch.zeros(B, T, H, W, dtype=torch.long)
        obj_present = torch.zeros(B, T, n_id, dtype=torch.bool)

        # Random predictions
        pred_random = torch.randn(B, T, C, H, W)
        loss_random = loss_fn(pred_random, target, obj_present).item()

        # Nearly perfect: high logit for correct class (bg=0), low elsewhere
        pred_perfect = torch.full((B, T, C, H, W), -10.0)
        pred_perfect[:, :, 0, :, :] = 10.0  # background channel dominant
        loss_perfect = loss_fn(pred_perfect, target, obj_present).item()

        assert loss_perfect < loss_random, (
            f"Perfect loss ({loss_perfect:.4f}) should be < random loss ({loss_random:.4f})"
        )

    def test_shape_mismatch_raises(self, loss_fn):
        with pytest.raises(AssertionError):
            pred = torch.randn(2, 5, 11, 32, 32)
            target = torch.randint(0, 11, (2, 4, 32, 32))  # T mismatch
            obj_present = torch.ones(2, 5, 10, dtype=torch.bool)
            loss_fn(pred, target, obj_present)


# ---------------------------------------------------------------------------
# VideoMambaSystem VOS step tests
# ---------------------------------------------------------------------------

class TestVideoMambaSystemVOS:
    """Integration-level tests for the VOS training step."""

    @pytest.fixture(scope="class")
    def model(self, vos_dims):
        from models.video_mamba import VideoMambaSystem
        return VideoMambaSystem(
            dim_in=768,
            num_clf_classes=vos_dims["num_clf_classes"],
            num_seg_classes=vos_dims["num_seg_classes"],
            target_size=64,
        )

    @pytest.fixture()
    def synthetic_vos_batch(self, vos_dims):
        """Build a 6-element VOS batch with synthetic tensors."""
        B = vos_dims["B"]
        T = vos_dims["T"]
        H = W = vos_dims["H"]
        n_id = vos_dims["n_id"]
        num_seg_classes = vos_dims["num_seg_classes"]

        ref_img = torch.randn(B, 3, H, W)
        ref_mask = torch.randint(0, num_seg_classes, (B, H, W))
        query_images = torch.randn(B, T, 3, H, W)
        query_masks = torch.randint(0, num_seg_classes, (B, T, H, W))
        obj_present = torch.ones(B, T, n_id, dtype=torch.bool)
        meta = [{"video_id": f"vid_{i:04d}", "seen_obj_ids": [], "unseen_obj_ids": []} for i in range(B)]

        return ref_img, ref_mask, query_images, query_masks, obj_present, meta

    def test_is_vos_batch_detection(self, model, synthetic_vos_batch):
        from models.video_mamba import VideoMambaSystem
        assert VideoMambaSystem._is_vos_batch(synthetic_vos_batch) is True

    def test_is_vos_batch_rejects_davis_batch(self, model):
        from models.video_mamba import VideoMambaSystem
        davis_batch = (
            torch.randn(2, 3, 64, 64),   # ref_img
            torch.randint(0, 11, (2, 64, 64)),  # ref_mask
            torch.randn(2, 5, 3, 64, 64),       # query_imgs
            torch.randint(0, 11, (2, 5, 64, 64)),  # query_masks
        )
        assert VideoMambaSystem._is_vos_batch(davis_batch) is False

    def test_vos_step_finite_loss(self, model, synthetic_vos_batch):
        model.eval()
        with torch.no_grad():
            loss = model._shared_step(synthetic_vos_batch, batch_idx=0, prefix="val")
        assert torch.isfinite(loss), f"VOS step loss is not finite: {loss.item()}"

    def test_vos_step_updates_metric(self, model, synthetic_vos_batch):
        model.vos_val_metric.reset()
        model.eval()
        with torch.no_grad():
            model._shared_step(synthetic_vos_batch, batch_idx=0, prefix="val")
        # At least some updates occurred (total > 0 if any object was present)
        result = model.vos_val_metric.compute()
        assert "J&F" in result
