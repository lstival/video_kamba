"""Multi-Object Video Object Segmentation Dataset.

Supports YouTube-VOS 2019 and MOSE 2023 with a unified interface that
mirrors the existing ``DAVISDataset`` / ``DAVISDataModule`` API so the
rest of the training pipeline requires minimal changes.

Key design decisions
--------------------
* **Dataset layout normalisation** – both datasets share the same
  ``<split>/{JPEGImages,Annotations}/<video_id>/<frame>`` structure once
  extracted.  YouTube-VOS additionally provides a ``meta.json`` that
  records category membership (seen vs. unseen) per object.

* **First-frame anchor** – ``__getitem__`` always places the frame on
  which an object *first* appears (annotated frame index 0) as the
  reference / long-term memory frame.  Query frames are then sampled
  uniformly (val) or randomly (train) from the remainder.

* **Object-presence tensor** – because both YouTube-VOS and especially
  MOSE (41.5% disappearance rate) contain frames where a tracked object
  is absent, each item carries an ``obj_present`` boolean tensor of shape
  ``[T, n_id]`` so the loss function can mask out absent objects.

* **ID permutation** – during training, object IDs are randomly shuffled
  so the model never overfits to a canonical ordering.

* **Spatial augmentation** – random affine + thin-plate spline (TPS) warp
  via ``kornia`` applied identically to all frames & masks in a clip.

Batch format (returned by DataLoader)
--------------------------------------
``(ref_img, ref_mask, query_images, query_masks, obj_present, meta)``

=================  ==============================
Tensor             Shape
=================  ==============================
ref_img            ``[B, 3, H, W]``  float32
ref_mask           ``[B, H, W]``     int64
query_images       ``[B, T, 3, H, W]`` float32
query_masks        ``[B, T, H, W]``  int64
obj_present        ``[B, T, n_id]``  bool
=================  ==============================

``meta`` is a list of per-sample dicts containing ``video_id`` and,
for YouTube-VOS, ``seen_obj_ids`` / ``unseen_obj_ids`` lists.

References
----------
* YouTube-VOS 2019: https://youtube-vos.org/dataset/vos/
* MOSE 2023: https://github.com/henghuiding/MOSE-api
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

import lightning as L


# ---------------------------------------------------------------------------
# Optional kornia import (graceful degradation during unit tests w/o GPU)
# ---------------------------------------------------------------------------
try:
    import kornia.geometry.transform as KGT
    import kornia.augmentation as KA
    _KORNIA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _KORNIA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str, img_size: int) -> torch.Tensor:
    """Load a JPEG frame and return a normalised float32 tensor ``[3, H, W]``."""
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, (img_size, img_size))
    tensor = TF.normalize(TF.to_tensor(img), mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return tensor  # [3, H, W]


def _load_mask(path: Optional[str], img_size: int) -> torch.Tensor:
    """Load an annotation PNG and return an int64 tensor ``[H, W]``.

    Mask pixel values encode object IDs (0 = background, 1…N = objects).
    If ``path`` is ``None`` or the file does not exist, returns a zero mask.
    """
    if path is None or not os.path.exists(path):
        return torch.zeros(img_size, img_size, dtype=torch.long)
    mask = Image.open(path)
    mask = TF.resize(mask, (img_size, img_size), interpolation=TF.InterpolationMode.NEAREST)
    return torch.from_numpy(np.array(mask)).long()


def _apply_permutation(masks: List[torch.Tensor], n_id: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Randomly permute object IDs in a list of masks.

    Background (0) is left unchanged.  The random permutation is applied to
    IDs 1…n_id inclusive.

    Args:
        masks: List of ``[H, W]`` integer tensors.
        n_id:  Number of object ID slots (identity bank size).

    Returns:
        permuted_masks: Same list with remapped IDs.
        perm:           1-D LongTensor of length ``n_id`` where ``perm[i]``
                        is the new ID assigned to the original object ``i+1``.
    """
    perm = torch.randperm(n_id) + 1  # values in [1, n_id]
    # Build a lookup table: old_id -> new_id (index 0 = background, stays 0)
    lut = torch.zeros(n_id + 1, dtype=torch.long)
    for new_id, old_id in enumerate(perm.tolist(), start=1):
        lut[old_id] = new_id

    permuted = []
    for mask in masks:
        # Clamp IDs exceeding n_id to background (safety)
        safe_mask = mask.clamp(0, n_id)
        permuted.append(lut[safe_mask])
    return permuted, perm


def _build_obj_present(masks: List[torch.Tensor], n_id: int) -> torch.Tensor:
    """Compute an object-presence boolean matrix for a sequence of masks.

    Args:
        masks: List of T ``[H, W]`` integer tensors (query frames only).
        n_id:  Number of object ID slots.

    Returns:
        Boolean tensor of shape ``[T, n_id]`` where entry ``[t, k]`` is
        ``True`` iff object ``k+1`` (1-indexed) is present in frame ``t``.
    """
    T = len(masks)
    obj_present = torch.zeros(T, n_id, dtype=torch.bool)
    for t, mask in enumerate(masks):
        for k in range(1, n_id + 1):
            obj_present[t, k - 1] = (mask == k).any()
    return obj_present


# ---------------------------------------------------------------------------
# Spatial augmentation (kornia)
# ---------------------------------------------------------------------------

class _SpatialAugmentor:
    """Apply the same random affine + TPS warp to frames and masks.

    Uses ``kornia`` for GPU-capable augmentation.  Falls back to a no-op
    if kornia is unavailable (e.g., CPU-only test environments).

    Args:
        img_size:           Spatial resolution after augmentation.
        affine_degrees:     Max rotation ± degrees.
        affine_translate:   Max translation as fraction of image size.
        affine_scale:       Scale range ``(min, max)``.
        tps_num_ctrl_pts:   Number of TPS control points (sqrt).
    """

    def __init__(
        self,
        img_size: int = 480,
        affine_degrees: float = 15.0,
        affine_translate: Tuple[float, float] = (0.05, 0.05),
        affine_scale: Tuple[float, float] = (0.9, 1.1),
        tps_num_ctrl_pts: int = 4,
    ) -> None:
        self.img_size = img_size
        self._available = _KORNIA_AVAILABLE

        if self._available:
            # Random affine parameters (sampled once per clip, shared across all frames)
            self._affine = KA.RandomAffine(
                degrees=affine_degrees,
                translate=affine_translate,
                scale=affine_scale,
                p=1.0,
                same_on_batch=True,
            )
            self._tps_num_ctrl = tps_num_ctrl_pts

    def __call__(
        self,
        images: torch.Tensor,   # [N, 3, H, W]  float32, normalised
        masks: torch.Tensor,    # [N, H, W]     int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply identical augmentation to images and masks.

        Args:
            images: Float tensor ``[N, 3, H, W]``.
            masks:  Integer tensor ``[N, H, W]``.

        Returns:
            aug_images: ``[N, 3, H, W]``
            aug_masks:  ``[N, H, W]``
        """
        if not self._available:
            return images, masks

        N = images.shape[0]
        H = W = self.img_size

        # ── 1. Random Affine ──────────────────────────────────────────────
        # Stack into a batch so same_on_batch produces identical transforms
        aug_images = self._affine(images)
        # Apply same transform to masks (treated as 1-channel float, then convert back)
        masks_float = masks.unsqueeze(1).float()  # [N, 1, H, W]
        # Retrieve and re-apply the last affine params (same_on_batch=True)
        masks_aug = self._affine(masks_float, params=self._affine._params)
        masks_aug = masks_aug.squeeze(1).round().long()  # [N, H, W]

        # ── 2. Thin-Plate Spline Warp ─────────────────────────────────────
        n = self._tps_num_ctrl
        # Source grid: uniform [n×n] points in [-1, 1] normalised coords
        grid_1d = torch.linspace(-0.8, 0.8, n, device=images.device)
        yy, xx = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        src_pts = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [n², 2]
        # Perturb to create dst points (small random displacement)
        noise = (torch.rand_like(src_pts) - 0.5) * 0.08
        dst_pts = (src_pts + noise).clamp(-1.0, 1.0)

        src_pts = src_pts.unsqueeze(0).expand(N, -1, -1)  # [N, n², 2]
        dst_pts = dst_pts.unsqueeze(0).expand(N, -1, -1)

        # Compute TPS kernel weights
        kernel, affine_tps = KGT.get_tps_transform(dst_pts, src_pts)

        # Warp images
        img_warped = KGT.warp_image_tps(aug_images, src_pts, kernel, affine_tps)

        # Warp masks (float → round → long)
        masks_float2 = masks_aug.unsqueeze(1).float()
        masks_warped = KGT.warp_image_tps(
            masks_float2, src_pts, kernel, affine_tps, interp_mode="nearest"
        )
        masks_warped = masks_warped.squeeze(1).round().long()

        return img_warped, masks_warped


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class MultiObjectVOSDataset(Dataset):
    """Unified dataset for YouTube-VOS 2019 and MOSE 2023.

    Args:
        root_dir:     Dataset root, e.g. ``data/YouTubeVOS`` or ``data/MOSE``.
        dataset_type: ``"youtubevos"`` or ``"mose"``.
        split:        ``"train"`` or ``"valid"``.
        seq_len:      Number of query frames sampled per clip (default 5).
        img_size:     Spatial resolution of returned tensors (default 480).
        n_id:         Identity bank size – maximum number of tracked objects
                      per clip.  IDs exceeding this are clipped to background
                      at load time (default 10).
        augment:      Whether to apply spatial augmentation (train only).
    """

    @staticmethod
    def _extract_categories(meta_path: str) -> Set[str]:
        """Extract all unique category names from a YouTube-VOS meta.json."""
        cats: Set[str] = set()
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    raw = json.load(f)
                for v in raw.get("videos", {}).values():
                    for o in v.get("objects", {}).values():
                        c = o.get("category")
                        if c:
                            cats.add(c)
            except (json.JSONDecodeError, IOError):
                pass
        return cats

    def __init__(
        self,
        root_dir: str,
        dataset_type: Literal["youtubevos", "mose"],
        split: Literal["train", "valid"] = "train",
        seq_len: int = 5,
        img_size: int = 480,
        n_id: int = 10,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.split = split
        self.seq_len = seq_len
        self.img_size = img_size
        self.n_id = n_id
        self.augment = augment

        self._img_dir = os.path.join(root_dir, split, "JPEGImages")
        self._ann_dir = os.path.join(root_dir, split, "Annotations")

        # Parse video list
        self._video_ids: List[str] = sorted(
            d for d in os.listdir(self._img_dir)
            if os.path.isdir(os.path.join(self._img_dir, d))
        )

        # YouTube-VOS: parse meta.json for seen/unseen category tracking
        self._meta: Dict[str, Dict] = {}
        if dataset_type == "youtubevos":
            # Identify unseen categories by comparing train vs valid
            train_cats = self._extract_categories(os.path.join(root_dir, "train", "meta.json"))
            valid_cats = self._extract_categories(os.path.join(root_dir, "valid", "meta.json"))
            unseen_categories = valid_cats - train_cats
            
            meta_path = os.path.join(root_dir, split, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as fh:
                    raw = json.load(fh)
                # raw["videos"][video_id]["objects"][obj_id] = {"category": ..., "frames": [...]}
                for vid_id, vid_data in raw.get("videos", {}).items():
                    seen_ids: List[int] = []
                    unseen_ids: List[int] = []
                    for obj_str, obj_info in vid_data.get("objects", {}).items():
                        oid = int(obj_str)
                        category = obj_info.get("category", "")
                        # Classify based on category since "split" field is missing
                        if category in unseen_categories:
                            unseen_ids.append(oid)
                        else:
                            seen_ids.append(oid)
                    
                    self._meta[vid_id] = {
                        "seen_obj_ids": seen_ids,
                        "unseen_obj_ids": unseen_ids,
                    }

        # Build per-video frame lists
        self._video_frames: Dict[str, List[str]] = {}
        for vid in self._video_ids:
            frames = sorted(
                f for f in os.listdir(os.path.join(self._img_dir, vid))
                if f.endswith(".jpg")
            )
            if len(frames) >= 2:  # need at least ref + 1 query
                self._video_frames[vid] = frames

        self._video_ids = [v for v in self._video_ids if v in self._video_frames]

        # Augmentor (no-op if augment=False or kornia unavailable)
        self._augmentor = _SpatialAugmentor(img_size=img_size) if augment else None

    def __len__(self) -> int:
        return len(self._video_ids)

    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Return one clip.

        Returns:
            ref_img:       ``[3, H, W]`` float32 – reference frame
            ref_mask:      ``[H, W]``    int64   – reference annotation
            query_images:  ``[T, 3, H, W]`` float32
            query_masks:   ``[T, H, W]``    int64
            obj_present:   ``[T, n_id]``    bool
            meta:          dict with ``video_id``, ``seen_obj_ids``, ``unseen_obj_ids``
        """
        vid = self._video_ids[idx]
        frames = self._video_frames[vid]
        num_frames = len(frames)

        # ── Select query indices ──────────────────────────────────────────
        # Frame 0 is always the reference / long-term memory anchor.
        # Query frames are drawn from frames[1:].
        candidate_range = list(range(1, num_frames))

        if self.split == "train":
            # Random sampling without replacement (or with if video is short)
            k = min(self.seq_len, len(candidate_range))
            query_indices = sorted(random.sample(candidate_range, k))
            # Pad by repeating the last frame if shorter than seq_len
            while len(query_indices) < self.seq_len:
                query_indices.append(query_indices[-1])
        else:
            # Uniform stride for reproducible validation
            stride = max(1, len(candidate_range) // self.seq_len)
            query_indices = candidate_range[::stride][: self.seq_len]
            while len(query_indices) < self.seq_len:
                query_indices.append(query_indices[-1])

        # ── Load frames ───────────────────────────────────────────────────
        def img_path(frame: str) -> str:
            return os.path.join(self._img_dir, vid, frame)

        def ann_path(frame: str) -> Optional[str]:
            ann_name = os.path.splitext(frame)[0] + ".png"
            p = os.path.join(self._ann_dir, vid, ann_name)
            return p if os.path.exists(p) else None

        ref_frame_name = frames[0]
        ref_img_t = _load_image(img_path(ref_frame_name), self.img_size)
        ref_mask_t = _load_mask(ann_path(ref_frame_name), self.img_size)

        query_imgs: List[torch.Tensor] = []
        query_masks_raw: List[torch.Tensor] = []
        for qi in query_indices:
            query_imgs.append(_load_image(img_path(frames[qi]), self.img_size))
            query_masks_raw.append(_load_mask(ann_path(frames[qi]), self.img_size))

        # ── Clamp IDs to [0, n_id] ────────────────────────────────────────
        ref_mask_t = ref_mask_t.clamp(0, self.n_id)
        query_masks_raw = [m.clamp(0, self.n_id) for m in query_masks_raw]

        # ── Random permutation (train only) ───────────────────────────────
        if self.split == "train":
            all_masks = [ref_mask_t] + query_masks_raw
            all_masks, _perm = _apply_permutation(all_masks, self.n_id)
            ref_mask_t = all_masks[0]
            query_masks_raw = all_masks[1:]

        # ── Spatial augmentation (train only, kornia) ─────────────────────
        if self.augment and self._augmentor is not None and self.split == "train":
            all_imgs = torch.stack([ref_img_t] + query_imgs)        # [1+T, 3, H, W]
            all_masks_t = torch.stack([ref_mask_t] + query_masks_raw)  # [1+T, H, W]
            all_imgs, all_masks_t = self._augmentor(all_imgs, all_masks_t)
            ref_img_t = all_imgs[0]
            ref_mask_t = all_masks_t[0]
            query_imgs = [all_imgs[i + 1] for i in range(len(query_imgs))]
            query_masks_raw = [all_masks_t[i + 1] for i in range(len(query_masks_raw))]

        # ── Stack query tensors ───────────────────────────────────────────
        query_images = torch.stack(query_imgs)              # [T, 3, H, W]
        query_masks = torch.stack(query_masks_raw)          # [T, H, W]

        # ── Object presence ───────────────────────────────────────────────
        obj_present = _build_obj_present(query_masks_raw, self.n_id)  # [T, n_id]

        # ── Metadata ──────────────────────────────────────────────────────
        meta = {"video_id": vid}
        meta.update(self._meta.get(vid, {"seen_obj_ids": [], "unseen_obj_ids": []}))

        return ref_img_t, ref_mask_t, query_images, query_masks, obj_present, meta


# ---------------------------------------------------------------------------
# Collate function (handles meta dict in batch)
# ---------------------------------------------------------------------------

def _vos_collate_fn(
    batch: List[Tuple],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """Collate function that separates tensor fields from meta dicts.

    Returns:
        ref_imgs:      ``[B, 3, H, W]``
        ref_masks:     ``[B, H, W]``
        query_images:  ``[B, T, 3, H, W]``
        query_masks:   ``[B, T, H, W]``
        obj_present:   ``[B, T, n_id]``
        metas:         list of B dicts
    """
    ref_imgs = torch.stack([b[0] for b in batch])
    ref_masks = torch.stack([b[1] for b in batch])
    query_images = torch.stack([b[2] for b in batch])
    query_masks = torch.stack([b[3] for b in batch])
    obj_present = torch.stack([b[4] for b in batch])
    metas = [b[5] for b in batch]
    return ref_imgs, ref_masks, query_images, query_masks, obj_present, metas


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class MultiObjectVOSDataModule(L.LightningDataModule):
    """PyTorch-Lightning DataModule for YouTube-VOS 2019 and MOSE 2023.

    Mirrors the ``DAVISDataModule`` API so config-driven model training
    requires no changes to ``train.py``.

    Args:
        data_dir:     Root directory of the chosen dataset.
        dataset_type: ``"youtubevos"`` or ``"mose"``.
        batch_size:   Mini-batch size (default 2 – 480p frames are large).
        num_workers:  DataLoader worker processes.
        seq_len:      Number of query frames per clip (default 5).
        img_size:     Spatial resolution (default 480).
        n_id:         Identity bank size / max tracked objects (default 10).
        augment_train: Apply spatial augmentation on training set.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: Literal["youtubevos", "mose"] = "youtubevos",
        batch_size: int = 2,
        num_workers: int = 4,
        seq_len: int = 5,
        img_size: int = 480,
        n_id: int = 10,
        augment_train: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[MultiObjectVOSDataset] = None
        self.val_dataset: Optional[MultiObjectVOSDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate train and/or val datasets.

        Args:
            stage: ``"fit"``, ``"validate"``, ``"test"``, or ``None`` (all).
        """
        hp = self.hparams
        if stage in ("fit", None):
            self.train_dataset = MultiObjectVOSDataset(
                root_dir=hp.data_dir,
                dataset_type=hp.dataset_type,
                split="train",
                seq_len=hp.seq_len,
                img_size=hp.img_size,
                n_id=hp.n_id,
                augment=hp.augment_train,
            )
            self.val_dataset = MultiObjectVOSDataset(
                root_dir=hp.data_dir,
                dataset_type=hp.dataset_type,
                split="valid",
                seq_len=hp.seq_len,
                img_size=hp.img_size,
                n_id=hp.n_id,
                augment=False,
            )

        if stage in ("validate",):
            self.val_dataset = MultiObjectVOSDataset(
                root_dir=hp.data_dir,
                dataset_type=hp.dataset_type,
                split="valid",
                seq_len=hp.seq_len,
                img_size=hp.img_size,
                n_id=hp.n_id,
                augment=False,
            )

        if stage in ("test", None):
            # Val split doubles as test (test-dev annotations not public)
            self.test_dataset = MultiObjectVOSDataset(
                root_dir=hp.data_dir,
                dataset_type=hp.dataset_type,
                split="valid",
                seq_len=hp.seq_len,
                img_size=hp.img_size,
                n_id=hp.n_id,
                augment=False,
            )

    # ── DataLoader factories ───────────────────────────────────────────────

    def _make_loader(self, dataset: MultiObjectVOSDataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=shuffle,
            drop_last=shuffle,
            collate_fn=_vos_collate_fn,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)
