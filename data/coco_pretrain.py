"""COCO Pre-training DataModule for Video Object Segmentation.

Implements the AOT/DeAOT static-image pre-training strategy using COCO 2017
segmentation data from HuggingFace (``detection-datasets/coco``).

Each COCO image is turned into a synthetic VOS clip:
- **Reference frame**: original image + instance segmentation mask (objects
  remapped to IDs 1..n_id, background = 0).
- **Query frames**: ``seq_len`` augmented copies of the same image created with
  random colour jitter and affine transforms applied identically to image and
  mask.

This teaches the memory-bank + cross-attention to localise objects under
controlled appearance variation, without ever needing a video.

Batch format mirrors ``MultiObjectVOSDataModule`` exactly:
``(ref_img, ref_mask, query_images, query_masks, obj_present, meta)``

so ``train.py`` requires **zero changes** — just swap the datamodule config.

Training recipe (following AOT):
  1. Pre-train ~15 epochs on COCO (scheduled_sampling_rate=0.1, lr=2e-4).
  2. Fine-tune on YouTube-VOS from the resulting checkpoint.

References
----------
* AOT static pre-training: Yang et al., NeurIPS 2021.
* HuggingFace COCO: https://huggingface.co/datasets/detection-datasets/coco
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms import functional as TF

import lightning as L

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_MAX_OBJECTS_PER_IMAGE = 20  # skip images with more (too cluttered for pre-training)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(img: torch.Tensor) -> torch.Tensor:
    """Normalise a [3, H, W] float32 tensor with ImageNet stats."""
    return TF.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _pil_to_tensor(img: Image.Image, img_size: int) -> torch.Tensor:
    """Resize PIL image → normalised float32 [3, H, W]."""
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return _normalize(TF.to_tensor(img))


def _mask_to_tensor(mask: np.ndarray, img_size: int) -> torch.Tensor:
    """Resize binary/label mask → long [H, W]."""
    pil = Image.fromarray(mask.astype(np.uint8)).resize(
        (img_size, img_size), Image.NEAREST
    )
    return torch.from_numpy(np.array(pil)).long()


def _build_instance_mask(
    segmentation_list: list,
    height: int,
    width: int,
    n_id: int,
) -> np.ndarray:
    """Convert COCO RLE/polygon segmentations to a label mask [H, W].

    Objects are assigned IDs 1..N (trimmed to n_id).  Overlapping pixels are
    won by the later object in the list (matches COCO convention).

    Args:
        segmentation_list: List of dicts, each with ``'segmentation'`` and
                           (optionally) ``'area'``.  As returned by the HF
                           COCO dataset ``objects`` field.
        height, width:     Image spatial dimensions.
        n_id:              Maximum number of object IDs to keep.

    Returns:
        Integer label mask ``[H, W]`` with values 0 (bg) … n_id.
    """
    canvas = np.zeros((height, width), dtype=np.int32)
    from pycocotools import mask as mask_util
    obj_id = 0
    for ann in segmentation_list:
        obj_id += 1
        if obj_id > n_id:
            break
        seg = ann.get("segmentation")
        if seg is None:
            continue
        # RLE format
        if isinstance(seg, dict):
            rle = seg
            if isinstance(rle.get("counts"), list):
                rle = mask_util.frPyObjects(rle, height, width)
            m = mask_util.decode(rle).astype(bool)
        # Polygon format
        elif isinstance(seg, list) and len(seg) > 0:
            rles = mask_util.frPyObjects(seg, height, width)
            m = mask_util.decode(mask_util.merge(rles)).astype(bool)
        else:
            continue
        canvas[m] = obj_id
    return canvas.astype(np.uint8)


def _augment_query(
    ref_img_pil: Image.Image,
    ref_mask_np: np.ndarray,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create one augmented query frame from the reference PIL image.

    Augmentations (applied to image only — mask is warped geometrically):
      * Random colour jitter (brightness, contrast, saturation, hue)
      * Random horizontal flip
      * Small random affine (rotation ±10°, translation ±5%, scale 0.95–1.05)
      * Random erasing of a small patch (simulates partial occlusion)

    Args:
        ref_img_pil:  Original PIL image (any size).
        ref_mask_np:  Integer mask at the original resolution ``[H, W]``.
        img_size:     Output spatial resolution.

    Returns:
        (query_img_tensor [3, H, W], query_mask_tensor [H, W])
    """
    # ── geometric — applied to both image and mask ─────────────────────────
    angle = random.uniform(-10, 10)
    translate = [
        int(ref_img_pil.width * random.uniform(-0.05, 0.05)),
        int(ref_img_pil.height * random.uniform(-0.05, 0.05)),
    ]
    scale = random.uniform(0.95, 1.05)
    flip = random.random() < 0.5

    img = ref_img_pil.copy()
    mask_pil = Image.fromarray(ref_mask_np)

    img  = TF.affine(img,  angle=angle, translate=translate, scale=scale, shear=0)
    mask_pil = TF.affine(mask_pil, angle=angle, translate=translate, scale=scale, shear=0,
                         interpolation=TF.InterpolationMode.NEAREST)
    if flip:
        img  = TF.hflip(img)
        mask_pil = TF.hflip(mask_pil)

    # ── appearance — image only ────────────────────────────────────────────
    # Colour jitter
    brightness = random.uniform(0.7, 1.3)
    contrast   = random.uniform(0.7, 1.3)
    saturation = random.uniform(0.7, 1.3)
    hue        = random.uniform(-0.05, 0.05)
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    img = TF.adjust_hue(img, hue)

    # ── to tensors ─────────────────────────────────────────────────────────
    img_t  = _normalize(TF.to_tensor(img.resize((img_size, img_size), Image.BILINEAR)))
    mask_t = _mask_to_tensor(np.array(mask_pil), img_size)

    # ── random erasing (image only) ────────────────────────────────────────
    if random.random() < 0.5:
        h_e = random.randint(img_size // 8, img_size // 4)
        w_e = random.randint(img_size // 8, img_size // 4)
        y0  = random.randint(0, img_size - h_e)
        x0  = random.randint(0, img_size - w_e)
        img_t[:, y0:y0 + h_e, x0:x0 + w_e] = 0.0

    return img_t, mask_t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCOPretrainDataset(Dataset):
    """In-memory COCO 2017 dataset for VOS pre-training.

    Downloads (and caches) the HuggingFace ``detection-datasets/coco`` dataset
    on first use.  The ``instances_train2017.json`` provides object-level
    segmenation masks via ``pycocotools``.

    Args:
        img_size:   Spatial resolution of output tensors.
        seq_len:    Number of augmented query frames per clip (default 3).
        n_id:       Maximum number of tracked object IDs.
        split:      ``'train'`` or ``'validation'``.
        cache_dir:  HuggingFace datasets cache directory.
        max_samples: Cap the number of samples (useful for smoke-tests).
    """

    def __init__(
        self,
        img_size: int = 448,
        seq_len: int = 3,
        n_id: int = 10,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert _HF_AVAILABLE, (
            "HuggingFace `datasets` package not installed. "
            "Run: pip install datasets pycocotools"
        )
        self.img_size   = img_size
        self.seq_len    = seq_len
        self.n_id       = n_id

        # Load the full split into memory (COCO train ≈ 118k, HF returns PIL images)
        hf_split = "train" if split == "train" else "validation"
        raw = load_dataset(
            "detection-datasets/coco",
            split=hf_split,
            cache_dir=cache_dir,
        )

        # Filter: keep images with 1..n_id objects that have segmentation masks
        def _has_masks(example):
            objs = example.get("objects", {})
            segs = objs.get("segmentations", [])
            n    = len(segs)
            return 1 <= n <= _MAX_OBJECTS_PER_IMAGE

        raw = raw.filter(_has_masks)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self._data = raw

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Return a synthetic VOS clip from one COCO image.

        Returns the same 6-tuple as ``MultiObjectVOSDataset``:
        ``(ref_img, ref_mask, query_images, query_masks, obj_present, meta)``
        """
        sample = self._data[idx]
        pil_img: Image.Image = sample["image"]
        H, W = pil_img.height, pil_img.width

        # ── Build instance mask ────────────────────────────────────────────
        objs = sample.get("objects", {})
        # HF COCO format: objects["segmentations"] is a list of seg dicts
        seg_list = [
            {"segmentation": s}
            for s in objs.get("segmentations", [])
            if s is not None
        ]
        mask_np = _build_instance_mask(seg_list, H, W, self.n_id)

        # ── Reference frame ────────────────────────────────────────────────
        ref_img_t  = _pil_to_tensor(pil_img, self.img_size)
        ref_mask_t = _mask_to_tensor(mask_np, self.img_size)

        # ── Query frames (augmented copies) ───────────────────────────────
        query_imgs:  List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            q_img, q_mask = _augment_query(pil_img, mask_np, self.img_size)
            query_imgs.append(q_img)
            query_masks.append(q_mask)

        query_images = torch.stack(query_imgs)   # [T, 3, H, W]
        query_masks_t = torch.stack(query_masks)  # [T, H, W]

        # ── Object-presence tensor ─────────────────────────────────────────
        obj_present = torch.zeros(self.seq_len, self.n_id, dtype=torch.bool)
        for t in range(self.seq_len):
            for k in range(1, self.n_id + 1):
                obj_present[t, k - 1] = (query_masks_t[t] == k).any()

        meta = {
            "video_id": f"coco_{sample.get('image_id', idx)}",
            "seen_obj_ids": list(range(1, self.n_id + 1)),
            "unseen_obj_ids": [],
        }

        return ref_img_t, ref_mask_t, query_images, query_masks_t, obj_present, meta


# ---------------------------------------------------------------------------
# Collate (reuse from multi_object_vos)
# ---------------------------------------------------------------------------

def _coco_collate_fn(batch):
    ref_imgs    = torch.stack([b[0] for b in batch])
    ref_masks   = torch.stack([b[1] for b in batch])
    query_imgs  = torch.stack([b[2] for b in batch])
    query_masks = torch.stack([b[3] for b in batch])
    obj_present = torch.stack([b[4] for b in batch])
    metas       = [b[5] for b in batch]
    return ref_imgs, ref_masks, query_imgs, query_masks, obj_present, metas


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class COCOPretrainDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping ``COCOPretrainDataset``.

    Drop-in replacement for ``MultiObjectVOSDataModule`` — same batch format.

    Args:
        img_size:     Spatial resolution (default 448, matching DINOv2 patch grid).
        seq_len:      Augmented query frames per clip (default 3).
        n_id:         Identity bank size / max tracked objects (default 10).
        batch_size:   Mini-batch size (default 4 — static images are cheaper than video).
        num_workers:  DataLoader workers.
        cache_dir:    HuggingFace datasets download cache.
        max_samples:  Cap dataset size (set low for smoke-tests).
    """

    def __init__(
        self,
        img_size: int = 448,
        seq_len: int = 3,
        n_id: int = 10,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[COCOPretrainDataset] = None
        self.val_dataset:   Optional[COCOPretrainDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        if stage in ("fit", None):
            self.train_dataset = COCOPretrainDataset(
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="train",
                cache_dir=hp.cache_dir,
                max_samples=hp.max_samples,
            )
            self.val_dataset = COCOPretrainDataset(
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="validation",
                cache_dir=hp.cache_dir,
                max_samples=hp.max_samples,
            )
        if stage in ("validate",):
            self.val_dataset = COCOPretrainDataset(
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="validation",
                cache_dir=hp.cache_dir,
                max_samples=hp.max_samples,
            )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=shuffle,
            drop_last=shuffle,
            collate_fn=_coco_collate_fn,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)
