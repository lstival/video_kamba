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


def _build_instance_mask_from_bboxes(
    bboxes: list,
    height: int,
    width: int,
    n_id: int,
) -> np.ndarray:
    """Convert COCO bounding boxes to a rectangular label mask [H, W].

    Each bounding box is filled with a unique object ID (1..N).
    This replaces the segmentation-based approach since `detection-datasets/coco`
    only provides bounding boxes, not polygon/RLE masks.

    Args:
        bboxes:        List of bounding boxes in [x, y, w, h] format.
        height, width: Image spatial dimensions.
        n_id:          Maximum number of object IDs to keep.

    Returns:
        Integer label mask ``[H, W]`` with values 0 (bg) … n_id.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    for obj_id, bbox in enumerate(bboxes[:n_id], start=1):
        # bbox is [x, y, w, h] in COCO format
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
        else:
            continue
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(width,  int(x + w))
        y2 = min(height, int(y + h))
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = obj_id
    return canvas


def _augment_query(
    ref_img_pil: Image.Image,
    ref_mask_np: np.ndarray,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy PIL-based augmentation — kept for reference. Use _augment_query_fast."""
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

    brightness = random.uniform(0.7, 1.3)
    contrast   = random.uniform(0.7, 1.3)
    saturation = random.uniform(0.7, 1.3)
    hue        = random.uniform(-0.05, 0.05)
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    img = TF.adjust_hue(img, hue)

    img_t  = _normalize(TF.to_tensor(img.resize((img_size, img_size), Image.BILINEAR)))
    mask_t = _mask_to_tensor(np.array(mask_pil), img_size)

    if random.random() < 0.5:
        h_e = random.randint(img_size // 8, img_size // 4)
        w_e = random.randint(img_size // 8, img_size // 4)
        y0  = random.randint(0, img_size - h_e)
        x0  = random.randint(0, img_size - w_e)
        img_t[:, y0:y0 + h_e, x0:x0 + w_e] = 0.0

    return img_t, mask_t


def _augment_query_fast(
    img_raw: torch.Tensor,
    mask_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-based query augmentation — 2-3× faster than PIL equivalent.

    Operates on a pre-resized float [0, 1] tensor so the expensive PIL resize
    is performed only once per sample (in ``_process_sample``) rather than once
    per query frame.

    Args:
        img_raw:  ``[3, H, W]`` float32 tensor in ``[0, 1]`` (NOT yet normalised).
        mask_t:   ``[H, W]`` long tensor with object IDs.

    Returns:
        ``(query_img [3, H, W] normalised, query_mask [H, W] long)``
    """
    _, H, W = img_raw.shape

    angle     = random.uniform(-10, 10)
    translate = [int(W * random.uniform(-0.05, 0.05)),
                 int(H * random.uniform(-0.05, 0.05))]
    scale     = random.uniform(0.95, 1.05)
    flip      = random.random() < 0.5

    # Geometric — same params applied to image and mask
    img = TF.affine(img_raw, angle=angle, translate=translate, scale=scale, shear=0,
                    interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.affine(
        mask_t.unsqueeze(0).float(),
        angle=angle, translate=translate, scale=scale, shear=0,
        interpolation=TF.InterpolationMode.NEAREST,
    ).squeeze(0).round().long()

    if flip:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)

    # Colour jitter on [0, 1] float tensor — applied before normalisation
    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
    img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
    img = TF.adjust_hue(img,        random.uniform(-0.05, 0.05))
    img = img.clamp(0.0, 1.0)

    # Random erasing (image only)
    if random.random() < 0.5:
        h_e = random.randint(H // 8, H // 4)
        w_e = random.randint(W // 8, W // 4)
        y0  = random.randint(0, H - h_e)
        x0  = random.randint(0, W - w_e)
        img = img.clone()
        img[:, y0:y0 + h_e, x0:x0 + w_e] = 0.0

    return _normalize(img), mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCOPretrainDataset(IterableDataset):
    """Streaming COCO 2017 dataset for VOS pre-training.

    Uses HuggingFace ``streaming=True`` to avoid downloading the entire 20GB+
    dataset and converting it to Arrow format, which avoids disk quota issues.

    Args:
        img_size:   Spatial resolution of output tensors.
        seq_len:    Number of augmented query frames per clip (default 3).
        n_id:       Maximum number of tracked object IDs.
        split:      ``'train'`` or ``'validation'``.
        cache_dir:  HuggingFace datasets cache directory.
        max_samples: Cap the number of samples (useful for smoke-tests).
        shuffle:    Whether to use a shuffle buffer (only for train).
    """

    def __init__(
        self,
        img_size: int = 448,
        seq_len: int = 3,
        n_id: int = 10,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        assert _HF_AVAILABLE, (
            "HuggingFace `datasets` package not installed. "
            "Run: pip install datasets pycocotools"
        )
        self.img_size   = img_size
        self.seq_len    = seq_len
        self.n_id       = n_id
        self.split      = split
        self.max_samples = max_samples
        self.shuffle    = shuffle

        hf_split = "train" if split == "train" else "val"
        
        # Use streaming=True to bypass disk-heavy Arrow generation if not explicitly forced to False
        raw = load_dataset(
            "detection-datasets/coco",
            split=hf_split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        # Filter: keep images that have at least 1 bounding box
        def _has_masks(example):
            objs = example.get("objects", {})
            bboxes = objs.get("bbox", [])
            n    = len(bboxes)
            return 1 <= n <= _MAX_OBJECTS_PER_IMAGE

        raw = raw.filter(_has_masks)
        
        if self.shuffle:
            if streaming:
                raw = raw.shuffle(buffer_size=250, seed=42)
            else:
                raw = raw.shuffle(seed=42)
            
        if max_samples is not None:
            if streaming:
                raw = raw.take(max_samples)
            else:
                raw = raw.select(range(min(max_samples, len(raw))))

        self._data = raw

    def _process_sample(
        self, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Convert a raw COCO sample into a synthetic VOS clip.

        Key optimisation: PIL image is resized to ``img_size`` ONCE and
        converted to a float [0, 1] tensor.  All ``seq_len`` query frames
        are then produced via ``_augment_query_fast`` which operates
        entirely on that pre-resized tensor — eliminating the repeated
        full-resolution PIL resizes of the original implementation.
        """
        pil_img: Image.Image = sample["image"]
        H, W = pil_img.height, pil_img.width

        # ── Build instance mask ────────────────────────────────────────────
        bboxes = sample.get("objects", {}).get("bbox", [])
        mask_np = _build_instance_mask_from_bboxes(bboxes, H, W, self.n_id)

        # ── ONE-TIME resize: PIL → unnormalised float tensor ──────────────
        # This is the only PIL operation paid per sample; all query frames
        # are augmented from this pre-computed tensor.
        img_raw = TF.to_tensor(
            pil_img.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        )  # [3, H, W] float32 in [0, 1]
        mask_t = torch.from_numpy(
            np.array(
                Image.fromarray(mask_np.astype(np.uint8)).resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )
            )
        ).long()  # [H, W]

        # ── Reference frame ────────────────────────────────────────────────
        ref_img_t  = _normalize(img_raw)
        ref_mask_t = mask_t

        # ── Query frames via fast tensor augmentation ─────────────────────
        query_imgs:  List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            q_img, q_mask = _augment_query_fast(img_raw, mask_t)
            query_imgs.append(q_img)
            query_masks.append(q_mask)

        query_images  = torch.stack(query_imgs)   # [T, 3, H, W]
        query_masks_t = torch.stack(query_masks)  # [T, H, W]

        # ── Object-presence tensor ─────────────────────────────────────────
        obj_present = torch.zeros(self.seq_len, self.n_id, dtype=torch.bool)
        for t in range(self.seq_len):
            for k in range(1, self.n_id + 1):
                obj_present[t, k - 1] = (query_masks_t[t] == k).any()

        meta = {
            "video_id": f"coco_{sample.get('image_id', 'unknown')}",
            "seen_obj_ids": list(range(1, self.n_id + 1)),
            "unseen_obj_ids": [],
        }

        return ref_img_t, ref_mask_t, query_images, query_masks_t, obj_present, meta

    def __iter__(self):
        """Yield processed synthetic VOS clips.

        When running with multiple DataLoader workers, each worker is assigned
        a non-overlapping shard of the dataset so work is not duplicated.
        """
        data = self._data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and hasattr(data, "__len__"):
            # Map-style HF Dataset (streaming=False): shard by worker ID
            indices = list(range(worker_info.id, len(data), worker_info.num_workers))
            data = data.select(indices)

        for sample in data:
            try:
                yield self._process_sample(sample)
            except Exception as e:
                # Skip corrupted / incomplete samples silently
                continue


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
    Uses streaming mode to work within disk quota limits.
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
        streaming: bool = False,
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
                shuffle=True,  # Shuffle enabled for training
                streaming=hp.streaming,
            )
            self.val_dataset = COCOPretrainDataset(
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="validation",
                cache_dir=hp.cache_dir,
                max_samples=hp.max_samples,
                shuffle=False,
                streaming=hp.streaming,
            )
        if stage in ("validate",):
            self.val_dataset = COCOPretrainDataset(
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="validation",
                cache_dir=hp.cache_dir,
                max_samples=hp.max_samples,
                shuffle=False,
                streaming=hp.streaming,
            )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        # IterableDataset does not support shuffle=True in DataLoader;
        # shuffling is handled by HuggingFace's shuffle buffer in the dataset.
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=_coco_collate_fn,
            pin_memory=True,
            persistent_workers=hp.num_workers > 0,
            prefetch_factor=4 if hp.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)
