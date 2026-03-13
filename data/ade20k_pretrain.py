"""ADE20K Pre-training DataModule for Video Object Segmentation.

Implements the AOT/DeAOT static-image pre-training strategy using ADE20K 
segmentation data from HuggingFace (``scene_parse_150``).

Each ADE20K image is turned into a synthetic VOS clip:
- **Reference frame**: original image + semantic/instance segmentation mask.
- **Query frames**: ``seq_len`` augmented copies of the same image.

This teaches the model to localise and track textures/objects under 
controlled appearance variation.

References
----------
* ADE20K on HF: https://huggingface.co/datasets/scene_parse_150
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

_MAX_OBJECTS_PER_IMAGE = 20  # Remap up to 20 dominant classes/instances


# ---------------------------------------------------------------------------
# Helpers (Shared with COCO logic for consistency)
# ---------------------------------------------------------------------------

def _normalize(img: torch.Tensor) -> torch.Tensor:
    return TF.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _augment_query_fast(
    img_raw: torch.Tensor,
    mask_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-based query augmentation — matches AOT recipes."""
    _, H, W = img_raw.shape

    angle     = random.uniform(-10, 10)
    translate = [int(W * random.uniform(-0.05, 0.05)),
                 int(H * random.uniform(-0.05, 0.05))]
    scale     = random.uniform(0.95, 1.05)
    flip      = random.random() < 0.5

    # Geometric
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

    # Colour jitter
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

class ADE20KPretrainDataset(IterableDataset):
    """Streaming ADE20K dataset for VOS pre-training.

    Args:
        img_size:   Spatial resolution.
        seq_len:    Augmented query frames.
        n_id:       Max objects to track.
        split:      'train' or 'validation'.
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
        assert _HF_AVAILABLE, "HuggingFace `datasets` package not installed."
        
        self.img_size   = img_size
        self.seq_len    = seq_len
        self.n_id       = n_id
        self.split      = split
        self.max_samples = max_samples
        self.shuffle    = shuffle

        hf_split = "train" if split == "train" else "validation"
        
        # Use 1aurent/ADE20K which is stored in Parquet format (modern, script-free)
        self.raw = load_dataset(
            "1aurent/ADE20K",
            split=hf_split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        if self.shuffle:
            if streaming:
                self.raw = self.raw.shuffle(buffer_size=250, seed=42)
            else:
                self.raw = self.raw.shuffle(seed=42)
            
        if max_samples is not None:
            if streaming:
                self.raw = self.raw.take(max_samples)
            else:
                self.raw = self.raw.select(range(min(max_samples, len(self.raw))))

    def _process_sample(self, sample: Dict):
        pil_img: Image.Image = sample["image"]
        # 1aurent/ADE20K provides 'instances' as a list of images
        instance_list: List[Image.Image] = sample.get("instances", [])

        # Build a unified mask from the instances list
        W, H = pil_img.size
        new_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Take up to n_id random instances
        indices = list(range(len(instance_list)))
        if len(indices) > self.n_id:
            indices = random.sample(indices, self.n_id)
        
        for i, idx in enumerate(indices, 1):
            inst_mask = np.array(instance_list[idx].convert("L"))
            new_mask[inst_mask > 0] = i
        
        # ── Resize ────────────────────────────────────────────────────────
        img_raw = TF.to_tensor(
            pil_img.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        )
        mask_t = torch.from_numpy(
            np.array(
                Image.fromarray(new_mask).resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )
            )
        ).long()

        # ── Reference ─────────────────────────────────────────────────────
        ref_img_t  = _normalize(img_raw)
        ref_mask_t = mask_t

        # ── Queries ───────────────────────────────────────────────────────
        query_imgs:  List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            q_img, q_mask = _augment_query_fast(img_raw, mask_t)
            query_imgs.append(q_img)
            query_masks.append(q_mask)

        query_images  = torch.stack(query_imgs)
        query_masks_t = torch.stack(query_masks)

        # ── Presence ──────────────────────────────────────────────────────
        obj_present = torch.zeros(self.seq_len, self.n_id, dtype=torch.bool)
        for t in range(self.seq_len):
            for k in range(1, self.n_id + 1):
                obj_present[t, k - 1] = (query_masks_t[t] == k).any()

        meta = {
            "video_id": f"ade20k_{sample.get('id', 'unknown')}",
            "seen_obj_ids": list(range(1, self.n_id + 1)),
            "unseen_obj_ids": [],
        }

        return ref_img_t, ref_mask_t, query_images, query_masks_t, obj_present, meta

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = self.raw
        if worker_info is not None and hasattr(data, "__len__"):
            indices = list(range(worker_info.id, len(data), worker_info.num_workers))
            data = data.select(indices)

        for sample in data:
            try:
                yield self._process_sample(sample)
            except Exception:
                continue


def _ade_collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        torch.stack([b[4] for b in batch]),
        [b[5] for b in batch]
    )


class ADE20KPretrainDataModule(L.LightningDataModule):
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
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        if stage in ("fit", None):
            self.train_dataset = ADE20KPretrainDataset(
                img_size=hp.img_size, seq_len=hp.seq_len, n_id=hp.n_id,
                split="train", cache_dir=hp.cache_dir, max_samples=hp.max_samples,
                shuffle=True, streaming=hp.streaming
            )
            self.val_dataset = ADE20KPretrainDataset(
                img_size=hp.img_size, seq_len=hp.seq_len, n_id=hp.n_id,
                split="validation", cache_dir=hp.cache_dir, max_samples=hp.max_samples,
                shuffle=False, streaming=hp.streaming
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, collate_fn=_ade_collate_fn,
            pin_memory=True, persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, collate_fn=_ade_collate_fn,
            pin_memory=True, persistent_workers=self.hparams.num_workers > 0
        )
