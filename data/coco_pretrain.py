"""COCO pre-training DataModule with real instance masks.

This module follows the AOT-style static-image pre-training recipe but uses the
official COCO instance annotations (polygon/RLE -> pixel masks) instead of
bbox-filled rectangles.

Expected data layout
--------------------
    data/coco/
      annotations/
        instances_train2017.json
        instances_val2017.json
      train2017/
        *.jpg
      val2017/
        *.jpg

Batch format mirrors ``MultiObjectVOSDataModule``:
``(ref_img, ref_mask, query_images, query_masks, obj_present, meta)``.
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

try:
    from pycocotools.coco import COCO

    _PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    COCO = None
    _PYCOCOTOOLS_AVAILABLE = False


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize(img: torch.Tensor) -> torch.Tensor:
    """Normalise a [3, H, W] tensor with ImageNet statistics."""
    return TF.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _augment_query_fast(
    img_raw: torch.Tensor,
    mask_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-based query augmentation applied consistently to image and mask."""
    _, h, w = img_raw.shape

    angle = random.uniform(-10, 10)
    translate = [
        int(w * random.uniform(-0.05, 0.05)),
        int(h * random.uniform(-0.05, 0.05)),
    ]
    scale = random.uniform(0.95, 1.05)
    flip = random.random() < 0.5

    img = TF.affine(
        img_raw,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    mask = TF.affine(
        mask_t.unsqueeze(0).float(),
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        interpolation=TF.InterpolationMode.NEAREST,
    ).squeeze(0).round().long()

    if flip:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
    img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
    img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
    img = img.clamp(0.0, 1.0)

    if random.random() < 0.5:
        h_e = random.randint(h // 8, h // 4)
        w_e = random.randint(w // 8, w // 4)
        y0 = random.randint(0, h - h_e)
        x0 = random.randint(0, w - w_e)
        img = img.clone()
        img[:, y0 : y0 + h_e, x0 : x0 + w_e] = 0.0

    return _normalize(img), mask


def _resolve_coco_paths(data_dir: str, split: str) -> tuple[str, str]:
    """Resolve annotation and image paths for COCO train/val splits."""
    split_key = "train" if split in {"train", "training"} else "val"

    ann_candidates = [
        os.path.join(data_dir, "annotations", f"instances_{split_key}2017.json"),
        os.path.join(data_dir, "annotations", f"instances_{split_key}.json"),
    ]
    img_candidates = [
        os.path.join(data_dir, f"{split_key}2017"),
        os.path.join(data_dir, split_key),
    ]

    ann_path = next((p for p in ann_candidates if os.path.isfile(p)), "")
    img_dir = next((p for p in img_candidates if os.path.isdir(p)), "")

    if not ann_path:
        raise FileNotFoundError(
            f"COCO annotation file not found for split '{split_key}'. Checked: {ann_candidates}"
        )
    if not img_dir:
        raise FileNotFoundError(
            f"COCO image directory not found for split '{split_key}'. Checked: {img_candidates}"
        )

    return ann_path, img_dir


def _valid_annotations(anns: List[Dict]) -> List[Dict]:
    """Return valid instance annotations for mask decoding."""
    valid = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        if float(ann.get("area", 0.0)) <= 1.0:
            continue
        if "segmentation" not in ann or ann["segmentation"] is None:
            continue
        valid.append(ann)
    return valid


class COCOPretrainDataset(Dataset):
    """Map-style COCO 2017 dataset for VOS pre-training with real masks."""

    def __init__(
        self,
        img_size: int = 448,
        seq_len: int = 3,
        n_id: int = 10,
        split: str = "train",
        data_dir: str = "data/coco",
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()

        if not _PYCOCOTOOLS_AVAILABLE:
            raise ImportError(
                "pycocotools is required for COCO instance masks. "
                "Install with: pip install pycocotools"
            )

        if cache_dir is not None:
            warnings.warn(
                "cache_dir is ignored in COCO real-mask mode (local files are used).",
                stacklevel=2,
            )
        if streaming:
            warnings.warn(
                "streaming=True is ignored in COCO real-mask mode.",
                stacklevel=2,
            )

        self.img_size = img_size
        self.seq_len = seq_len
        self.n_id = n_id
        self.split = split
        self.data_dir = data_dir

        ann_path, self.img_dir = _resolve_coco_paths(data_dir, split)
        self.coco = COCO(ann_path)

        image_ids: List[int] = []
        for img_id in self.coco.getImgIds():
            anns = self.coco.imgToAnns.get(img_id, [])
            if len(_valid_annotations(anns)) > 0:
                image_ids.append(img_id)

        if shuffle:
            random.Random(42).shuffle(image_ids)

        if max_samples is not None:
            image_ids = image_ids[: max(0, int(max_samples))]

        if len(image_ids) == 0:
            raise RuntimeError(
                "No valid COCO samples found. "
                "Check annotation paths and that instance segmentations are available."
            )

        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def _decode_instance_mask(self, img_id: int, h: int, w: int) -> np.ndarray:
        """Decode COCO instance annotations into an ID mask [H, W]."""
        anns_all = self.coco.imgToAnns.get(img_id, [])
        anns = _valid_annotations(anns_all)

        # Keep up to n_id largest instances for stable supervision.
        anns = sorted(anns, key=lambda x: float(x.get("area", 0.0)), reverse=True)[: self.n_id]

        mask = np.zeros((h, w), dtype=np.uint8)
        obj_id = 1
        for ann in anns:
            try:
                inst_mask = self.coco.annToMask(ann)
            except Exception:
                continue
            if inst_mask is None or inst_mask.shape != (h, w):
                continue
            if np.max(inst_mask) == 0:
                continue

            mask[inst_mask.astype(bool)] = obj_id
            obj_id += 1
            if obj_id > self.n_id:
                break

        return mask

    def _build_sample(self, idx: int):
        img_id = self.image_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, info["file_name"])

        pil_img = Image.open(img_path).convert("RGB")
        h, w = pil_img.height, pil_img.width

        mask_np = self._decode_instance_mask(img_id, h, w)
        if int(mask_np.max()) == 0:
            raise RuntimeError(f"Decoded empty mask for image_id={img_id}")

        img_raw = TF.to_tensor(
            pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        )
        mask_t = torch.from_numpy(
            np.array(
                Image.fromarray(mask_np).resize((self.img_size, self.img_size), Image.NEAREST)
            )
        ).long()

        ref_img_t = _normalize(img_raw)
        ref_mask_t = mask_t

        query_imgs: List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            q_img, q_mask = _augment_query_fast(img_raw, mask_t)
            query_imgs.append(q_img)
            query_masks.append(q_mask)

        query_images = torch.stack(query_imgs)
        query_masks_t = torch.stack(query_masks)

        obj_present = torch.zeros(self.seq_len, self.n_id, dtype=torch.bool)
        for t in range(self.seq_len):
            for k in range(1, self.n_id + 1):
                obj_present[t, k - 1] = (query_masks_t[t] == k).any()

        seen_ids = sorted(int(v) for v in torch.unique(ref_mask_t).tolist() if int(v) > 0)
        meta = {
            "video_id": f"coco_{img_id}",
            "seen_obj_ids": seen_ids,
            "unseen_obj_ids": [],
        }

        return ref_img_t, ref_mask_t, query_images, query_masks_t, obj_present, meta

    def __getitem__(self, idx: int):
        # Rare decode failures are handled by trying nearby samples.
        for shift in range(8):
            try_idx = (idx + shift) % len(self.image_ids)
            try:
                return self._build_sample(try_idx)
            except Exception:
                continue
        raise RuntimeError(f"Failed to build a valid COCO sample at index {idx}.")


def _coco_collate_fn(batch):
    ref_imgs = torch.stack([b[0] for b in batch])
    ref_masks = torch.stack([b[1] for b in batch])
    query_imgs = torch.stack([b[2] for b in batch])
    query_masks = torch.stack([b[3] for b in batch])
    obj_present = torch.stack([b[4] for b in batch])
    metas = [b[5] for b in batch]
    return ref_imgs, ref_masks, query_imgs, query_masks, obj_present, metas


class COCOPretrainDataModule(L.LightningDataModule):
    """Lightning DataModule for real-mask COCO static pre-training."""

    def __init__(
        self,
        data_dir: str = "data/coco",
        img_size: int = 448,
        seq_len: int = 3,
        n_id: int = 10,
        batch_size: int = 4,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[COCOPretrainDataset] = None
        self.val_dataset: Optional[COCOPretrainDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        if stage in ("fit", None):
            self.train_dataset = COCOPretrainDataset(
                data_dir=hp.data_dir,
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="train",
                max_samples=hp.max_samples,
                shuffle=True,
                cache_dir=hp.cache_dir,
                streaming=hp.streaming,
            )
            self.val_dataset = COCOPretrainDataset(
                data_dir=hp.data_dir,
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="validation",
                max_samples=hp.max_samples,
                shuffle=False,
                cache_dir=hp.cache_dir,
                streaming=hp.streaming,
            )

        if stage in ("validate",):
            self.val_dataset = COCOPretrainDataset(
                data_dir=hp.data_dir,
                img_size=hp.img_size,
                seq_len=hp.seq_len,
                n_id=hp.n_id,
                split="validation",
                max_samples=hp.max_samples,
                shuffle=False,
                cache_dir=hp.cache_dir,
                streaming=hp.streaming,
            )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=shuffle,
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
