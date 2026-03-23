"""BL30K pre-training DataModule.

BL30K contains ~30K synthetic video sequences of rendered foreground objects
on complex backgrounds, with per-frame pixel-accurate binary masks.

It is the standard synthetic pre-training dataset for lightweight VOS models
(TrickVOS, WarpFormer-S, AOTT).  This module loads BL30K clips in the same
batch format as COCOPretrainDataModule so the training pipeline is identical.

Expected data layout
--------------------
    data/BL30K/
      BL30K_part1/
        <seq_id>/
          00000.jpg  (or .png)
          00000_mask.png   (binary 0/1 PNG, or palette PNG where >0 = object)
          00001.jpg
          00001_mask.png
          ...
      BL30K_part2/
        <seq_id>/
          ...

Download: https://henghuiding.github.io/MIVOS/ (BL30K section)

Batch format (identical to COCOPretrainDataModule)
---------------------------------------------------
    (ref_img, ref_mask, query_images, query_masks, obj_present, meta)
    ref_img      : [3, H, W]          float32, ImageNet-normalised
    ref_mask     : [H, W]             int64, values in {0, 1}
    query_images : [T, 3, H, W]       float32
    query_masks  : [T, H, W]          int64
    obj_present  : [T, 1]             bool (BL30K is single-object)
    meta         : dict
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_IMG_EXTS  = {".jpg", ".jpeg", ".png"}
_MASK_SFXS = ["_mask.png", ".png"]   # try _mask.png first, fall back to paletted .png


def _normalize(img: torch.Tensor) -> torch.Tensor:
    return TF.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _resize_pair(
    img: Image.Image,
    mask: Image.Image,
    output_size: int,
) -> Tuple[Image.Image, Image.Image]:
    """Resize shorter side to output_size, preserving aspect ratio."""
    w, h = img.size
    if h < w:
        new_h, new_w = output_size, int(w * output_size / h)
    else:
        new_h, new_w = int(h * output_size / w), output_size
    img  = img.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    return img, mask


def _random_crop_pair(
    img: Image.Image,
    mask: Image.Image,
    crop_size: int,
) -> Tuple[Image.Image, Image.Image]:
    w, h = img.size
    x0 = random.randint(0, max(0, w - crop_size))
    y0 = random.randint(0, max(0, h - crop_size))
    img  = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    mask = mask.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    return img, mask


def _pil_mask_to_binary(mask_pil: Image.Image) -> np.ndarray:
    """Convert any PIL mask to a binary uint8 numpy array (0 = bg, 1 = obj)."""
    arr = np.array(mask_pil)
    if arr.ndim == 3:
        arr = arr[..., 0]   # take first channel if RGB-encoded
    return (arr > 0).astype(np.uint8)


def _find_frames(seq_dir: str) -> List[str]:
    """Return sorted list of frame image paths, excluding mask files."""
    if not os.path.isdir(seq_dir):
        return []
    files = sorted(os.listdir(seq_dir))
    frames = []
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() not in _IMG_EXTS:
            continue
        if name.endswith("_mask"):
            continue
        frames.append(os.path.join(seq_dir, f))
    return frames


def _find_mask(frame_path: str, ann_dir: Optional[str] = None) -> Optional[str]:
    """Try to locate the mask file corresponding to a frame."""
    base_name, _ = os.path.splitext(os.path.basename(frame_path))
    
    # Mode 1: Parallel structure (Annotations/seq_id/frame_id.png)
    if ann_dir:
        seq_id = os.path.basename(os.path.dirname(frame_path))
        candidate = os.path.join(ann_dir, seq_id, base_name + ".png")
        if os.path.isfile(candidate):
            return candidate

    # Mode 2: Inline structure (data_dir/seq_id/frame_id_mask.png)
    base = os.path.splitext(frame_path)[0]
    for sfx in _MASK_SFXS:
        candidate = base + sfx
        if os.path.isfile(candidate):
            return candidate
    
    # Fallback to _mask.png
    candidate = base + "_mask.png"
    if os.path.isfile(candidate):
        return candidate
        
    return None


class BL30KDataset(Dataset):
    """Map-style dataset over BL30K synthetic video sequences.

    Each item is a (reference_frame, clip) pair sampled from a random sequence.
    The reference frame is always the first frame of the clip; the query frames
    are sampled with a random stride (max_gap) after it.
    """

    def __init__(
        self,
        data_dir: str,
        output_size: int = 480,
        clip_len: int = 6,
        max_gap: int = 5,
        split: str = "train",
        val_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.clip_len    = clip_len
        self.max_gap     = max_gap

        self.img_dir = None
        self.ann_dir = None
        all_seqs: List[str] = []

        # Try Layout A: Standard split (JPEGImages/seq_id, Annotations/seq_id)
        if os.path.isdir(os.path.join(data_dir, "BL30K", "JPEGImages")):
            self.img_dir = os.path.join(data_dir, "BL30K", "JPEGImages")
            self.ann_dir = os.path.join(data_dir, "BL30K", "Annotations")
            for seq in sorted(os.listdir(self.img_dir)):
                seq_path = os.path.join(self.img_dir, seq)
                if os.path.isdir(seq_path):
                    all_seqs.append(seq_path)
        
        # Try Layout B: Part-based flat structure (BL30K_part1/seq_id/frame_id.jpg + mask.png)
        if not all_seqs:
            for part in ["BL30K_part1", "BL30K_part2"]:
                part_dir = os.path.join(data_dir, part)
                if not os.path.isdir(part_dir):
                    continue
                for seq in sorted(os.listdir(part_dir)):
                    seq_dir = os.path.join(part_dir, seq)
                    if os.path.isdir(seq_dir):
                        all_seqs.append(seq_dir)

        if not all_seqs:
            raise FileNotFoundError(
                f"No BL30K sequences found under {data_dir}. "
                "Checked for BL30K/JPEGImages/ and BL30K_part1/ subdirectories."
            )

        rng = random.Random(0)
        rng.shuffle(all_seqs)
        n_val = max(1, int(len(all_seqs) * val_fraction))

        if split in ("train", "fit"):
            self.sequences = all_seqs[n_val:]
        else:
            self.sequences = all_seqs[:n_val]

    def __len__(self) -> int:
        return len(self.sequences)

    def _load_frame(
        self, frame_path: str
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception:
            return None, None
        
        mask_path = _find_mask(frame_path, ann_dir=self.ann_dir)
        if mask_path is None:
            return None, None
        try:
            mask = Image.open(mask_path)
        except Exception:
            return None, None
        return img, mask

    def _build_sample(self, idx: int):
        seq_dir = self.sequences[idx]
        frames  = _find_frames(seq_dir)

        if len(frames) < self.clip_len:
            raise RuntimeError(f"Sequence too short: {seq_dir}")

        # Sample reference index and clip indices with bounded stride
        max_start = len(frames) - self.clip_len * self.max_gap
        max_start = max(0, max_start)
        ref_idx   = random.randint(0, max_start)

        gap = random.randint(1, self.max_gap)
        indices = [ref_idx + gap * t for t in range(self.clip_len)]
        indices = [min(i, len(frames) - 1) for i in indices]

        # Load reference frame
        ref_img_pil, ref_mask_pil = self._load_frame(frames[ref_idx])
        if ref_img_pil is None:
            raise RuntimeError(f"Could not load reference frame: {frames[ref_idx]}")

        # Resize to output_size (shorter side)
        ref_img_pil, ref_mask_pil = _resize_pair(ref_img_pil, ref_mask_pil, self.output_size)

        # Random crop — use same crop window for all frames to maintain coherence
        w, h = ref_img_pil.size
        x0 = random.randint(0, max(0, w - self.output_size))
        y0 = random.randint(0, max(0, h - self.output_size))
        crop_box = (x0, y0, x0 + self.output_size, y0 + self.output_size)

        ref_img_pil  = ref_img_pil.crop(crop_box)
        ref_mask_pil = ref_mask_pil.crop(crop_box)

        # Random horizontal flip (consistent across clip)
        do_flip = random.random() < 0.5

        if do_flip:
            ref_img_pil  = TF.hflip(ref_img_pil)
            ref_mask_pil = TF.hflip(ref_mask_pil)

        ref_img  = _normalize(TF.to_tensor(ref_img_pil))
        ref_mask = torch.from_numpy(_pil_mask_to_binary(ref_mask_pil)).long()

        query_imgs : List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []

        for fi in indices:
            q_img_pil, q_mask_pil = self._load_frame(frames[fi])
            if q_img_pil is None:
                # Fall back to reference frame on load failure
                q_img_pil, q_mask_pil = self._load_frame(frames[ref_idx])

            q_img_pil, q_mask_pil = _resize_pair(q_img_pil, q_mask_pil, self.output_size)
            q_img_pil  = q_img_pil.crop(crop_box)
            q_mask_pil = q_mask_pil.crop(crop_box)

            if do_flip:
                q_img_pil  = TF.hflip(q_img_pil)
                q_mask_pil = TF.hflip(q_mask_pil)

            # Per-frame colour jitter (mild, independent)
            if random.random() < 0.8:
                brightness = random.uniform(0.8, 1.2)
                contrast   = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                hue        = random.uniform(-0.05, 0.05)
                q_img_pil = TF.adjust_brightness(q_img_pil, brightness)
                q_img_pil = TF.adjust_contrast(q_img_pil, contrast)
                q_img_pil = TF.adjust_saturation(q_img_pil, saturation)
                q_img_pil = TF.adjust_hue(q_img_pil, hue)

            query_imgs.append(_normalize(TF.to_tensor(q_img_pil)))
            query_masks.append(
                torch.from_numpy(_pil_mask_to_binary(q_mask_pil)).long()
            )

        query_images  = torch.stack(query_imgs)   # [T, 3, H, W]
        query_masks_t = torch.stack(query_masks)  # [T, H, W]

        # BL30K is single-object (n_id=1)
        n_id = 1
        obj_present = torch.zeros(self.clip_len, n_id, dtype=torch.bool)
        for t in range(self.clip_len):
            obj_present[t, 0] = (query_masks_t[t] > 0).any()

        meta: Dict = {
            "video_id": f"bl30k_{os.path.basename(seq_dir)}",
            "seen_obj_ids": [1],
            "unseen_obj_ids": [],
        }

        return ref_img, ref_mask, query_images, query_masks_t, obj_present, meta

    def __getitem__(self, idx: int):
        for shift in range(8):
            try_idx = (idx + shift) % len(self.sequences)
            try:
                return self._build_sample(try_idx)
            except Exception:
                continue
        raise RuntimeError(f"Failed to build a valid BL30K sample at index {idx}.")


def _bl30k_collate_fn(batch):
    ref_imgs    = torch.stack([b[0] for b in batch])
    ref_masks   = torch.stack([b[1] for b in batch])
    query_imgs  = torch.stack([b[2] for b in batch])
    query_masks = torch.stack([b[3] for b in batch])
    obj_present = torch.stack([b[4] for b in batch])
    metas       = [b[5] for b in batch]
    return ref_imgs, ref_masks, query_imgs, query_masks, obj_present, metas


class BL30KPretrainDataModule(L.LightningDataModule):
    """Lightning DataModule for BL30K synthetic VOS pre-training."""

    def __init__(
        self,
        data_dir: str = "data/BL30K",
        output_size: int = 480,
        clip_len: int = 6,
        max_gap: int = 5,
        batch_size: int = 8,
        num_workers: int = 8,
        val_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[BL30KDataset] = None
        self.val_dataset:   Optional[BL30KDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        if stage in ("fit", None):
            self.train_dataset = BL30KDataset(
                data_dir=hp.data_dir,
                output_size=hp.output_size,
                clip_len=hp.clip_len,
                max_gap=hp.max_gap,
                split="train",
                val_fraction=hp.val_fraction,
            )
            self.val_dataset = BL30KDataset(
                data_dir=hp.data_dir,
                output_size=hp.output_size,
                clip_len=hp.clip_len,
                max_gap=hp.max_gap,
                split="val",
                val_fraction=hp.val_fraction,
            )
        if stage in ("validate",):
            self.val_dataset = BL30KDataset(
                data_dir=hp.data_dir,
                output_size=hp.output_size,
                clip_len=hp.clip_len,
                max_gap=hp.max_gap,
                split="val",
                val_fraction=hp.val_fraction,
            )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=_bl30k_collate_fn,
            pin_memory=True,
            persistent_workers=hp.num_workers > 0,
            prefetch_factor=4 if hp.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)
