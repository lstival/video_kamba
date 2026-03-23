"""LVOS V2 fine-tuning DataModule.

LVOS (Long-term Video Object Segmentation) V2 contains 720 videos averaging
~408 frames (~68 seconds at 6 fps), explicitly designed to stress long-range
temporal propagation — the scenario our DTSM slow SSM path targets.

Dataset layout (after extraction)
----------------------------------
    data/LVOS/
      train/
        JPEGImages/<video_id>/<frame_id:08d>.jpg
        Annotations/<video_id>/<frame_id:08d>.png   (palette PNG, 0=bg, 1..N=obj)
        meta_expressions.json
      valid/
        JPEGImages/<video_id>/<frame_id:08d>.jpg
        Annotations/<video_id>/<frame_id:08d>.png
        meta_expressions.json

Batch format (identical to VOSDataModule / BL30KPretrainDataModule)
--------------------------------------------------------------------
    ref_img      : [B, 3, H, W]         float32, ImageNet-normalised
    ref_mask     : [B, H, W]            int64, 0=bg 1..n_id=objects
    query_images : [B, T, 3, H, W]      float32
    query_masks  : [B, T, H, W]         int64
    obj_present  : [B, T, n_id]         bool
    meta         : list[dict]

Training strategy
-----------------
LVOS sequences are far too long to load in full during training.  We sample
random clips of ``clip_len`` frames with a random stride ``max_gap`` (up to 10)
giving an effective temporal span of up to 120 frames per clip — sufficient for
the DTSM slow SSM path to see meaningful long-range signal.

For validation we take the first ``max_val_frames`` frames of each sequence as
a contiguous clip, giving a reproducible snapshot that tracks propagation quality
without requiring a full-sequence evaluation pass (which needs the LVOS server).

References
----------
* Hong et al., "LVOS: A Long-term Video Object Segmentation Benchmark", ICCV 2023.
* Evaluation toolkit: https://github.com/LingyiHongfd/lvos-evaluation
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

LOGGER = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_VOID_LABEL    = 255


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(img: torch.Tensor) -> torch.Tensor:
    return TF.normalize(img, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, (size, size))
    return _normalize(TF.to_tensor(img))


def _load_mask(path: str, size: int) -> torch.Tensor:
    """Load a palette-indexed PNG; returns an int64 tensor [H, W] with object IDs."""
    mask = Image.open(path)
    mask = TF.resize(mask, (size, size), interpolation=TF.InterpolationMode.NEAREST)
    arr = np.array(mask, dtype=np.int64)
    # Void pixels (255) are mapped to background
    arr[arr == _VOID_LABEL] = 0
    return torch.from_numpy(arr)


def _consistent_flip(
    frames: List[torch.Tensor],
    masks: List[torch.Tensor],
    p: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if random.random() < p:
        frames = [TF.hflip(f) for f in frames]
        masks  = [TF.hflip(m.unsqueeze(0)).squeeze(0) for m in masks]
    return frames, masks


def _build_obj_present(masks: List[torch.Tensor], n_id: int) -> torch.Tensor:
    """Boolean [T, n_id] — True if object k+1 is visible in frame t."""
    T = len(masks)
    obj_present = torch.zeros(T, n_id, dtype=torch.bool)
    for t, mask in enumerate(masks):
        for k in range(1, n_id + 1):
            obj_present[t, k - 1] = (mask == k).any()
    return obj_present


def _read_meta(split_dir: str) -> Optional[dict]:
    for name in ("meta_expressions.json", "meta.json"):
        path = os.path.join(split_dir, name)
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LVOSClipDataset(Dataset):
    """Clip-sampling Dataset over one LVOS split (train or valid).

    Each ``__getitem__`` returns a (ref, clip) pair sampled from a random
    sequence.  The reference frame is always the first annotated frame.  The
    query clip is sampled with a random stride within ``[1, max_gap]``.

    Because LVOS sequences average 408 frames, ``max_gap=10`` gives an
    effective temporal span of up to 120 frames per clip of length 12 — enough
    context for the DTSM slow SSM path to learn meaningful long-range dynamics.

    Args:
        data_dir:     Path to the LVOS root (contains ``train/`` and ``valid/``).
        split:        One of ``"train"`` or ``"valid"``.
        clip_len:     Number of query frames per sample.
        max_gap:      Maximum frame stride between query frames.
        output_size:  Spatial resolution of output tensors.
        n_id:         Maximum simultaneous objects per clip (identity-bank size).
        max_val_frames: If split == "valid", truncate each sequence to this many
                      frames to keep validation fast.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        clip_len: int = 12,
        max_gap: int = 10,
        output_size: int = 480,
        n_id: int = 10,
        max_val_frames: int = 24,
    ) -> None:
        super().__init__()
        assert split in ("train", "valid"), f"Unknown split: {split}"

        self.split          = split
        self.clip_len       = clip_len
        self.max_gap        = max_gap
        self.output_size    = output_size
        self.n_id           = n_id
        self.max_val_frames = max_val_frames

        split_dir   = os.path.join(data_dir, split)
        img_root    = os.path.join(split_dir, "JPEGImages")
        ann_root    = os.path.join(split_dir, "Annotations")

        if not os.path.isdir(img_root):
            raise FileNotFoundError(
                f"LVOS {split} images not found at {img_root}\n"
                "Run scripts/slurm_setup_lvos_lustre.sh to download the dataset."
            )
        if not os.path.isdir(ann_root):
            raise FileNotFoundError(
                f"LVOS {split} annotations not found at {ann_root}\n"
                "Run scripts/slurm_setup_lvos_lustre.sh to download the dataset."
            )

        # Discover sequences: must have images AND annotations
        all_video_ids = sorted(os.listdir(img_root))
        sequences: List[Dict] = []
        for vid in all_video_ids:
            img_dir = os.path.join(img_root, vid)
            ann_dir = os.path.join(ann_root, vid)
            if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
                continue
            img_frames  = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
            ann_frames  = sorted(f for f in os.listdir(ann_dir) if f.endswith(".png"))
            if len(img_frames) < 2 or len(ann_frames) < 1:
                continue
            # Only keep frames that have an annotation
            ann_stems = {os.path.splitext(f)[0] for f in ann_frames}
            valid_frames = [f for f in img_frames if os.path.splitext(f)[0] in ann_stems]
            if len(valid_frames) < 2:
                continue
            sequences.append({
                "video_id":  vid,
                "img_dir":   img_dir,
                "ann_dir":   ann_dir,
                "frames":    img_frames,      # all image frames (some may lack annotation)
                "ann_frames": valid_frames,   # frames with annotation
            })

        if not sequences:
            raise RuntimeError(
                f"No valid LVOS sequences found in {split_dir}. "
                "Check the dataset download and directory structure."
            )

        self.sequences = sequences
        LOGGER.info(
            "[LVOSClipDataset] split=%s  seqs=%d  clip_len=%d  max_gap=%d",
            split, len(sequences), clip_len, max_gap,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def _sample_indices(self, n_frames: int) -> Tuple[int, List[int]]:
        """Sample a reference index and clip indices respecting max_gap."""
        gap = random.randint(1, self.max_gap)
        span = self.clip_len * gap
        if span >= n_frames:
            # Sequence too short for the desired gap — fall back to gap=1
            gap = 1
            span = self.clip_len

        max_ref = max(0, n_frames - span - 1)
        ref_idx = random.randint(0, max_ref)
        query_indices = [ref_idx + gap * (t + 1) for t in range(self.clip_len)]
        query_indices = [min(q, n_frames - 1) for q in query_indices]
        return ref_idx, query_indices

    def _sample_val_indices(self, n_frames: int) -> Tuple[int, List[int]]:
        """For validation: take the first max_val_frames frames contiguously."""
        total = min(n_frames, self.max_val_frames)
        ref_idx = 0
        query_indices = list(range(1, min(self.clip_len + 1, total)))
        # Pad with last frame if sequence shorter than clip_len
        while len(query_indices) < self.clip_len:
            query_indices.append(query_indices[-1])
        return ref_idx, query_indices[:self.clip_len]

    def _get_ann_path(self, seq: Dict, frame_name: str) -> Optional[str]:
        """Return annotation path for a frame, or None if not annotated."""
        stem = os.path.splitext(frame_name)[0]
        path = os.path.join(seq["ann_dir"], stem + ".png")
        return path if os.path.isfile(path) else None

    def _build_sample(self, idx: int) -> tuple:
        seq = self.sequences[idx]
        frames  = seq["frames"]          # all image frames
        n_frames = len(frames)

        if self.split == "train":
            ref_frame_idx, query_frame_idxs = self._sample_indices(n_frames)
        else:
            ref_frame_idx, query_frame_idxs = self._sample_val_indices(n_frames)

        # Load reference frame and mask
        ref_img_path = os.path.join(seq["img_dir"], frames[ref_frame_idx])
        ref_ann_path = self._get_ann_path(seq, frames[ref_frame_idx])

        ref_img  = _load_image(ref_img_path, self.output_size)
        ref_mask = (
            _load_mask(ref_ann_path, self.output_size)
            if ref_ann_path else
            torch.zeros(self.output_size, self.output_size, dtype=torch.long)
        )

        # Load query frames
        query_imgs : List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        for q_idx in query_frame_idxs:
            q_img_path = os.path.join(seq["img_dir"], frames[q_idx])
            q_ann_path = self._get_ann_path(seq, frames[q_idx])

            q_img  = _load_image(q_img_path, self.output_size)
            q_mask = (
                _load_mask(q_ann_path, self.output_size)
                if q_ann_path else
                torch.zeros(self.output_size, self.output_size, dtype=torch.long)
            )
            query_imgs.append(q_img)
            query_masks.append(q_mask)

        # Consistent horizontal flip across all frames during training
        if self.split == "train":
            all_imgs  = [ref_img]  + query_imgs
            all_masks = [ref_mask] + query_masks
            all_imgs, all_masks = _consistent_flip(all_imgs, all_masks, p=0.5)
            ref_img, ref_mask = all_imgs[0], all_masks[0]
            query_imgs, query_masks = all_imgs[1:], all_masks[1:]

        # Cap object IDs to n_id (safety)
        ref_mask = ref_mask.clamp(0, self.n_id)
        query_masks = [m.clamp(0, self.n_id) for m in query_masks]

        query_images  = torch.stack(query_imgs)   # [T, 3, H, W]
        query_masks_t = torch.stack(query_masks)  # [T, H, W]
        obj_present   = _build_obj_present(query_masks, self.n_id)  # [T, n_id]

        meta: Dict = {
            "video_id":      seq["video_id"],
            "seen_obj_ids":  list(range(1, self.n_id + 1)),
            "unseen_obj_ids": [],
            "split":         self.split,
        }

        return ref_img, ref_mask, query_images, query_masks_t, obj_present, meta

    def __getitem__(self, idx: int) -> tuple:
        for shift in range(8):
            try_idx = (idx + shift) % len(self.sequences)
            try:
                return self._build_sample(try_idx)
            except Exception as exc:
                LOGGER.debug("[LVOSClipDataset] skipping idx=%d shift=%d: %s", idx, shift, exc)
        raise RuntimeError(f"[LVOSClipDataset] Failed to load a valid sample at idx={idx}")


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def _lvos_collate_fn(batch: list) -> tuple:
    ref_imgs    = torch.stack([b[0] for b in batch])
    ref_masks   = torch.stack([b[1] for b in batch])
    query_imgs  = torch.stack([b[2] for b in batch])
    query_masks = torch.stack([b[3] for b in batch])
    obj_present = torch.stack([b[4] for b in batch])
    metas       = [b[5] for b in batch]
    return ref_imgs, ref_masks, query_imgs, query_masks, obj_present, metas


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class LVOSFinetuneDataModule(L.LightningDataModule):
    """Lightning DataModule for LVOS V2 fine-tuning.

    Phase 3 of the curriculum: adapts the DTSM model (pre-trained on BL30K
    synthetic sequences) to the distribution of real long-range VOS videos.

    Args:
        data_dir:       Path to the LVOS root (must contain ``train/`` and
                        ``valid/`` subdirectories with JPEGImages + Annotations).
        output_size:    Spatial resolution (height = width) of all output tensors.
        clip_len:       Number of query frames per training clip.
        max_gap:        Maximum frame stride between query frames.
                        clip_len × max_gap determines the effective temporal span.
        batch_size:     Training batch size.
        num_workers:    DataLoader workers per split.
        n_id:           Maximum tracked objects per clip (identity-bank size).
        max_val_frames: Validation clips are capped at this many frames for
                        speed (full-sequence LVOS eval is done offline).
    """

    def __init__(
        self,
        data_dir: str = "data/LVOS",
        output_size: int = 480,
        clip_len: int = 12,
        max_gap: int = 10,
        batch_size: int = 2,
        num_workers: int = 8,
        n_id: int = 10,
        max_val_frames: int = 24,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._train_ds: Optional[LVOSClipDataset] = None
        self._val_ds:   Optional[LVOSClipDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        if stage in ("fit", None):
            self._train_ds = LVOSClipDataset(
                data_dir     = hp.data_dir,
                split        = "train",
                clip_len     = hp.clip_len,
                max_gap      = hp.max_gap,
                output_size  = hp.output_size,
                n_id         = hp.n_id,
                max_val_frames=hp.max_val_frames,
            )
            self._val_ds = LVOSClipDataset(
                data_dir     = hp.data_dir,
                split        = "valid",
                clip_len     = hp.clip_len,
                max_gap      = hp.max_gap,
                output_size  = hp.output_size,
                n_id         = hp.n_id,
                max_val_frames=hp.max_val_frames,
            )
            LOGGER.info(
                "[LVOSFinetuneDataModule] train=%d  val=%d  clip_len=%d  max_gap=%d",
                len(self._train_ds), len(self._val_ds), hp.clip_len, hp.max_gap,
            )
        if stage == "validate":
            self._val_ds = LVOSClipDataset(
                data_dir     = hp.data_dir,
                split        = "valid",
                clip_len     = hp.clip_len,
                max_gap      = hp.max_gap,
                output_size  = hp.output_size,
                n_id         = hp.n_id,
                max_val_frames=hp.max_val_frames,
            )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=_lvos_collate_fn,
            pin_memory=True,
            persistent_workers=hp.num_workers > 0,
            prefetch_factor=4 if hp.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)
