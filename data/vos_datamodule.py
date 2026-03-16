"""AOT-protocol VOS DataModule for training & evaluating our SSM-KAN model.

This module adapts the yoxu515/aot-benchmark training/evaluation dataset
protocol so that our VideoMambaSystem can be trained and evaluated with the
same data splits and augmentations used by the SOTA AOT/AOST family.

Training protocol (mirrors AOT's VOSTrain):
    Each sample is a *clip*:
        ref_img:     [3, H, W]   — first annotated frame
        ref_mask:    [H, W]      — integer IDs (0=bg, 1..N=objects, 255=void)
        query_imgs:  [T, 3, H, W]
        query_masks: [T, H, W]   — integer IDs
    Augmentations are applied *identically* across all frames in the clip
    (same crop, same flip), with per-frame colour jitter for appearance variety.

Evaluation protocol (mirrors AOT's evaluator):
    Returns one frame at a time via VOSEvalDataset.  Caller accumulates
    predictions and computes J&F offline.

Supported datasets
    DAVIS 2017   — train (60 seqs) and val (30 seqs)
    YouTube-VOS  — train (3,471 seqs) — optional, set ytv_root in config

References:
    Yang et al., "Associating Objects with Transformers for VOS", NeurIPS 2021.
"""

from __future__ import annotations

import logging
import os
import random
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
import lightning as L

LOGGER = logging.getLogger(__name__)

# ── Normalisation constants (ImageNet / DINOv2) ───────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Palette used by DAVIS / YouTube-VOS annotation PNGs
_VOID_LABEL = 255


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(path: str) -> np.ndarray:
    """Load a palette-indexed PNG as a uint8 array of integer object IDs."""
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)


def _resize_img(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio so that the short side == size."""
    w, h = img.size
    if h < w:
        new_h, new_w = size, int(size * w / h)
    else:
        new_h, new_w = int(size * h / w), size
    return img.resize((new_w, new_h), Image.BILINEAR)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask, mode="P")
    w, h = pil.size
    if h < w:
        new_h, new_w = size, int(size * w / h)
    else:
        new_h, new_w = int(size * h / w), size
    return np.array(
        pil.resize((new_w, new_h), Image.NEAREST), dtype=np.uint8
    )


def _random_crop_params(img_h: int, img_w: int, crop_h: int, crop_w: int):
    """Return (top, left) for a random crop."""
    top  = random.randint(0, max(0, img_h - crop_h))
    left = random.randint(0, max(0, img_w - crop_w))
    return top, left


def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → normalised float32 tensor [3, H, W]."""
    t = TF.to_tensor(img)                       # [3,H,W] float [0,1]
    t = TF.normalize(t, mean=_MEAN, std=_STD)   # ImageNet normalisation
    return t


def _mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """uint8 mask array → int64 tensor [H, W]."""
    return torch.from_numpy(mask.astype(np.int64))


# ══════════════════════════════════════════════════════════════════════════════
# Coordinated augmentation — same spatial params for all frames in a clip
# ══════════════════════════════════════════════════════════════════════════════

class ClipAugmentor:
    """Apply consistent spatial transforms + independent colour jitter.

    All *spatial* operations (crop, flip, resize) use the *same* random
    parameters for every frame and its corresponding mask in the clip.
    Colour jitter is applied independently per frame so the model learns
    appearance-invariant matching (matching AOT's augmentation strategy).
    """

    def __init__(
        self,
        output_size: int = 224,
        min_scale: float = 0.7,
        max_scale: float = 1.3,
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
    ) -> None:
        self.output_size  = output_size
        self.min_scale    = min_scale
        self.max_scale    = max_scale
        self.flip_prob    = flip_prob
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
        )
        self.color_jitter_prob = color_jitter_prob

    def __call__(
        self,
        images: List[Image.Image],
        masks: List[np.ndarray],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert len(images) == len(masks), "images and masks must have same length"

        # ── 1. Scale ──────────────────────────────────────────────────────────
        scale = random.uniform(self.min_scale, self.max_scale)
        scaled_size = max(self.output_size, int(self.output_size * scale))

        images_pil = [_resize_img(img, scaled_size) for img in images]
        masks_np   = [_resize_mask(m,   scaled_size) for m in masks]

        # ── 2. Common random crop ─────────────────────────────────────────────
        h, w    = masks_np[0].shape
        crop_h  = min(self.output_size, h)
        crop_w  = min(self.output_size, w)
        top, left = _random_crop_params(h, w, crop_h, crop_w)

        images_pil = [
            TF.crop(img, top, left, crop_h, crop_w) for img in images_pil
        ]
        masks_np = [
            m[top:top + crop_h, left:left + crop_w] for m in masks_np
        ]

        # Pad to exact output size if the crop was smaller
        if crop_h < self.output_size or crop_w < self.output_size:
            images_pil = [
                TF.pad(img,
                       [0, 0,
                        self.output_size - crop_w,
                        self.output_size - crop_h]) for img in images_pil
            ]
            padded = []
            for m in masks_np:
                pad_m = np.zeros((self.output_size, self.output_size),
                                 dtype=np.uint8)
                pad_m[:crop_h, :crop_w] = m
                padded.append(pad_m)
            masks_np = padded

        # ── 3. Common horizontal flip ─────────────────────────────────────────
        if random.random() < self.flip_prob:
            images_pil = [TF.hflip(img) for img in images_pil]
            masks_np   = [np.fliplr(m).copy() for m in masks_np]

        # ── 4. Per-frame colour jitter (spatial ops finished) ─────────────────
        img_tensors  = []
        mask_tensors = []
        for img, m in zip(images_pil, masks_np):
            if random.random() < self.color_jitter_prob:
                img = self.color_jitter(img)
            img_tensors.append(_img_to_tensor(img))
            mask_tensors.append(_mask_to_tensor(m))

        return img_tensors, mask_tensors


class ValTransform:
    """Simple resize + centre crop for validation — no random ops."""

    def __init__(self, output_size: int = 224) -> None:
        self.output_size = output_size

    def __call__(
        self,
        images: List[Image.Image],
        masks: List[np.ndarray],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        img_tensors  = []
        mask_tensors = []
        for img, m in zip(images, masks):
            img_resized = TF.center_crop(
                TF.resize(img, self.output_size, interpolation=Image.BILINEAR),
                self.output_size,
            )
            # Mask: NEAREST to preserve integer IDs
            m_pil = Image.fromarray(m, mode="P")
            m_pil = TF.center_crop(
                TF.resize(m_pil, self.output_size, interpolation=Image.NEAREST),
                self.output_size,
            )
            img_tensors.append(_img_to_tensor(img_resized))
            mask_tensors.append(_mask_to_tensor(np.array(m_pil, dtype=np.uint8)))
        return img_tensors, mask_tensors


# ══════════════════════════════════════════════════════════════════════════════
# DAVIS 2017 — VOS Training Dataset
# ══════════════════════════════════════════════════════════════════════════════

class DAVISVOSTrain(Dataset):
    """DAVIS 2017 training set following the AOT clip-sampling protocol.

    Each ``__getitem__`` returns a clip of ``clip_len + 1`` frames as::

        {
          "ref_img":     FloatTensor[3, H, W]
          "ref_mask":    LongTensor[H, W]
          "query_imgs":  FloatTensor[T, 3, H, W]
          "query_masks": LongTensor[T, H, W]
          "meta": {"seq": str, "obj_num": int}
        }

    The reference frame is the *first annotated frame* of the sequence.
    Query frames are contiguous frames sampled after the reference.

    Args:
        root:      Path to DAVIS root (contains ``JPEGImages/``, ``Annotations/``).
        clip_len:  Number of query frames per clip (T).  Default 4.
        output_size: Spatial output resolution in pixels.
        resolution: DAVIS resolution tier (``"480p"`` or ``"Full-Resolution"``).
        max_gap:   Maximum temporal stride between consecutive query frames.
        augmentor: Augmentation callable; defaults to :class:`ClipAugmentor`.
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 4,
        output_size: int = 224,
        resolution: str = "480p",
        max_gap: int = 3,
        augmentor: Optional[Callable] = None,
    ) -> None:
        self.root        = Path(root)
        self.clip_len    = clip_len
        self.output_size = output_size
        self.max_gap     = max_gap

        self.img_dir  = self.root / "JPEGImages"  / resolution
        self.ann_dir  = self.root / "Annotations" / resolution
        self.split_f  = self.root / "ImageSets" / "2017" / "train.txt"

        if not self.split_f.exists():
            raise FileNotFoundError(f"DAVIS train split not found: {self.split_f}")

        with open(self.split_f) as f:
            seqs = [l.strip() for l in f if l.strip()]

        # Build per-sequence frame lists
        self._seqs: Dict[str, List[str]] = {}
        for seq in seqs:
            frames = sorted((self.img_dir / seq).glob("*.jpg"))
            if len(frames) >= 2:
                self._seqs[seq] = [fr.name for fr in frames]

        # Expand into individual (seq, ref_idx, start_idx) clips
        self._clips: List[Tuple[str, int, int]] = []
        for seq, framenames in self._seqs.items():
            n = len(framenames)
            # Reference is always index 0 (first annotated frame)
            for start in range(1, n - clip_len + 1):
                self._clips.append((seq, 0, start))

        self.augmentor = augmentor or ClipAugmentor(output_size=output_size)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _load_frame(self, seq: str, name: str) -> Image.Image:
        return _load_image(str(self.img_dir / seq / name))

    def _load_ann(self, seq: str, name: str) -> np.ndarray:
        ann_name = name.replace(".jpg", ".png")
        ann_path = self.ann_dir / seq / ann_name
        if ann_path.exists():
            return _load_mask(str(ann_path))
        # Return all-zero mask if annotation missing (shouldn't happen for DAVIS)
        return np.zeros((1, 1), dtype=np.uint8)

    def _count_objects(self, mask: np.ndarray) -> int:
        ids = set(np.unique(mask).tolist()) - {0, int(_VOID_LABEL)}
        return len(ids)

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Dict:
        seq, ref_idx, start_idx = self._clips[idx]
        frames = self._seqs[seq]

        # Gather frame names for this clip
        ref_name   = frames[ref_idx]
        query_names = []
        for i in range(self.clip_len):
            # Allow random intra-clip gap up to max_gap
            gap = random.randint(1, self.max_gap)
            qidx = min(start_idx + i * gap, len(frames) - 1)
            query_names.append(frames[qidx])

        # Load images & masks
        all_imgs  = [self._load_frame(seq, ref_name)] + \
                    [self._load_frame(seq, qn) for qn in query_names]
        all_masks = [self._load_ann(seq, ref_name)] + \
                    [self._load_ann(seq, qn) for qn in query_names]

        # Apply coordinated augmentation
        img_tensors, mask_tensors = self.augmentor(all_imgs, all_masks)

        ref_img  = img_tensors[0]               # [3, H, W]
        ref_mask = mask_tensors[0]              # [H, W]
        query_imgs  = torch.stack(img_tensors[1:],  dim=0)   # [T, 3, H, W]
        query_masks = torch.stack(mask_tensors[1:], dim=0)   # [T, H, W]

        return {
            "ref_img":     ref_img,
            "ref_mask":    ref_mask,
            "query_imgs":  query_imgs,
            "query_masks": query_masks,
            "meta": {
                "seq":     seq,
                "obj_num": self._count_objects(all_masks[0]),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# DAVIS 2017 — VOS Evaluation Dataset (sequence-level, AOT-compatible)
# ══════════════════════════════════════════════════════════════════════════════

class DAVISVOSEval(Dataset):
    """DAVIS 2017 validation set for sequence-level evaluation.

    Returns frames one-by-one for a *single* sequence.  Instantiate a new
    object per sequence.  Matches the ``VOSTest`` protocol from the
    aot-benchmark so our evaluation script can directly call the same inference
    loop.

    ``__getitem__`` returns::

        {
          "img":        FloatTensor[3, H, W]   — normalised RGB
          "mask":       LongTensor[H, W] | None — GT mask (only if annotated)
          "has_mask":   bool
          "frame_name": str
          "obj_num":    int
          "obj_idx":    List[int]              — object IDs present in this seq
        }
    """

    def __init__(
        self,
        root: str,
        seq_name: str,
        resolution: str = "480p",
        output_size: Optional[int] = None,
    ) -> None:
        self.root       = Path(root)
        self.seq_name   = seq_name
        self.output_size = output_size  # None = use original resolution

        self.img_dir  = self.root / "JPEGImages"  / resolution / seq_name
        self.ann_dir  = self.root / "Annotations" / resolution / seq_name

        frames = sorted(self.img_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(
                f"No frames found for sequence {seq_name} at {self.img_dir}"
            )
        self.frame_names = [fr.name for fr in frames]
        ann_names = {p.stem for p in self.ann_dir.glob("*.png")}
        self.has_ann     = [fr.stem in ann_names for fr in frames]

        # Determine all object IDs by scanning the first available annotation
        first_ann_path = self.ann_dir / (frames[0].stem + ".png")
        if first_ann_path.exists():
            first_mask = _load_mask(str(first_ann_path))
            obj_ids = sorted(
                set(np.unique(first_mask).tolist()) - {0, int(_VOID_LABEL)}
            )
        else:
            obj_ids = []
        self.obj_idx = [0] + obj_ids
        self.obj_num = len(obj_ids)

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, idx: int) -> Dict:
        name = self.frame_names[idx]
        img  = _load_image(str(self.img_dir / name))

        if self.output_size is not None:
            img = TF.resize(img, self.output_size, interpolation=Image.BILINEAR)

        img_t = _img_to_tensor(img)

        has_ann = self.has_ann[idx]
        if has_ann:
            ann_name = name.replace(".jpg", ".png")
            mask_np  = _load_mask(str(self.ann_dir / ann_name))
            if self.output_size is not None:
                mask_np = _resize_mask(mask_np, self.output_size)
            mask_t = _mask_to_tensor(mask_np)
        else:
            mask_t = None

        return {
            "img":        img_t,
            "mask":       mask_t,
            "has_mask":   has_ann,
            "frame_name": name,
            "obj_num":    self.obj_num,
            "obj_idx":    self.obj_idx,
        }


# ══════════════════════════════════════════════════════════════════════════════
# YouTube-VOS — optional training set (same clip protocol)
# ══════════════════════════════════════════════════════════════════════════════

class YouTubeVOSTrain(Dataset):
    """YouTube-VOS 2019 training set.

    Mirrors DAVISVOSTrain but handles the YouTube-VOS directory structure::

        <root>/
          train/
            JPEGImages/<seq_id>/<frame>.jpg
            Annotations/<seq_id>/<frame>.png
            meta.json

    Annotations are *sparse* (only a subset of frames are annotated).
    Unannotated query frames get a zero mask during training
    (the model ignores 0-label entries in the loss function).
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 4,
        output_size: int = 224,
        max_gap: int = 3,
        augmentor: Optional[Callable] = None,
    ) -> None:
        self.root       = Path(root) / "train"
        self.clip_len   = clip_len
        self.output_size = output_size
        self.max_gap    = max_gap

        self.img_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"

        import json
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"YouTube-VOS meta.json not found at {meta_path}. "
                "Please download the dataset first."
            )
        with open(meta_path) as f:
            meta = json.load(f)["videos"]

        # Only use sequences where JPEGImages directory exists
        self._seqs: Dict[str, List[str]] = {}
        for seq_id in meta:
            seq_img_dir = self.img_dir / seq_id
            if seq_img_dir.exists():
                frames = sorted(seq_img_dir.glob("*.jpg"))
                if len(frames) >= clip_len + 1:
                    self._seqs[seq_id] = [fr.name for fr in frames]

        # Build clips: use first annotated frame as reference
        self._clips: List[Tuple[str, int, int]] = []
        for seq_id, framenames in self._seqs.items():
            n = len(framenames)
            for start in range(1, n - clip_len + 1):
                self._clips.append((seq_id, 0, start))

        self.augmentor = augmentor or ClipAugmentor(output_size=output_size)

    def _has_annotation(self, seq: str, name: str) -> bool:
        ann_name = name.replace(".jpg", ".png")
        return (self.ann_dir / seq / ann_name).exists()

    def _load_ann_or_zero(self, seq: str, name: str,
                          ref_shape: Tuple[int, int]) -> np.ndarray:
        ann_name = name.replace(".jpg", ".png")
        ann_path = self.ann_dir / seq / ann_name
        if ann_path.exists():
            return _load_mask(str(ann_path))
        return np.zeros(ref_shape, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Dict:
        seq, ref_idx, start_idx = self._clips[idx]
        frames = self._seqs[seq]

        ref_name    = frames[ref_idx]
        query_names = []
        for i in range(self.clip_len):
            gap  = random.randint(1, self.max_gap)
            qidx = min(start_idx + i * gap, len(frames) - 1)
            query_names.append(frames[qidx])

        all_imgs  = [_load_image(str(self.img_dir / seq / n))
                     for n in [ref_name] + query_names]
        h0, w0    = np.array(all_imgs[0]).shape[:2]
        all_masks = [self._load_ann_or_zero(seq, ref_name, (h0, w0))] + \
                    [self._load_ann_or_zero(seq, qn,       (h0, w0)) for qn in query_names]

        img_tensors, mask_tensors = self.augmentor(all_imgs, all_masks)

        return {
            "ref_img":     img_tensors[0],
            "ref_mask":    mask_tensors[0],
            "query_imgs":  torch.stack(img_tensors[1:],  dim=0),
            "query_masks": torch.stack(mask_tensors[1:], dim=0),
            "meta": {"seq": seq},
        }


# ══════════════════════════════════════════════════════════════════════════════
# Collate function — handles optional fields
# ══════════════════════════════════════════════════════════════════════════════

def _vos_collate(batch: List[Dict]) -> Tuple:
    """Collate a list of VOS dicts into (ref_img, ref_mask, query_imgs, query_masks, seq_names).

    The 5-tuple format extends the VideoMambaSystem forward signature with sequence
    identifiers used by SpikeDiagnosticsCallback to attribute high-loss batches to
    specific sequences.
    """
    ref_imgs    = torch.stack([b["ref_img"]     for b in batch])   # [B,3,H,W]
    ref_masks   = torch.stack([b["ref_mask"]    for b in batch])   # [B,H,W]
    query_imgs  = torch.stack([b["query_imgs"]  for b in batch])   # [B,T,3,H,W]
    query_masks = torch.stack([b["query_masks"] for b in batch])   # [B,T,H,W]
    seq_names   = [b["meta"]["seq"] for b in batch]                # [B] list[str]
    return ref_imgs, ref_masks, query_imgs, query_masks, seq_names


# ══════════════════════════════════════════════════════════════════════════════
# LightningDataModule
# ══════════════════════════════════════════════════════════════════════════════

class VOSDataModule(L.LightningDataModule):
    """LightningDataModule wrapping the VOS training/validation datasets.

    Designed to plug directly into the existing Hydra training pipeline::

        datamodule:
          _target_: data.vos_datamodule.VOSDataModule
          davis_root: data/DAVIS/DAVIS
          batch_size: 4
          clip_len: 4
          output_size: 224
          num_workers: 4
          ytv_root: null   # set to YouTube-VOS path to mix datasets

    Args:
        davis_root:   Absolute path to the DAVIS root directory.
        batch_size:   Training batch size.
        clip_len:     Number of query frames per training clip (T).
        output_size:  Spatial crop size in pixels (use 448 for higher accuracy).
        num_workers:  DataLoader worker processes.
        resolution:   DAVIS resolution tier — ``"480p"`` or ``"Full-Resolution"``.
        max_gap:      Maximum frame stride within a training clip.
        ytv_root:     Optional path to YouTube-VOS 2019 root.  When set, the
                      training set is a ConcatDataset of DAVIS-train and YTV-train.
        davis_sampling_ratio: Fraction of each epoch's samples drawn from DAVIS
                      when joint training (0 < ratio < 1).  Default 0.25 means
                      25 % DAVIS / 75 % YouTube-VOS, matching the AOT PRE_YTB_DAV
                      balance.  Ignored when ``ytv_root`` is None.
        val_output_size: Validation resolution (defaults to ``output_size``).
    """

    def __init__(
        self,
        davis_root: str,
        batch_size: int = 4,
        clip_len: int = 4,
        output_size: int = 224,
        num_workers: int = 4,
        resolution: str = "480p",
        max_gap: int = 3,
        ytv_root: Optional[str] = None,
        davis_sampling_ratio: float = 0.25,
        val_output_size: Optional[int] = None,
        train_min_scale: float = 0.7,
        train_max_scale: float = 1.3,
        train_flip_prob: float = 0.5,
        train_color_jitter_prob: float = 0.8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.davis_root           = Path(davis_root)
        self.batch_size           = batch_size
        self.clip_len             = clip_len
        self.output_size          = output_size
        self.num_workers          = num_workers
        self.resolution           = resolution
        self.max_gap              = max_gap
        self.ytv_root             = ytv_root
        self.davis_sampling_ratio = max(0.01, min(0.99, davis_sampling_ratio))
        self.val_output_size      = val_output_size or output_size
        self.train_min_scale      = train_min_scale
        self.train_max_scale      = train_max_scale
        self.train_flip_prob      = train_flip_prob
        self.train_color_jitter_prob = train_color_jitter_prob

        self._train_ds: Optional[Dataset] = None
        self._val_ds:   Optional[Dataset] = None
        self._sampler:  Optional[WeightedRandomSampler] = None

    # ── setup ──────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        train_aug = ClipAugmentor(
            output_size=self.output_size,
            min_scale=self.train_min_scale,
            max_scale=self.train_max_scale,
            flip_prob=self.train_flip_prob,
            color_jitter_prob=self.train_color_jitter_prob,
        )
        val_aug   = ValTransform(output_size=self.val_output_size)

        if stage in ("fit", None):
            davis_train = DAVISVOSTrain(
                root       = str(self.davis_root),
                clip_len   = self.clip_len,
                output_size= self.output_size,
                resolution = self.resolution,
                max_gap    = self.max_gap,
                augmentor  = train_aug,
            )
            if self.ytv_root is not None and Path(self.ytv_root).exists():
                from torch.utils.data import ConcatDataset
                ytv_train = YouTubeVOSTrain(
                    root       = self.ytv_root,
                    clip_len   = self.clip_len,
                    output_size= self.output_size,
                    max_gap    = self.max_gap,
                    augmentor  = train_aug,
                )
                self._train_ds = ConcatDataset([davis_train, ytv_train])

                # WeightedRandomSampler so DAVIS is not drowned by YouTube-VOS.
                # With naive ConcatDataset DAVIS would be ~1.8 % of samples;
                # the sampler ensures it contributes davis_sampling_ratio of
                # every epoch (default 25 %, matching AOT PRE_YTB_DAV balance).
                n_d = len(davis_train)
                n_y = len(ytv_train)
                r   = self.davis_sampling_ratio
                # w_d / (w_d + 1) = r  →  w_d = r * n_y / ((1-r) * n_d)
                w_d = r * n_y / ((1.0 - r) * n_d)
                weights = [w_d] * n_d + [1.0] * n_y
                self._sampler = WeightedRandomSampler(
                    weights,
                    num_samples=len(self._train_ds),
                    replacement=True,
                )
                LOGGER.info(
                    f"[VOSDataModule] Train: DAVIS({n_d}) + YTV({n_y}) "
                    f"= {len(self._train_ds)} clips | "
                    f"DAVIS sampling ratio={r:.0%} (w_davis={w_d:.2f})"
                )
            else:
                self._train_ds = davis_train
                LOGGER.info("[VOSDataModule] Train: DAVIS-only %d clips", len(self._train_ds))

        if stage in ("fit", "validate", None):
            # Val: return first annotated frame of each DAVIS-val sequence
            # as a flat list of (ref, query) clips — one clip per sequence
            self._val_ds = _DAVISValClipDataset(
                root       = str(self.davis_root),
                output_size= self.val_output_size,
                resolution = self.resolution,
                transform  = val_aug,
            )
            LOGGER.info("[VOSDataModule] Val: %d sequences", len(self._val_ds))

    # ── loaders ────────────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        # Use WeightedRandomSampler for joint DAVIS+YTB (shuffle=True is
        # mutually exclusive with a custom sampler).
        return DataLoader(
            self._train_ds,
            batch_size  = self.batch_size,
            sampler     = self._sampler,   # None → default sequential, shuffle below
            shuffle     = self._sampler is None,
            num_workers = self.num_workers,
            collate_fn  = _vos_collate,
            pin_memory  = True,
            drop_last   = True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size  = 1,          # Sequence-level: evaluate one seq at a time
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = _vos_collate,
            pin_memory  = True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Internal helper: val dataset that packages entire sequences as clips
# ══════════════════════════════════════════════════════════════════════════════

class _DAVISValClipDataset(Dataset):
    """Returns each DAVIS-val sequence as a single (ref + all queries) clip.

    Used by :class:`VOSDataModule` for the validation dataloader.
    One item = one full-length sequence (T = total frames - 1).
    """

    def __init__(
        self,
        root: str,
        output_size: int,
        resolution: str = "480p",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root        = Path(root)
        self.output_size = output_size
        self.resolution  = resolution
        self.transform   = transform or ValTransform(output_size)

        split_f = self.root / "ImageSets" / "2017" / "val.txt"
        with open(split_f) as f:
            self.sequences = [l.strip() for l in f if l.strip()]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq  = self.sequences[idx]
        imgs_dir = self.root / "JPEGImages"  / self.resolution / seq
        anns_dir = self.root / "Annotations" / self.resolution / seq

        frame_names = sorted(p.name for p in imgs_dir.glob("*.jpg"))

        all_imgs  = [_load_image(str(imgs_dir / n)) for n in frame_names]
        all_masks = []
        for n in frame_names:
            ap = anns_dir / n.replace(".jpg", ".png")
            all_masks.append(_load_mask(str(ap)) if ap.exists() else
                             np.zeros_like(np.array(all_imgs[0]))[:, :, 0].astype(np.uint8))

        img_tensors, mask_tensors = self.transform(all_imgs, all_masks)

        return {
            "ref_img":     img_tensors[0],
            "ref_mask":    mask_tensors[0],
            "query_imgs":  torch.stack(img_tensors[1:],  dim=0),
            "query_masks": torch.stack(mask_tensors[1:], dim=0),
            "meta": {"seq": seq},
        }


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    DAVIS_ROOT = Path(__file__).resolve().parent.parent / "data" / "DAVIS" / "DAVIS"
    if not DAVIS_ROOT.exists():
        LOGGER.error("[smoke] DAVIS not found at %s", DAVIS_ROOT)
        sys.exit(1)

    LOGGER.info("[smoke] Loading DAVIS train dataset from %s ...", DAVIS_ROOT)
    ds_train = DAVISVOSTrain(str(DAVIS_ROOT), clip_len=4, output_size=224, max_gap=2)
    sample = ds_train[0]
    LOGGER.info("  ref_img:     %s", sample["ref_img"].shape)
    LOGGER.info(
        "  ref_mask:    %s  ids=%s",
        sample["ref_mask"].shape,
        sample["ref_mask"].unique().tolist(),
    )
    LOGGER.info("  query_imgs:  %s", sample["query_imgs"].shape)
    LOGGER.info(
        "  query_masks: %s  ids=%s",
        sample["query_masks"].shape,
        sample["query_masks"].unique().tolist(),
    )

    loader = DataLoader(ds_train, batch_size=2, collate_fn=_vos_collate)
    ref_img, ref_mask, q_imgs, q_masks = next(iter(loader))
    LOGGER.info("[smoke] DataLoader batch:")
    LOGGER.info("  ref_img    %s   dtype=%s", ref_img.shape, ref_img.dtype)
    LOGGER.info("  ref_mask   %s  dtype=%s", ref_mask.shape, ref_mask.dtype)
    LOGGER.info("  q_imgs     %s   dtype=%s", q_imgs.shape, q_imgs.dtype)
    LOGGER.info("  q_masks    %s  dtype=%s", q_masks.shape, q_masks.dtype)

    LOGGER.info("[smoke] data/vos_datamodule.py OK")
