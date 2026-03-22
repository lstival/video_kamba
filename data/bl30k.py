"""BL30K dataset loader for VOS training.

Follows the AOT-protocol used in this project.
BL30K is a large-scale synthetic dataset for Video Object Segmentation.
It is organized into 6 segments (a-f), each with ~5,000 sequences.
Each sequence contains JPEGImages and Annotations.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .vos_datamodule import (
    ClipAugmentor,
    _load_image,
    _load_mask,
    _VOID_LABEL
)

LOGGER = logging.getLogger(__name__)

class BL30KVOSTrain(Dataset):
    """BL30K training set following the AOT clip-sampling protocol.
    
    Args:
        root:      Path to a BL30K segment (contains ``JPEGImages/``, ``Annotations/``).
        clip_len:  Number of query frames per clip (T).
        output_size: Spatial output resolution in pixels.
        max_gap:   Maximum temporal stride between consecutive query frames.
        augmentor: Augmentation callable; defaults to :class:`ClipAugmentor`.
        max_seqs:  Optionally limit the number of sequences (e.g., to fit 100GB).
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 4,
        output_size: int = 224,
        max_gap: int = 3,
        augmentor: Optional[Callable] = None,
        max_seqs: Optional[int] = None,
    ) -> None:
        self.root        = Path(root)
        self.clip_len    = clip_len
        self.output_size = output_size
        self.max_gap     = max_gap

        # BL30K structure: root/JPEGImages/{seq_name}/{frame}.jpg
        # Or sometimes root/{segment_id}/JPEGImages/...
        self.img_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"

        if not self.img_dir.exists():
            # Check one level deeper (e.g., root/BL30K_a/JPEGImages)
            potential_subdirs = [d for d in self.root.iterdir() if d.is_dir() and (d / "JPEGImages").exists()]
            if potential_subdirs:
                self.img_dir = potential_subdirs[0] / "JPEGImages"
                self.ann_dir = potential_subdirs[0] / "Annotations"
                LOGGER.info(f"Found BL30K data in subfolder: {potential_subdirs[0].name}")
            else:
                raise FileNotFoundError(f"BL30K JPEGImages dir not found at {self.img_dir} or its subfolders.")

        # List all sequence IDs (folders in JPEGImages)
        all_seq_ids: List[str] = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
        
        if max_seqs is not None:
            seq_ids = all_seq_ids[:max_seqs]
            LOGGER.info(f"Limiting BL30K to first {max_seqs} sequences.")
        else:
            seq_ids = all_seq_ids

        self._seqs: Dict[str, List[str]] = {}
        for seq in seq_ids:
            frames = sorted([f.name for f in (self.img_dir / seq).glob("*.jpg")])
            if len(frames) >= clip_len + 1:
                self._seqs[seq] = frames

        # Build clips: use first frame as reference
        self._clips: List[Tuple[str, int, int]] = []
        for seq, framenames in self._seqs.items():
            n = len(framenames)
            # Reference is index 0
            for start in range(1, n - clip_len + 1):
                self._clips.append((seq, 0, start))

        self.augmentor = augmentor or ClipAugmentor(output_size=output_size)
        LOGGER.info(f"Loaded BL30K dataset from {root}: {len(self._seqs)} seqs, {len(self._clips)} clips.")

    def _load_frame(self, seq: str, name: str) -> Image.Image:
        return _load_image(str(self.img_dir / seq / name))

    def _load_ann(self, seq: str, name: str) -> np.ndarray:
        ann_name = name.replace(".jpg", ".png")
        ann_path = self.ann_dir / seq / ann_name
        if ann_path.exists():
            return _load_mask(str(ann_path))
        return np.zeros((1, 1), dtype=np.uint8)

    def _count_objects(self, mask: np.ndarray) -> int:
        ids = set(np.unique(mask).tolist()) - {0, int(_VOID_LABEL)}
        return len(ids)

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

        # Load images & masks
        all_imgs  = [self._load_frame(seq, ref_name)] + \
                    [self._load_frame(seq, qn) for qn in query_names]
        all_masks = [self._load_ann(seq, ref_name)] + \
                    [self._load_ann(seq, qn) for qn in query_names]

        # Apply coordinated augmentation
        img_tensors, mask_tensors = self.augmentor(all_imgs, all_masks)

        return {
            "ref_img":     img_tensors[0],
            "ref_mask":    mask_tensors[0],
            "query_imgs":  torch.stack(img_tensors[1:],  dim=0),
            "query_masks": torch.stack(mask_tensors[1:], dim=0),
            "meta": {
                "seq":     seq,
                "obj_num": self._count_objects(all_masks[0]),
            },
        }
