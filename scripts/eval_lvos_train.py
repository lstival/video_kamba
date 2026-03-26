"""LVOS V2 train-set full-sequence evaluation.

Runs the sliding-window SSM inference protocol over every sequence in
``data/LVOS/train/`` and reports per-sequence and aggregate J&F metrics.

Metrics are computed only on annotated frames; unannotated frames are
propagated through the model but excluded from scoring — matching the
LVOS official evaluation protocol.

Reuses :class:`SlidingWindowVOSInference` from ``eval_davis_full_sequence``.

References
----------
Hong et al., "LVOS: A Long-term Video Object Segmentation Benchmark",
ICCV 2023.  https://github.com/LingyiHongfd/lvos-evaluation
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

# PyTorch 2.6 changed the default of `weights_only` in torch.load from False to True,
# breaking Lightning checkpoint loading for checkpoints that contain OmegaConf objects.
# Our checkpoints are internally produced and trusted — revert the default to False.
_torch_load_orig = torch.load
torch.load = lambda *args, **kwargs: _torch_load_orig(  # type: ignore[assignment]
    *args, **{**kwargs, "weights_only": False}
)

# ── Project root and scripts dir on sys.path ──────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Shared helpers from the DAVIS eval script ─────────────────────────────────
from eval_davis_full_sequence import (
    SlidingWindowVOSInference,
    _resize_pred_to_orig,
    _save_prediction_png,
    _seed_everything,
)

from data.vos_datamodule import (
    _MEAN,
    _STD,
    _img_to_tensor,
    _load_image,
    _load_mask,
    _mask_to_tensor,
    _resize_img,
    _resize_mask,
)
from models.video_mamba import VideoMambaSystem
from utils.davis_metrics import evaluate_j_f

LOGGER = logging.getLogger(__name__)

_VOID_LABEL: int = 255


# ══════════════════════════════════════════════════════════════════════════════
# Per-sequence dataset — mirrors DAVISVOSEval for the LVOS train split
# ══════════════════════════════════════════════════════════════════════════════

class LVOSSeqEval:
    """Frame-by-frame dataset for a single LVOS train sequence.

    Mirrors the ``DAVISVOSEval`` interface expected by
    :class:`SlidingWindowVOSInference`.  Frame 0 is always the first
    annotated frame so the engine can initialise the memory bank.

    LVOS train annotations are sparse — only keyframes have PNG masks.
    ``has_ann[i]`` is True only for annotated frames; all other frames
    are included in the sequence and propagated but excluded from scoring.

    Args:
        train_dir:   Path to ``data/LVOS/train/`` (contains ``JPEGImages/``
                     and ``Annotations/``).
        video_id:    Sequence folder name inside ``JPEGImages/``.
        output_size: Shorter-side resize in pixels applied to images and masks
                     before returning tensors.  Pass ``None`` for original size.
    """

    def __init__(
        self,
        train_dir: Path,
        video_id: str,
        output_size: Optional[int] = 480,
    ) -> None:
        self.seq_name   = video_id
        self.output_size = output_size

        self.img_dir = train_dir / "JPEGImages"  / video_id
        self.ann_dir = train_dir / "Annotations" / video_id

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"LVOS image dir not found: {self.img_dir}")
        if not self.ann_dir.is_dir():
            raise FileNotFoundError(f"LVOS annotation dir not found: {self.ann_dir}")

        all_frames  = sorted(self.img_dir.glob("*.jpg"), key=lambda p: p.stem)
        ann_stems   = {p.stem for p in self.ann_dir.glob("*.png")}

        # Find first annotated frame; discard preceding unannotated frames so
        # that frame index 0 is always the reference (with a valid GT mask).
        first_ann_idx = next(
            (i for i, f in enumerate(all_frames) if f.stem in ann_stems),
            None,
        )
        if first_ann_idx is None:
            raise RuntimeError(
                f"[LVOSSeqEval] No annotated frames found for sequence {video_id}"
            )

        # Keep frames from the first annotated frame onwards.
        frames_from_ref = all_frames[first_ann_idx:]
        self.frame_names: List[str] = [f.name for f in frames_from_ref]
        self.has_ann: List[bool]    = [f.stem in ann_stems for f in frames_from_ref]

        # Object IDs from the reference (first annotated) frame.
        ref_ann_path = self.ann_dir / (frames_from_ref[0].stem + ".png")
        ref_mask_np  = _load_mask(str(ref_ann_path))
        obj_ids = sorted(
            set(np.unique(ref_mask_np).tolist()) - {0, _VOID_LABEL}
        )
        self.obj_idx: List[int] = [0] + obj_ids
        self.obj_num: int       = len(obj_ids)

        LOGGER.debug(
            "[LVOSSeqEval] %s  total_frames=%d  annotated=%d  objects=%d",
            video_id,
            len(self.frame_names),
            sum(self.has_ann),
            self.obj_num,
        )

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, idx: int) -> Dict:
        name    = self.frame_names[idx]
        img_pil = _load_image(str(self.img_dir / name))

        if self.output_size is not None:
            img_pil = _resize_img(img_pil, self.output_size)
        img_t = _img_to_tensor(img_pil)

        has_ann = self.has_ann[idx]
        if has_ann:
            ann_name = name.replace(".jpg", ".png")
            mask_np  = _load_mask(str(self.ann_dir / ann_name))
            if self.output_size is not None:
                mask_np = _resize_mask(mask_np, self.output_size)
            mask_t: Optional[torch.Tensor] = _mask_to_tensor(mask_np)
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
# GT and prediction alignment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _collect_gt_masks_lvos(
    seq: LVOSSeqEval,
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, list[int], int]:
    """Load GT masks at original resolution for annotated query frames only.

    Args:
        seq:    Sequence dataset (index 0 is reference, skipped here).
        orig_h: Original frame height.
        orig_w: Original frame width.

    Returns:
        Tuple ``(gt_stack, annotated_query_indices, num_objects)`` where:
        - ``gt_stack`` is uint8 ``[T_ann, orig_h, orig_w]`` — GT for annotated
          query frames only (frame 0 / reference excluded).
        - ``annotated_query_indices`` — dataset indices (>0) that have GT.
        - ``num_objects`` — max object ID across all GT frames.
    """
    gt_frames: list[np.ndarray] = []
    ann_indices: list[int] = []

    for frame_idx in range(1, len(seq)):
        if not seq.has_ann[frame_idx]:
            continue
        name     = seq.frame_names[frame_idx]
        ann_name = name.replace(".jpg", ".png")
        ann_path = seq.ann_dir / ann_name
        if not ann_path.exists():
            continue

        mask_np = _load_mask(str(ann_path))
        if mask_np.shape != (orig_h, orig_w):
            pil = Image.fromarray(mask_np, mode="P")
            pil = pil.resize((orig_w, orig_h), Image.NEAREST)
            mask_np = np.array(pil, dtype=np.uint8)

        gt_frames.append(mask_np)
        ann_indices.append(frame_idx)

    if not gt_frames:
        return np.zeros((0, orig_h, orig_w), dtype=np.uint8), [], 0

    gt_stack = np.stack(gt_frames, axis=0)
    valid    = gt_stack[gt_stack != _VOID_LABEL]
    num_objects = int(valid.max()) if valid.size > 0 else 0
    return gt_stack, ann_indices, num_objects


def _align_predictions_lvos(
    predictions: dict[str, np.ndarray],
    seq: LVOSSeqEval,
    ann_indices: list[int],
) -> np.ndarray:
    """Collect predicted masks for annotated query frames, in GT order.

    Args:
        predictions:  Frame-name → uint8 ``[H, W]`` prediction dict.
        seq:          Sequence dataset (for frame name lookup).
        ann_indices:  Dataset indices of annotated query frames.

    Returns:
        uint8 array ``[T_ann, H, W]``.
    """
    pred_frames: list[np.ndarray] = []
    fallback_shape = next(iter(predictions.values())).shape

    for frame_idx in ann_indices:
        name = seq.frame_names[frame_idx]
        if name in predictions:
            pred_frames.append(predictions[name])
        else:
            LOGGER.warning("Missing prediction for %s — filling with background.", name)
            pred_frames.append(np.zeros(fallback_shape, dtype=np.uint8))

    return np.stack(pred_frames, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Sequence discovery
# ══════════════════════════════════════════════════════════════════════════════

def _load_lvos_train_sequences(lvos_root: Path) -> List[str]:
    """Return sorted video IDs found in the LVOS train split.

    Args:
        lvos_root: Path to the LVOS root (contains ``train/``).

    Returns:
        Sorted list of video ID strings.
    """
    img_root = lvos_root / "train" / "JPEGImages"
    if not img_root.is_dir():
        raise FileNotFoundError(
            f"LVOS train images not found at {img_root}\n"
            "Run scripts/slurm_setup_lvos_lustre.sh to download the dataset."
        )
    return sorted(d.name for d in img_root.iterdir() if d.is_dir())


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point — full-sequence LVOS train-set evaluation.

    Args:
        cfg: Hydra configuration.  Expected keys live under ``cfg.analysis``
             (from ``configs/analysis/lvos_train.yaml``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    analysis_cfg = cfg.get("analysis", cfg)

    checkpoint   = str(analysis_cfg.checkpoint)
    lvos_root    = Path(str(analysis_cfg.lvos_root))
    output_size  = analysis_cfg.get("output_size", 480)
    clip_len     = int(analysis_cfg.clip_len)
    output_dir   = Path(str(analysis_cfg.output_dir))
    save_preds   = bool(analysis_cfg.save_predictions)
    device_str   = str(analysis_cfg.get("device", "cuda"))
    seed         = int(analysis_cfg.get("seed", 42))

    _seed_everything(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    LOGGER.info("Loading checkpoint: %s", checkpoint)
    model: VideoMambaSystem = VideoMambaSystem.load_from_checkpoint(
        checkpoint, strict=False, map_location=device
    )
    model.eval().to(device)
    LOGGER.info(
        "Model loaded — encoder=%s  clip_len=%d  dtsm=%s",
        model.hparams.encoder_type,
        clip_len,
        getattr(model.hparams, "use_dual_timescale_memory", False),
    )

    # ── Inference engine ──────────────────────────────────────────────────────
    engine = SlidingWindowVOSInference(
        model=model,
        clip_len=clip_len,
        output_size=output_size,
        device=device,
    )

    # ── Sequence loop ─────────────────────────────────────────────────────────
    video_ids = _load_lvos_train_sequences(lvos_root)
    LOGGER.info(
        "Evaluating %d LVOS train sequences (full sequence, annotated frames only).",
        len(video_ids),
    )

    train_dir     = lvos_root / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seq_j: list[float] = []
    per_seq_f: list[float] = []
    per_seq_results: dict[str, dict] = {}

    for video_id in video_ids:
        LOGGER.info("[%s] start", video_id)
        try:
            seq = LVOSSeqEval(
                train_dir=train_dir,
                video_id=video_id,
                output_size=output_size,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            LOGGER.warning("[%s] skipping — %s", video_id, exc)
            continue

        if len(seq) < 2:
            LOGGER.warning("[%s] too few frames — skipping.", video_id)
            continue

        if seq.obj_num == 0:
            LOGGER.warning("[%s] no annotated objects — skipping.", video_id)
            continue

        # Run sliding-window inference over the full sequence.
        predictions = engine.run_sequence(seq)

        # Resolve original frame resolution from the reference image.
        ref_img_orig = _load_image(str(seq.img_dir / seq.frame_names[0]))
        orig_h, orig_w = np.array(ref_img_orig).shape[:2]

        # Collect GT and predictions for annotated query frames only.
        gt_stack, ann_indices, num_objects = _collect_gt_masks_lvos(
            seq, orig_h, orig_w
        )

        if gt_stack.shape[0] == 0:
            LOGGER.warning("[%s] no annotated query frames — skipping.", video_id)
            continue

        pred_stack = _align_predictions_lvos(predictions, seq, ann_indices)

        assert pred_stack.shape == gt_stack.shape, (
            f"[{video_id}] shape mismatch: pred={pred_stack.shape} gt={gt_stack.shape}"
        )

        j_score, f_score = evaluate_j_f(
            torch.from_numpy(pred_stack),
            torch.from_numpy(gt_stack),
            num_objects=num_objects,
        )

        per_seq_j.append(float(j_score))
        per_seq_f.append(float(f_score))
        per_seq_results[video_id] = {
            "J":              float(j_score),
            "F":              float(f_score),
            "J&F":            float((j_score + f_score) / 2),
            "num_objects":    num_objects,
            "total_frames":   len(seq) - 1,
            "annotated_frames": len(ann_indices),
        }

        LOGGER.info(
            "[%s] J=%.3f  F=%.3f  J&F=%.3f  (ann=%d/%d frames)",
            video_id, j_score, f_score, (j_score + f_score) / 2,
            len(ann_indices), len(seq) - 1,
        )

        # Save prediction PNGs for all propagated frames.
        if save_preds:
            seq_out_dir = output_dir / "Annotations" / video_id
            seq_out_dir.mkdir(parents=True, exist_ok=True)
            for frame_name, pred_mask in predictions.items():
                out_name = frame_name.replace(".jpg", ".png")
                _save_prediction_png(pred_mask, seq_out_dir / out_name)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_j  = float(np.mean(per_seq_j))  if per_seq_j else 0.0
    mean_f  = float(np.mean(per_seq_f))  if per_seq_f else 0.0
    mean_jf = (mean_j + mean_f) / 2.0

    results = {
        "mean_J":         mean_j,
        "mean_F":         mean_f,
        "mean_J&F":       mean_jf,
        "num_sequences":  len(per_seq_j),
        "per_sequence":   per_seq_results,
        "config": {
            "checkpoint":  checkpoint,
            "clip_len":    clip_len,
            "output_size": output_size,
            "lvos_root":   str(lvos_root),
            "split":       "train",
        },
    }

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    LOGGER.info("=" * 60)
    LOGGER.info("LVOS train J&F evaluation  (%d sequences)", len(per_seq_j))
    LOGGER.info("  Mean J  : %.4f", mean_j)
    LOGGER.info("  Mean F  : %.4f", mean_f)
    LOGGER.info("  Mean J&F: %.4f", mean_jf)
    LOGGER.info("Results saved → %s", results_path)
    if save_preds:
        LOGGER.info("Prediction PNGs → %s", output_dir / "Annotations")


if __name__ == "__main__":
    main()
