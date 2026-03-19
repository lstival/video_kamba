"""Full-sequence DAVIS benchmark evaluation with sliding-window SSM inference.

Protocol
--------
For each DAVIS-2017 val sequence:

    1. Encode the first annotated frame → MemoryBank (persistent across the
       entire sequence).
    2. Process query frames in non-overlapping blocks of ``clip_len`` frames:
       * SSM hidden-state **resets** at every block boundary — stays within the
         training-time distribution (trained on clips of exactly ``clip_len``).
       * MemoryBank **accumulates** across all blocks — propagates identity
         information from the reference and every confirmed frame forward.
    3. Collect per-frame segmentation masks → compute J&F over the full
       sequence using the standard DAVIS metrics.

This is the correct long-video inference mode.  The Lightning validation loop
uses ``max_val_frames=16`` purely for training-speed; this script processes the
full sequence for benchmark reporting.

References
----------
Yang et al., "Associating Objects with Transformers for VOS", NeurIPS 2021.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

# ── Project root on path ──────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.vos_datamodule import (
    DAVISVOSEval,
    _MEAN,
    _STD,
    _load_image,
    _load_mask,
    _img_to_tensor,
    _mask_to_tensor,
    _resize_img,
    _resize_mask,
)
from models.video_mamba import VideoMambaSystem
from utils.davis_metrics import evaluate_j_f

LOGGER = logging.getLogger(__name__)

# DAVIS 2017 official colour palette (first 256 colours, matches annotation PNGs).
_DAVIS_PALETTE: list[int] = [
    0,   0,   0,    # 0 background
    128,   0,   0,  # 1
    0, 128,   0,    # 2
    128, 128,   0,  # 3
    0,   0, 128,    # 4
    128,   0, 128,  # 5
    0, 128, 128,    # 6
    128, 128, 128,  # 7
    64,   0,   0,   # 8
    192,   0,   0,  # 9
    64, 128,   0,   # 10
] + [0] * (256 * 3 - 11 * 3)  # fill remainder with zeros


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_prediction_png(mask: np.ndarray, path: Path) -> None:
    """Save a uint8 label mask as a DAVIS-format palette-indexed PNG.

    Args:
        mask: Integer label array ``[H, W]``, dtype uint8.
        path: Destination file path.  Parent directory is created if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(mask, mode="P")
    pil.putpalette(_DAVIS_PALETTE)
    pil.save(str(path))


def _resize_pred_to_orig(
    pred: Float[Tensor, "H W"],
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Resize an integer prediction mask to the original frame resolution.

    Uses nearest-neighbour interpolation to preserve label integrity.

    Args:
        pred:   Predicted label map ``[H, W]``.
        orig_h: Target height in pixels.
        orig_w: Target width in pixels.

    Returns:
        NumPy uint8 array ``[orig_h, orig_w]``.
    """
    if pred.shape == (orig_h, orig_w):
        return pred.numpy().astype(np.uint8)
    pil = Image.fromarray(pred.numpy().astype(np.uint8), mode="P")
    pil = pil.resize((orig_w, orig_h), Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Sequence-level sliding-window inference engine
# ══════════════════════════════════════════════════════════════════════════════

class SlidingWindowVOSInference:
    """Segment a single video sequence using a sliding-window SSM strategy.

    The MemoryBank on ``model`` persists across the entire sequence; the SSM
    hidden state is reset at the start of each ``clip_len``-frame block.  This
    replicates the exact training-time distribution for the SSM (always at most
    ``clip_len`` steps per recurrent pass) while allowing the cross-attention
    memory to grow with the sequence.

    Args:
        model:       Loaded :class:`VideoMambaSystem` in eval mode.
        clip_len:    Block size for the sliding window (should match training).
        output_size: Shorter-side resize in pixels before encoding.  Pass
                     ``None`` to use the original frame resolution.
        device:      Torch device string or object.
    """

    def __init__(
        self,
        model: VideoMambaSystem,
        clip_len: int,
        output_size: Optional[int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.clip_len = clip_len
        self.output_size = output_size
        self.device = device

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run_sequence(
        self,
        seq_dataset: DAVISVOSEval,
    ) -> dict[str, np.ndarray]:
        """Run full-sequence inference and return per-frame label maps.

        Args:
            seq_dataset: Per-sequence evaluation dataset.  Index 0 is the
                         reference frame; indices 1..N are query frames.

        Returns:
            Dictionary mapping ``frame_name`` → uint8 NumPy array ``[H, W]``
            with integer object labels (0 = background).
        """
        model = self.model
        device = self.device

        # ── Reference frame ───────────────────────────────────────────────────
        ref_sample = seq_dataset[0]
        assert ref_sample["has_mask"], (
            f"Sequence '{seq_dataset.seq_name}': reference frame (idx 0) has no "
            "annotation — cannot initialise memory bank."
        )

        ref_img_orig = _load_image(
            str(seq_dataset.img_dir / seq_dataset.frame_names[0])
        )
        orig_h, orig_w = np.array(ref_img_orig).shape[:2]

        ref_img_t = self._preprocess_image(ref_img_orig)   # [3, H', W']
        ref_mask_t = ref_sample["mask"].to(device)          # [H', W'] — already at output_size

        # Feature extraction: [1, 1, 3, H, W]
        ref_img_batch = ref_img_t.unsqueeze(0).unsqueeze(0).to(device)
        _, ref_raw = model.feature_extractor(ref_img_batch)

        ref_ms = model._map_ms_features(ref_raw)
        ref_patch = ref_ms["stage_3"]  # [1, 1, D, P]
        ref_patch_hw = ref_ms.get("stage_3_hw")
        ref_patch_p = ref_patch.squeeze(1).permute(0, 2, 1)  # [1, P, D]

        ref_patch_fine, ref_patch_fine_hw = model._get_fine_features(ref_raw, t=0)  # [1, P2, Ch]

        # Optional identity vector for identity-conditioned modulation.
        identity_vec: Optional[Float[Tensor, "1 D"]] = None
        if getattr(model.hparams, "use_identity_modulation", False):
            identity_vec = self._extract_identity_vec(
                ref_patch_p, ref_mask_t.unsqueeze(0), ref_patch_hw
            )

        # Initialise memory bank with reference frame (resets any previous state).
        model.memory_bank.encode_reference(
            ref_patch_p,
            ref_mask_t.unsqueeze(0),
            ref_patch_fine,
            feat_hw=ref_patch_hw,
            feat_hw_fine=ref_patch_fine_hw,
        )

        # Spatial decoder guidance: binary objectness from reference mask.
        ref_mask_safe = ref_mask_t.clone()
        ref_mask_safe[ref_mask_t == 255] = 0
        _, H_enc, W_enc = ref_img_t.shape
        prev_guide_mask = (ref_mask_safe > 0).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # ── Query frames ─────────────────────────────────────────────────────
        predictions: dict[str, np.ndarray] = {
            seq_dataset.frame_names[0]: _resize_pred_to_orig(
                ref_mask_t.cpu().clamp(0, 255),
                orig_h, orig_w,
            )
        }

        query_indices = list(range(1, len(seq_dataset)))
        n_blocks = (len(query_indices) + self.clip_len - 1) // self.clip_len

        LOGGER.info(
            "  seq=%s  frames=%d  blocks=%d  clip_len=%d",
            seq_dataset.seq_name, len(query_indices), n_blocks, self.clip_len,
        )

        for block_idx in range(n_blocks):
            block_start = block_idx * self.clip_len
            block_end = min(block_start + self.clip_len, len(query_indices))
            block_frame_indices = query_indices[block_start:block_end]

            prev_guide_mask, predictions = self._run_block(
                block_frame_indices=block_frame_indices,
                seq_dataset=seq_dataset,
                prev_guide_mask=prev_guide_mask,
                identity_vec=identity_vec,
                predictions=predictions,
                orig_h=orig_h,
                orig_w=orig_w,
            )

        return predictions

    # ── private helpers ───────────────────────────────────────────────────────

    def _preprocess_image(self, img: Image.Image) -> Float[Tensor, "3 H W"]:
        """Resize (if configured) and normalise an RGB PIL image."""
        if self.output_size is not None:
            img = _resize_img(img, self.output_size)
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(img)
        return TF.normalize(t, mean=_MEAN, std=_STD)

    def _extract_identity_vec(
        self,
        ref_patch_p: Float[Tensor, "1 P D"],
        ref_mask: Float[Tensor, "1 H W"],
        ref_patch_hw: tuple[int, int] | None,
    ) -> Float[Tensor, "1 D"]:
        """Pool reference features weighted by mask objectness → identity vector.

        Mirrors the identity-vector extraction in :meth:`VideoMambaSystem.forward`.

        Args:
            ref_patch_p: Reference coarse patches ``[1, P, D]``.
            ref_mask:    Integer reference mask ``[1, H, W]``.
            ref_patch_hw: Spatial shape ``(gh, gw)`` for patch grid.

        Returns:
            Identity vector ``[1, D]``.
        """
        B, P, D = ref_patch_p.shape
        H, W = ref_mask.shape[-2:]
        if ref_patch_hw is not None:
            gh, gw = ref_patch_hw
        else:
            gh, gw = H // 16, W // 16

        m_down = F.interpolate(
            ref_mask.unsqueeze(1).float(), size=(gh, gw),
            mode="bilinear", align_corners=False,
        )
        m_flat = (m_down.reshape(B, -1) > 0.5).float()       # [1, P]
        denom = m_flat.sum(dim=1, keepdim=True) + 1e-6
        identity_vec = (ref_patch_p * m_flat.unsqueeze(-1)).sum(dim=1) / denom  # [1, D]
        return identity_vec

    def _run_block(
        self,
        block_frame_indices: list[int],
        seq_dataset: DAVISVOSEval,
        prev_guide_mask: Float[Tensor, "1 1 H W"],
        identity_vec: Optional[Float[Tensor, "1 D"]],
        predictions: dict[str, np.ndarray],
        orig_h: int,
        orig_w: int,
    ) -> tuple[Float[Tensor, "1 1 H W"], dict[str, np.ndarray]]:
        """Process one block of frames, returning updated state.

        Feature extraction is batched over the whole block for efficiency.
        The SSM hidden-state starts fresh (``ssm_states=None``) at the top of
        each block, and the MemoryBank carries over from the previous block.

        Args:
            block_frame_indices: Dataset indices for this block (all query frames).
            seq_dataset:         Per-sequence evaluation dataset.
            prev_guide_mask:     Binary objectness guide from the previous block's
                                 last frame ``[1, 1, H, W]``.
            identity_vec:        Pre-computed identity vector (or ``None``).
            predictions:         Accumulator dict — updated in place.
            orig_h:              Original frame height (for mask upsampling).
            orig_w:              Original frame width.

        Returns:
            Tuple ``(updated_prev_guide_mask, updated_predictions)``.
        """
        model = self.model
        device = self.device

        # ── Batch feature extraction for the whole block ──────────────────────
        block_imgs: list[Float[Tensor, "3 H W"]] = []
        for frame_idx in block_frame_indices:
            img_orig = _load_image(
                str(seq_dataset.img_dir / seq_dataset.frame_names[frame_idx])
            )
            block_imgs.append(self._preprocess_image(img_orig))

        # [1, T_block, 3, H, W]
        block_tensor = torch.stack(block_imgs, dim=0).unsqueeze(0).to(device)
        T_block = block_tensor.shape[1]

        _, block_raw = model.feature_extractor(block_tensor)
        block_ms = model._map_ms_features(block_raw)
        block_patch = block_ms["stage_3"]  # [1, T_block, D, P]
        block_patch_hw = block_ms.get("stage_3_hw")
        D = block_patch.shape[2]
        P = block_patch.shape[3]

        # ── Frame-by-frame SSM + propagation loop ─────────────────────────────
        # SSM state resets at the block boundary (stays in training distribution).
        ssm_states = None

        for local_t, frame_idx in enumerate(block_frame_indices):
            frame_name = seq_dataset.frame_names[frame_idx]

            # Coarse patches for this frame: [1, P, D]
            curr_patch = block_patch[:, local_t].permute(0, 2, 1)  # [1, P, D]

            # Fine-scale patches: [1, P2, Ch]
            curr_patch_fine, curr_patch_fine_hw = model._get_fine_features(
                block_raw, t=local_t
            )

            # ── Propagation: cross-attend query patches to memory bank ────────
            K_mem, V_mem = model.memory_bank.get_memory()
            prop_feat = model.propagation_attention(
                curr_patch, K_mem, V_mem
            )  # [1, P, prop_d_value]

            # ── KAN-SSM temporal refinement ───────────────────────────────────
            # T=1 per step (one token per SSM call) — recurrent hidden-state
            # carries temporal context across the block.
            prop_flat = prop_feat.reshape(P, 1, -1)  # [P, 1, prop_d_value]

            # Broadcast identity if enabled: [1, D] → [P, D]
            curr_identity: Optional[Float[Tensor, "P D"]] = None
            if identity_vec is not None:
                curr_identity = (
                    identity_vec.squeeze(0)    # [D]
                    .unsqueeze(0)              # [1, D]
                    .expand(P, -1)             # [P, D]
                )

            ssm_out_flat, ssm_states = model.temporal_model(
                prop_flat,
                prev_states=ssm_states,
                return_last_state=True,
                identity=curr_identity,
            )
            # [P, 1, D] → [1, 1, D, P] for decoder
            last_feat = (
                ssm_out_flat.squeeze(1)    # [P, D]
                .reshape(1, P, D)          # [1, P, D]
                .permute(0, 2, 1)          # [1, D, P]
                .unsqueeze(1)              # [1, 1, D, P]
            )

            # ── Hierarchical decoding ─────────────────────────────────────────
            frame_dec_ms = model._build_dec_ms(block_raw, t=local_t)
            logits_t = model.seg_decoder(
                last_feat, frame_dec_ms, prev_mask=prev_guide_mask
            )  # [1, 1, num_classes, H, W]

            # ── Update memory bank with soft predicted mask ───────────────────
            pred_soft = torch.softmax(logits_t, dim=2).squeeze(1)  # [1, C, H, W]
            model.memory_bank.add_frame(
                curr_patch.detach(),
                pred_soft.detach(),
                curr_patch_fine.detach(),
                feat_hw=block_patch_hw,
                feat_hw_fine=curr_patch_fine_hw,
            )

            # ── Update spatial objectness guide ───────────────────────────────
            prev_guide_mask = (
                pred_soft[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0).detach()
            )  # [1, 1, H, W]

            # ── Store prediction at original resolution ───────────────────────
            pred_hard = torch.argmax(pred_soft, dim=1).squeeze(0).cpu()  # [H, W]
            predictions[frame_name] = _resize_pred_to_orig(
                pred_hard, orig_h, orig_w
            )

        return prev_guide_mask, predictions


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation harness
# ══════════════════════════════════════════════════════════════════════════════

def _load_davis_val_sequences(davis_root: Path, resolution: str) -> list[str]:
    """Return sorted DAVIS-2017 val sequence names."""
    split_f = davis_root / "ImageSets" / "2017" / "val.txt"
    if not split_f.exists():
        raise FileNotFoundError(f"DAVIS val split not found: {split_f}")
    with open(split_f) as fh:
        return [line.strip() for line in fh if line.strip()]


def _collect_gt_masks_orig_res(
    seq_dataset: DAVISVOSEval,
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, int]:
    """Load GT query masks at the **original** frame resolution for benchmark eval.

    Benchmark scores should be computed at original resolution, not the
    network's internal ``output_size``.  This function loads annotation PNGs
    directly from disk (bypassing the dataset's optional resize) so that GT and
    predicted masks share the same spatial dimensions.

    Args:
        seq_dataset: Per-sequence evaluation dataset (used for paths / metadata).
        orig_h:      Original frame height in pixels.
        orig_w:      Original frame width in pixels.

    Returns:
        Tuple ``(gt_stack, num_objects)`` where ``gt_stack`` is a uint8 array
        ``[T_q, orig_h, orig_w]`` and ``num_objects`` is the maximum object ID
        present across all annotated frames.
    """
    gt_frames: list[np.ndarray] = []
    for frame_idx in range(1, len(seq_dataset)):
        name = seq_dataset.frame_names[frame_idx]
        ann_path = seq_dataset.ann_dir / name.replace(".jpg", ".png")
        if ann_path.exists():
            mask_np = _load_mask(str(ann_path))  # at original resolution
            if mask_np.shape != (orig_h, orig_w):
                # Resize with nearest neighbour to match original frame size.
                pil = Image.fromarray(mask_np, mode="P")
                pil = pil.resize((orig_w, orig_h), Image.NEAREST)
                mask_np = np.array(pil, dtype=np.uint8)
        else:
            # No annotation for this frame — fill with void.
            mask_np = np.full((orig_h, orig_w), 255, dtype=np.uint8)
        gt_frames.append(mask_np)

    gt_stack = np.stack(gt_frames, axis=0)  # [T_q, orig_h, orig_w]
    valid = gt_stack[gt_stack != 255]
    num_objects = int(valid.max()) if valid.size > 0 else 0
    return gt_stack, num_objects


def _align_predictions(
    predictions: dict[str, np.ndarray],
    seq_dataset: DAVISVOSEval,
) -> np.ndarray:
    """Build a prediction array ``[T_q, H, W]`` aligned to GT frame order."""
    pred_frames: list[np.ndarray] = []
    for frame_idx in range(1, len(seq_dataset)):
        name = seq_dataset.frame_names[frame_idx]
        if name in predictions:
            pred_frames.append(predictions[name])
        else:
            # Fallback: background if prediction is missing.
            h, w = next(iter(predictions.values())).shape
            LOGGER.warning("Missing prediction for %s — filling with background.", name)
            pred_frames.append(np.zeros((h, w), dtype=np.uint8))
    return np.stack(pred_frames, axis=0)  # [T_q, H, W]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point — full-sequence DAVIS evaluation.

    Args:
        cfg: Hydra configuration.  Expected keys live under ``cfg.analysis``
             (from ``configs/analysis/davis_full_sequence.yaml``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    analysis_cfg = cfg.get("analysis", cfg)

    checkpoint = str(analysis_cfg.checkpoint)
    davis_root = Path(str(analysis_cfg.davis_root))
    resolution = str(analysis_cfg.resolution)
    output_size = analysis_cfg.get("output_size", 480)
    clip_len = int(analysis_cfg.clip_len)
    output_dir = Path(str(analysis_cfg.output_dir))
    save_preds = bool(analysis_cfg.save_predictions)
    device_str = str(analysis_cfg.get("device", "cuda"))
    seed = int(analysis_cfg.get("seed", 42))

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
        "Model loaded — encoder=%s  clip_len=%d  mem_frames=%d",
        model.hparams.encoder_type, clip_len,
        model.hparams.max_mem_frames,
    )

    # ── Inference engine ──────────────────────────────────────────────────────
    engine = SlidingWindowVOSInference(
        model=model,
        clip_len=clip_len,
        output_size=output_size,
        device=device,
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    sequences = _load_davis_val_sequences(davis_root, resolution)
    LOGGER.info("Evaluating %d DAVIS-2017 val sequences (full sequence).", len(sequences))

    per_seq_j: list[float] = []
    per_seq_f: list[float] = []
    per_seq_results: dict[str, dict] = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sequences:
        LOGGER.info("[%s] start", seq_name)
        seq_dataset = DAVISVOSEval(
            root=str(davis_root),
            seq_name=seq_name,
            resolution=resolution,
            output_size=output_size,
        )

        if len(seq_dataset) < 2:
            LOGGER.warning("[%s] too few frames — skipping.", seq_name)
            continue

        if seq_dataset.obj_num == 0:
            LOGGER.warning("[%s] no objects annotated — skipping.", seq_name)
            continue

        # Run sliding-window inference.
        predictions = engine.run_sequence(seq_dataset)

        # Collect GT at original resolution; predictions are also at orig res.
        ref_img_orig_path = seq_dataset.img_dir / seq_dataset.frame_names[0]
        ref_img_pil = _load_image(str(ref_img_orig_path))
        _orig_h, _orig_w = np.array(ref_img_pil).shape[:2]
        gt_stack, num_objects = _collect_gt_masks_orig_res(seq_dataset, _orig_h, _orig_w)
        pred_stack = _align_predictions(predictions, seq_dataset)

        assert pred_stack.shape == gt_stack.shape, (
            f"[{seq_name}] shape mismatch: pred={pred_stack.shape} gt={gt_stack.shape}"
        )

        j_score, f_score = evaluate_j_f(
            torch.from_numpy(pred_stack),
            torch.from_numpy(gt_stack),
            num_objects=num_objects,
        )

        per_seq_j.append(float(j_score))
        per_seq_f.append(float(f_score))
        per_seq_results[seq_name] = {
            "J": float(j_score),
            "F": float(f_score),
            "J&F": float((j_score + f_score) / 2),
            "num_objects": num_objects,
            "num_frames": len(seq_dataset) - 1,
        }

        LOGGER.info(
            "[%s] J=%.3f  F=%.3f  J&F=%.3f",
            seq_name, j_score, f_score, (j_score + f_score) / 2,
        )

        # Save prediction PNGs in DAVIS format.
        if save_preds:
            seq_out_dir = output_dir / "Annotations" / resolution / seq_name
            # Reference frame — copy as-is.
            ref_name = seq_dataset.frame_names[0]
            _save_prediction_png(
                predictions[ref_name],
                seq_out_dir / ref_name.replace(".jpg", ".png"),
            )
            for frame_idx in range(1, len(seq_dataset)):
                name = seq_dataset.frame_names[frame_idx]
                if name in predictions:
                    _save_prediction_png(
                        predictions[name],
                        seq_out_dir / name.replace(".jpg", ".png"),
                    )

    # ── Aggregate results ─────────────────────────────────────────────────────
    mean_j = float(np.mean(per_seq_j)) if per_seq_j else 0.0
    mean_f = float(np.mean(per_seq_f)) if per_seq_f else 0.0
    mean_jf = (mean_j + mean_f) / 2.0

    results = {
        "mean_J": mean_j,
        "mean_F": mean_f,
        "mean_J&F": mean_jf,
        "num_sequences": len(per_seq_j),
        "per_sequence": per_seq_results,
        "config": {
            "checkpoint": checkpoint,
            "clip_len": clip_len,
            "output_size": output_size,
            "resolution": resolution,
            "davis_root": str(davis_root),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    LOGGER.info("=" * 60)
    LOGGER.info("DAVIS full-sequence J&F evaluation  (%d sequences)", len(per_seq_j))
    LOGGER.info("  Mean J  : %.4f", mean_j)
    LOGGER.info("  Mean F  : %.4f", mean_f)
    LOGGER.info("  Mean J&F: %.4f", mean_jf)
    LOGGER.info("Results saved → %s", results_path)
    if save_preds:
        LOGGER.info("Prediction PNGs → %s", output_dir / "Annotations")


if __name__ == "__main__":
    main()
