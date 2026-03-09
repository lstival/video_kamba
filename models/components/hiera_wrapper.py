"""Hiera Backbone Wrapper for Video Object Segmentation.

Replaces the flat DINOv2 backbone with a native hierarchical encoder —
Hiera (facebook/hiera-base-plus-224) pre-trained with MAE on ImageNet-1K.

Stage dimensions for 224×224 input (patch stride-4):
    Stage 1: [B, 96,  56×56]  — fine spatial (stride-4)
    Stage 2: [B, 192, 28×28]  — medium spatial (stride-8)
    Stage 3: [B, 384, 14×14]  — semantic (stride-16), primary matching key
    Stage 4: [B, 768,  7×7]   — global context (captured in cls_token)

The wrapper returns features in the same [B, T, D, P] format as DinoV3Wrapper
so the rest of the pipeline (MemoryBank, PropagationAttention, KangaSSM,
SegmentationDecoder) can be updated independently.

Reference:
    Ryali et al., "Hiera: A Hierarchical Vision Transformer without the
    Bells-and-Whistles", ICML 2023.
    Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class HieraWrapper(nn.Module):
    """Frozen Hiera-Base-Plus backbone returning native multi-scale features.

    Primary output (Stage 3, 14×14, 384-dim) drives MemoryBank and
    PropagationAttention.  Stage 2 (28×28, 192-dim) provides fine-grained
    secondary keys when ``use_dual_scale=True`` is set on the MemoryBank.
    All three spatial stages feed the genuine FPN decoder.

    Args:
        freeze: Freeze all backbone weights (default True).  When False,
                limited fine-tuning of the backbone is possible.
    """

    # Known channel widths per stage for facebook/hiera-base-plus-224.
    # Spatial dims at 224×224 input: 56, 28, 14, 7.
    STAGE_CHANNELS: tuple[int, ...] = (96, 192, 384, 768)

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        from transformers import HieraModel  # lazy import — optional dependency

        self.backbone: nn.Module = HieraModel.from_pretrained(
            "facebook/hiera-base-plus-224",
        )
        self.feature_dim = 384  # Stage 3 primary dim (matches SSM d_model)
        self.freeze = freeze

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API (same signature as DinoV3Wrapper.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Float[torch.Tensor, "B T C H W"],
    ) -> Tuple[torch.Tensor, dict]:
        """Extract multi-scale spatial features via the Hiera backbone.

        Args:
            x: Video clip ``[B, T, C, H, W]`` — float32, pixel values in
               ``[0, 1]`` or normalised by ImageNet mean/std.

        Returns:
            cls_token: Global feature ``[B, T, 384]`` — average-pooled Stage 3,
                       used as temporal summary for the classification head.
            features:  Dict with keys ``"stage_1"``, ``"stage_2"``,
                       ``"stage_3"`` each in ``[B, T, D, P]`` format::

                           "stage_1": [B, T,  96, 3136]   # 56×56
                           "stage_2": [B, T, 192,  784]   # 28×28
                           "stage_3": [B, T, 384,  196]   # 14×14
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)

        # Hiera expects 224×224 inputs (patch stride-4 → 56-token grid).
        if H != 224 or W != 224:
            x_flat = F.interpolate(
                x_flat, size=(224, 224), mode="bilinear", align_corners=False
            )

        outputs = self.backbone(pixel_values=x_flat, output_hidden_states=True)

        # Retrieve per-stage spatial feature maps in NCHW format.
        s1, s2, s3 = self._extract_stage_maps(outputs)
        # s1: [BT,  96, 56, 56]
        # s2: [BT, 192, 28, 28]
        # s3: [BT, 384, 14, 14]

        # CLS token — global avg pool of Stage 3.
        cls_token = s3.mean(dim=[-2, -1]).reshape(B, T, self.feature_dim)  # [B, T, 384]

        def to_btdp(feat: torch.Tensor, D: int, P: int) -> torch.Tensor:
            """[BT, D, H, W] → [B, T, D, P]."""
            return feat.reshape(B * T, D, P).reshape(B, T, D, P)

        features = {
            "stage_1": to_btdp(s1,  96, 56 * 56),  # [B, T,  96, 3136]
            "stage_2": to_btdp(s2, 192, 28 * 28),  # [B, T, 192,  784]
            "stage_3": to_btdp(s3, 384, 14 * 14),  # [B, T, 384,  196]
        }

        return cls_token, features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_stage_maps(
        self,
        outputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(s1, s2, s3)`` stage feature maps in ``[BT, C, H, W]``.

        Primary path: ``outputs.reshaped_hidden_states`` — per-stage NCHW
        tensors emitted by HuggingFace's HieraModel when
        ``output_hidden_states=True``.

        Fallback: scan ``outputs.hidden_states`` and match by channel count.
        """
        rhs = getattr(outputs, "reshaped_hidden_states", None)
        if rhs is not None and len(rhs) >= 3:
            maps = [self._ensure_nchw(m) for m in rhs]
            s1 = self._pick_by_channel(maps, 96)
            s2 = self._pick_by_channel(maps, 192)
            s3 = self._pick_by_channel(maps, 384)
            if s1 is not None and s2 is not None and s3 is not None:
                return s1, s2, s3

        # Fallback — collect from flat hidden_states tuple.
        hs = getattr(outputs, "hidden_states", None) or ()
        stage_maps: dict[int, torch.Tensor] = {}
        for feat in hs:
            nchw = self._ensure_nchw(feat)
            if nchw is not None and nchw.shape[1] in self.STAGE_CHANNELS:
                stage_maps[nchw.shape[1]] = nchw  # keep last occurrence per C

        missing = [c for c in (96, 192, 384) if c not in stage_maps]
        if missing:
            raise RuntimeError(
                f"HieraWrapper: could not find stage features for channels "
                f"{missing}. Found dims: {sorted(stage_maps)}. "
                f"Ensure transformers>=4.40 and that output_hidden_states=True "
                f"is honoured by the HieraModel."
            )
        return stage_maps[96], stage_maps[192], stage_maps[384]

    @staticmethod
    def _ensure_nchw(feat: torch.Tensor | None) -> torch.Tensor | None:
        """Coerce a tensor to ``[B, C, H, W]`` (NCHW) if it is 4-D.

        HieraModel may return stage features in NHWC (windowed-attention
        native format, last dim = channels) or NCHW.  We distinguish the
        two layouts by checking whether the *last* dim matches a known Hiera
        channel width — spatial dims (56, 28, 14, 7) never overlap with
        channel dims (96, 192, 384, 768) for hiera-base-plus.
        """
        if feat is None or feat.ndim != 4:
            return None
        _, d1, d2, d3 = feat.shape
        # NHWC: last dim is channel (96, 192, 384, or 768).
        if d3 in (96, 192, 384, 768):
            return feat.permute(0, 3, 1, 2).contiguous()
        # NCHW: second dim is channel.
        return feat

    @staticmethod
    def _pick_by_channel(
        maps: list[torch.Tensor | None],
        channel: int,
    ) -> torch.Tensor | None:
        """Return the first tensor in *maps* whose channel dim equals *channel*."""
        for m in maps:
            if m is not None and m.ndim == 4 and m.shape[1] == channel:
                return m
        return None
