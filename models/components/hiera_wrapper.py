"""Hiera Backbone Wrapper for Video Object Segmentation.

Replaces the flat DINOv2 backbone with a native hierarchical encoder —
Hiera-Base-Plus-224 from the official facebookresearch/hiera package,
pre-trained with MAE on ImageNet-1K.

Stage dimensions for 224×224 input (embed_dim=112, dim_mul=2.0):
    Stage 1: [B, 112, 56×56]  — fine spatial (stride-4)
    Stage 2: [B, 224, 28×28]  — medium spatial (stride-8)
    Stage 3: [B, 448, 14×14]  — semantic (stride-16), primary matching key
    Stage 4: [B, 896,  7×7]   — global context (captured in cls_token)

Installation:
    pip install hiera-transformer timm

The wrapper returns features in the same [B, T, D, P] format as DinoV3Wrapper
so the rest of the pipeline is unchanged.

Reference:
    Ryali et al., "Hiera: A Hierarchical Vision Transformer without the
    Bells-and-Whistles", ICML 2023.
    https://github.com/facebookresearch/hiera
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class HieraWrapper(nn.Module):
    """Frozen Hiera-Base-Plus backbone returning native multi-scale features.

    Uses the official ``facebookresearch/hiera`` package (``hiera-transformer``
    on PyPI) rather than the HuggingFace ``transformers`` port.
    ``model(x, return_intermediates=True)`` returns NHWC stage tensors that
    are permuted to NCHW here.

    Stage channel widths for ``hiera_base_plus_224``:
        Stage 1 → 112-dim  (56×56)
        Stage 2 → 224-dim  (28×28)
        Stage 3 → 448-dim  (14×14, primary)
        Stage 4 → 896-dim  (7×7, not used directly)

    Args:
        freeze:     Freeze all backbone weights (default True).
        checkpoint: Hiera checkpoint name passed to ``from_pretrained`` /
                    ``torch.hub.load``.  ``"mae_in1k"`` = MAE-only (no
                    fine-tuning), ``"mae_in1k_ft_in1k"`` = fine-tuned.
    """

    # Stage indices in the intermediates list returned by return_intermediates.
    # hiera_base_plus_224 stage_ends = [1, 4, 20, 23]  →  4 intermediates.
    _S1, _S2, _S3 = 0, 1, 2

    # Channel widths per stage for hiera_base_plus_224.
    STAGE_CHANNELS: tuple[int, ...] = (112, 224, 448, 896)

    def __init__(self, freeze: bool = True, checkpoint: str = "mae_in1k") -> None:
        super().__init__()

        # Lazy import — requires: pip install hiera-transformer timm
        try:
            from hiera import Hiera  # type: ignore[import]
            self.backbone: nn.Module = Hiera.from_pretrained(
                f"facebook/hiera_base_plus_224.{checkpoint}"
            )
        except Exception:
            # Fallback to torch.hub (no pip install needed)
            self.backbone = torch.hub.load(
                "facebookresearch/hiera",
                model="hiera_base_plus_224",
                pretrained=True,
                checkpoint=checkpoint,
            )

        self.feature_dim = 448  # Stage 3 primary dim
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
            x: Video clip ``[B, T, C, H, W]`` — float32, normalised by
               ImageNet mean/std.

        Returns:
            cls_token: Global feature ``[B, T, 448]`` — average-pooled Stage 3.
            features:  Dict with keys ``"stage_1"``, ``"stage_2"``,
                       ``"stage_3"`` each in ``[B, T, D, P]`` format::

                           "stage_1": [B, T, 112, 3136]   # 56×56
                           "stage_2": [B, T, 224,  784]   # 28×28
                           "stage_3": [B, T, 448,  196]   # 14×14
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)

        # Hiera expects 224×224 inputs (patch stride-4 → 56-token grid).
        if H != 224 or W != 224:
            x_flat = F.interpolate(
                x_flat, size=(224, 224), mode="bilinear", align_corners=False
            )

        # return_intermediates=True → (logits, [s1, s2, s3, s4])
        # each intermediate is NHWC: [BT, H, W, C]
        _, intermediates = self.backbone(x_flat, return_intermediates=True)

        # Permute NHWC → NCHW
        s1 = intermediates[self._S1].permute(0, 3, 1, 2).contiguous()  # [BT, 112, 56, 56]
        s2 = intermediates[self._S2].permute(0, 3, 1, 2).contiguous()  # [BT, 224, 28, 28]
        s3 = intermediates[self._S3].permute(0, 3, 1, 2).contiguous()  # [BT, 448, 14, 14]

        # CLS token — global avg pool of Stage 3.
        cls_token = s3.mean(dim=[-2, -1]).reshape(B, T, self.feature_dim)  # [B, T, 448]

        def to_btdp(feat: torch.Tensor, D: int, P: int) -> torch.Tensor:
            """[BT, D, H, W] → [B, T, D, P]."""
            return feat.reshape(B * T, D, P).reshape(B, T, D, P)

        features = {
            "stage_1": to_btdp(s1, 112, 56 * 56),  # [B, T, 112, 3136]
            "stage_2": to_btdp(s2, 224, 28 * 28),  # [B, T, 224,  784]
            "stage_3": to_btdp(s3, 448, 14 * 14),  # [B, T, 448,  196]
        }
