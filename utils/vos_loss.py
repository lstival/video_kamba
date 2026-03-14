"""Hybrid VOS Segmentation Loss.

Combines Binary Cross-Entropy (BCE) and Soft Dice to balance pixel-wise
accuracy with region-overlap quality:

.. math::

    \\mathcal{L} = \\beta \\cdot \\mathcal{L}_{\\text{BCE}}
                 + (1 - \\beta) \\cdot \\mathcal{L}_{\\text{SoftDice}}

Each object channel is evaluated independently. A boolean ``obj_present``
mask ensures that absent objects (common in MOSE and YouTube-VOS) are never
penalised – they contribute neither numerator nor denominator to averages.

References
----------
* Milletari et al. (2016): V-Net – fully convolutional neural networks for
  volumetric medical image segmentation (Soft Dice formulation).
* vos.md design specification for this project.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Bool


class HybridVOSLoss(nn.Module):
    """Hybrid BCE + Soft-Dice loss for multi-object VOS segmentation.

    Args:
        beta:         Weight for BCE term.  ``1 - beta`` weights Dice.
                      Default ``0.5`` balances both components equally.
        smooth:       Smoothing constant added to Dice numerator and
                      denominator to avoid division by zero.
        from_logits:  Whether ``pred`` is in logit space (default ``True``).
                      If ``True``, sigmoid is applied internally.
        reduction:    ``"mean"`` averages over present objects; ``"sum"``
                      returns the total.

    Shape expectations
    ------------------
    * ``pred``        – ``[B, T, C, H, W]`` where ``C = n_id + 1`` (channel
                        0 = background; channels 1…n_id = objects).
    * ``target``      – ``[B, T, H, W]``  integer mask (background=0).
    * ``obj_present`` – ``[B, T, n_id]``  boolean; ``True`` iff the object is
                        present in that frame.  Background channel is always
                        included.
    """

    def __init__(
        self,
        beta: float = 0.5,
        smooth: float = 1.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")

        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: Float[torch.Tensor, "B T C H W"],
        target: torch.Tensor,           # [B, T, H, W] int64
        obj_present: Bool[torch.Tensor, "B T n_id"],
    ) -> torch.Tensor:
        """Compute the hybrid loss.

        Args:
            pred:         Logits (or probabilities) ``[B, T, C, H, W]``.
            target:       Integer segmentation masks ``[B, T, H, W]``.
            obj_present:  Boolean presence matrix ``[B, T, n_id]``.

        Returns:
            Scalar loss tensor.
        """
        B, T, C, H, W = pred.shape
        n_id = C - 1  # background does not count in obj_present

        assert target.shape == (B, T, H, W), (
            f"target shape mismatch: expected {(B, T, H, W)}, got {target.shape}"
        )
        assert obj_present.shape == (B, T, n_id), (
            f"obj_present shape mismatch: expected {(B, T, n_id)}, got {obj_present.shape}"
        )

        # Convert logits → probabilities
        if self.from_logits:
            prob = torch.sigmoid(pred)  # [B, T, C, H, W]
        else:
            prob = pred

        # Build one-hot target: [B, T, C, H, W]
        # Clamp target to [0, C-1] to avoid index errors on malformed masks
        target_clamped = target.clamp(0, C - 1)
        target_onehot = F.one_hot(target_clamped, num_classes=C).permute(0, 1, 4, 2, 3).float()

        # ── BCE ───────────────────────────────────────────────────────────
        # Compute per-channel BCE, shape [B, T, C, H, W]
        bce_per_px = F.binary_cross_entropy_with_logits(
            pred if self.from_logits else torch.log(prob.clamp(min=1e-7)),
            target_onehot,
            reduction="none",
        )
        # Average over spatial dims → [B, T, C]
        bce_per_ch = bce_per_px.mean(dim=(-2, -1))

        # ── Soft Dice ─────────────────────────────────────────────────────
        # Compute per-channel Dice, shape [B, T, C]
        numerator = 2.0 * (prob * target_onehot).sum(dim=(-2, -1)) + self.smooth
        denominator = (prob + target_onehot).sum(dim=(-2, -1)) + self.smooth
        dice_per_ch = 1.0 - numerator / denominator  # [B, T, C]

        # ── Combine ───────────────────────────────────────────────────────
        combined = self.beta * bce_per_ch + (1.0 - self.beta) * dice_per_ch  # [B, T, C]

        # ── Mask absent objects ───────────────────────────────────────────
        # Background channel (index 0) is always active
        # Object channels 1…n_id are masked by obj_present[:, :, k-1]
        bg_loss = combined[:, :, 0]  # [B, T]

        # obj_present: [B, T, n_id] → float for weighted averaging
        obj_loss = combined[:, :, 1:]  # [B, T, n_id]
        obj_mask = obj_present.float()  # [B, T, n_id]

        # Sum present-object losses and normalise by number of present objects
        present_count = obj_mask.sum(dim=-1).clamp(min=1.0)  # [B, T]
        masked_obj_loss = (obj_loss * obj_mask).sum(dim=-1) / present_count  # [B, T]

        total_per_frame = bg_loss + masked_obj_loss  # [B, T]

        if self.reduction == "mean":
            loss = total_per_frame.mean()
        else:
            loss = total_per_frame.sum()

        # Sanity guard – should never occur with valid inputs
        assert not torch.isnan(loss).any(), (
            "HybridVOSLoss produced NaN. Check predictions and targets for invalid values."
        )

        return loss

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta}, smooth={self.smooth}, "
            f"from_logits={self.from_logits}, reduction={self.reduction!r}"
        )
