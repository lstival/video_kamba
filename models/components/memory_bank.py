"""Memory Bank for Video Object Segmentation.

Maintains an explicit key-value memory of reference and past predicted frames.
Object identity is injected into memory values via learned ID embeddings at
occupied patch locations — aligning with the AOST/AOT propagation mechanism
while using DINOv2 patch features as the matching signal.

Reference:
    Yang et al., "Associating Objects with Transformers for Video Object
    Segmentation", NeurIPS 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class MemoryBank(nn.Module):
    """Explicit key-value memory bank for VOS propagation.

    Stores per-frame (K, V) pairs where:
    - Keys are projected DINOv2 patch features used for cross-frame matching.
    - Values are projected DINOv2 features enriched with per-object ID
      embeddings at patches that belong to each object (weighted by the
      object's soft mask assignment).

    The bank always retains the reference frame (never evicted) and evicts
    the oldest non-reference frame when ``max_mem_frames`` is exceeded.

    Args:
        d_model:        Input DINOv2 feature dimension (e.g. 768).
        d_key:          Key projection dimension for attention matching.
        d_value:        Value projection dimension for attention readout.
        n_objects:      Maximum number of tracked object IDs (excl. background).
        max_mem_frames: Maximum stored frames including the reference.
    """

    def __init__(
        self,
        d_model: int = 768,
        d_key: int = 256,
        d_value: int = 256,
        n_objects: int = 10,
        max_mem_frames: int = 5,
    ) -> None:
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.n_objects = n_objects
        self.max_mem_frames = max_mem_frames

        # Projection heads: map DINOv2 features to K/V space
        self.proj_key = nn.Sequential(
            nn.Linear(d_model, d_key, bias=False),
            nn.LayerNorm(d_key),
        )
        self.proj_value = nn.Sequential(
            nn.Linear(d_model, d_value, bias=False),
            nn.LayerNorm(d_value),
        )

        # Per-object ID embeddings injected into memory values.
        # Index 0 = background, indices 1..n_objects = object IDs.
        # Shape: [n_objects + 1, d_value]
        self.id_embeddings = nn.Embedding(n_objects + 1, d_value)

        # Internal non-parameter state: managed per video, reset per sequence.
        # Each list entry is a tensor [B, P, d_key/d_value].
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._is_reference: list[bool] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the bank — must be called between videos at inference time."""
        self._keys.clear()
        self._values.clear()
        self._is_reference.clear()

    def encode_reference(
        self,
        frame_features: Float[torch.Tensor, "B P D"],
        mask: torch.Tensor,  # [B, H, W] integer object IDs
    ) -> None:
        """Encode the first annotated frame as the permanent reference entry.

        Resets any previously stored frames before inserting the reference,
        so this method is safe to call at the start of each new video.

        Args:
            frame_features: DINOv2 layer-11 patch features ``[B, P, D]``.
            mask:           Integer segmentation mask ``[B, H, W]``.
        """
        self.reset()
        K, V = self._encode_frame(frame_features, mask)
        self._keys.append(K)
        self._values.append(V)
        self._is_reference.append(True)

    def add_frame(
        self,
        frame_features: Float[torch.Tensor, "B P D"],
        mask: torch.Tensor,  # [B, C, H, W] soft probs or [B, H, W] hard IDs
    ) -> None:
        """Append a new (K, V) pair; evict the oldest non-reference when full.

        Args:
            frame_features: DINOv2 layer-11 patch features ``[B, P, D]``.
            mask:           Soft probability mask ``[B, C, H, W]`` (preferred)
                            or hard integer mask ``[B, H, W]``.
        """
        K, V = self._encode_frame(frame_features, mask)

        if len(self._keys) >= self.max_mem_frames:
            # Evict the oldest non-reference entry (FIFO)
            for i, is_ref in enumerate(self._is_reference):
                if not is_ref:
                    self._keys.pop(i)
                    self._values.pop(i)
                    self._is_reference.pop(i)
                    break

        self._keys.append(K)
        self._values.append(V)
        self._is_reference.append(False)

    def get_memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory concatenated across all stored frames.

        Returns:
            K: Stacked memory keys ``[B, M*P, d_key]``.
            V: Stacked memory values ``[B, M*P, d_value]``.
        """
        assert len(self._keys) > 0, (
            "MemoryBank is empty — call encode_reference() before get_memory()."
        )
        K = torch.cat(self._keys, dim=1)   # [B, M*P, d_key]
        V = torch.cat(self._values, dim=1)  # [B, M*P, d_value]
        return K, V

    def __len__(self) -> int:
        return len(self._keys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_frame(
        self,
        frame_features: Float[torch.Tensor, "B P D"],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project features to K/V space and inject object ID embeddings.

        Object embeddings are added to each patch value as a weighted sum:

        .. math::
            V_{patch} = \\text{proj}_v(f_{patch})
                        + \\sum_o s_o(patch) \\cdot \\mathbf{e}_o

        where :math:`s_o(patch)` is the soft assignment of patch to object
        :math:`o` and :math:`\\mathbf{e}_o = \\text{id\_embeddings}[o]`.

        For a hard integer mask this reduces to an exact ID lookup per patch.

        Args:
            frame_features: ``[B, P, D]``
            mask:           ``[B, H, W]`` (hard) or ``[B, C, H, W]`` (soft)

        Returns:
            K: ``[B, P, d_key]``
            V: ``[B, P, d_value]``
        """
        B, P, _ = frame_features.shape
        h = w = int(P ** 0.5)
        n_cls = self.n_objects + 1  # background + objects

        K = self.proj_key(frame_features)    # [B, P, d_key]
        V = self.proj_value(frame_features)  # [B, P, d_value]

        # Build soft assignment matrix: [B, P, n_cls]
        if mask.ndim == 3:
            # Hard integer mask [B, H, W]
            mask_small = F.interpolate(
                mask.unsqueeze(1).float(), size=(h, w), mode="nearest"
            ).long().squeeze(1)                          # [B, h, w]
            mask_small[mask_small == 255] = 0            # void → background
            mask_small = mask_small.clamp(0, n_cls - 1)
            soft = F.one_hot(mask_small, num_classes=n_cls).float()  # [B, h, w, n_cls]
            soft = soft.reshape(B, P, n_cls)
        else:
            # Soft probability mask [B, C, H, W]
            C = mask.shape[1]
            mask_small = F.interpolate(mask.float(), size=(h, w), mode="area")  # [B, C, h, w]
            soft = mask_small.view(B, C, P).permute(0, 2, 1)                    # [B, P, C]
            if C < n_cls:
                pad = torch.zeros(B, P, n_cls - C, device=mask.device, dtype=soft.dtype)
                soft = torch.cat([soft, pad], dim=-1)
            elif C > n_cls:
                soft = soft[:, :, :n_cls]

        # Weighted ID embedding injection: [B, P, n_cls] @ [n_cls, d_value] → [B, P, d_value]
        id_signal = torch.matmul(soft, self.id_embeddings.weight)  # [B, P, d_value]
        V = V + id_signal

        return K, V
