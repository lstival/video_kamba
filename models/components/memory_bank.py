"""Memory Bank for Video Object Segmentation.

Maintains an explicit key-value memory of reference and past predicted frames.
Object identity is injected into memory values via learned ID embeddings at
occupied patch locations — aligning with the AOST/AOT propagation mechanism
while using Hiera patch features as the matching signal.

Phase 2 — Dual-Scale Keys:
    With the Hiera backbone, the bank stores keys from two scales per frame:
    - **Coarse** (Stage 3, 14×14, 384-dim): semantic matching key.
    - **Fine**   (Stage 2, 28×28, 192-dim): fine-grained localisation key.
    Scale-discriminating embeddings (``scale_embed_coarse/fine``) are added
    to the projected keys so the cross-attention can tell the two scales apart.
    When ``use_dual_scale=False`` the bank behaves exactly as in Phase 1
    (only coarse keys — backward compatible with DINOv2).

Reference:
    Yang et al., "Associating Objects with Transformers for Video Object
    Segmentation", NeurIPS 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

# Import lazily to avoid circular imports; KANKeyAdapter is in the same package
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .kan_key_adapter import KANKeyAdapter


class MemoryBank(nn.Module):
    """Explicit key-value memory bank for VOS propagation.

    Stores per-frame (K, V) pairs where:
    - Keys are projected Hiera Stage-3 (coarse) *and* Stage-2 (fine) patch
      features, concatenated along the sequence dimension so
      PropagationAttention can attend to both scales in a single pass.
    - Values are projected features enriched with per-object ID embeddings
      at patches belonging to each object (weighted by soft mask assignment).

    The bank always retains the reference frame (never evicted) and evicts
    the oldest non-reference frame when ``max_mem_frames`` is exceeded.

    Args:
        d_model:        Coarse (Stage 3) feature dimension (384 for Hiera-B+).
        d_model_fine:   Fine (Stage 2) feature dimension (192 for Hiera-B+).
        d_key:          Key projection dimension for attention matching.
        d_value:        Value projection dimension for attention readout.
        n_objects:      Maximum number of tracked object IDs (excl. background).
        max_mem_frames: Maximum stored frames including the reference.
        use_dual_scale: If True, concatenate coarse + fine keys per frame.
                        If False, only coarse keys are used (Phase 1 / DINOv2
                        backward-compatible mode).
    """

    def __init__(
        self,
        d_model: int = 384,
        d_model_fine: int = 192,
        d_key: int = 256,
        d_value: int = 256,
        n_objects: int = 10,
        max_mem_frames: int = 5,
        use_dual_scale: bool = True,
        key_adapter: "KANKeyAdapter | None" = None,
    ) -> None:
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.n_objects = n_objects
        self.max_mem_frames = max_mem_frames
        self.use_dual_scale = use_dual_scale
        # Optional KAN-SSM adapter that refines non-reference frame keys
        self.key_adapter = key_adapter

        # ── Coarse (Stage 3) projection heads ───────────────────────────
        self.proj_key = nn.Sequential(
            nn.Linear(d_model, d_key, bias=False),
            nn.LayerNorm(d_key),
        )
        self.proj_value = nn.Sequential(
            nn.Linear(d_model, d_value, bias=False),
            nn.LayerNorm(d_value),
        )

        # ── Fine (Stage 2) projection heads (dual-scale only) ───────────
        if use_dual_scale:
            self.proj_key_fine = nn.Sequential(
                nn.Linear(d_model_fine, d_key, bias=False),
                nn.LayerNorm(d_key),
            )
            self.proj_value_fine = nn.Sequential(
                nn.Linear(d_model_fine, d_value, bias=False),
                nn.LayerNorm(d_value),
            )
            # Learnable scale discriminators added to projected keys so
            # PropagationAttention can distinguish coarse vs fine entries.
            # Shape [1, 1, d_key] — broadcast over (B, P).
            self.scale_embed_coarse = nn.Parameter(torch.zeros(1, 1, d_key))
            self.scale_embed_fine   = nn.Parameter(torch.zeros(1, 1, d_key))

        # ── Per-object ID embeddings injected into memory values ─────────
        # Index 0 = background, indices 1..n_objects = object IDs.
        self.id_embeddings = nn.Embedding(n_objects + 1, d_value)

        # ── Internal non-parameter state — managed per video ─────────────
        # Each list entry: [B, P, d_key] / [B, P, d_value].
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
        if self.key_adapter is not None:
            self.key_adapter.reset()

    def encode_reference(
        self,
        frame_features: Float[torch.Tensor, "B P D"],
        mask: torch.Tensor,  # [B, H, W] integer object IDs
        frame_features_fine: Float[torch.Tensor, "B P2 D2"] | None = None,
        feat_hw: tuple[int, int] | None = None,
        feat_hw_fine: tuple[int, int] | None = None,
    ) -> None:
        """Encode the first annotated frame as the permanent reference entry.

        Resets any previously stored frames before inserting the reference,
        so this method is safe to call at the start of each new video.

        Args:
            frame_features:      Coarse (Stage 3) patch features ``[B, P, D]``.
            mask:                Integer segmentation mask ``[B, H, W]``.
            frame_features_fine: Fine (Stage 2) patch features ``[B, P2, D2]``
                                 (optional; used when ``use_dual_scale=True``).
            feat_hw:             Spatial shape ``(Hf, Wf)`` for ``frame_features``.
            feat_hw_fine:        Spatial shape ``(Hf2, Wf2)`` for fine features.
        """
        self.reset()
        K, V = self._encode_frame(
            frame_features,
            mask,
            frame_features_fine,
            feat_hw=feat_hw,
            feat_hw_fine=feat_hw_fine,
        )
        self._keys.append(K)
        self._values.append(V)
        self._is_reference.append(True)

    def add_frame(
        self,
        frame_features: Float[torch.Tensor, "B P D"],
        mask: torch.Tensor,  # [B, C, H, W] soft probs or [B, H, W] hard IDs
        frame_features_fine: Float[torch.Tensor, "B P2 D2"] | None = None,
        feat_hw: tuple[int, int] | None = None,
        feat_hw_fine: tuple[int, int] | None = None,
    ) -> None:
        """Append a new (K, V) pair; evict the oldest non-reference when full.

        Args:
            frame_features:      Coarse (Stage 3) patch features ``[B, P, D]``.
            mask:                Soft probability mask ``[B, C, H, W]``
                                 (preferred) or hard integer mask ``[B, H, W]``.
            frame_features_fine: Fine (Stage 2) patch features ``[B, P2, D2]``
                                 (optional; used when ``use_dual_scale=True``).
            feat_hw:             Spatial shape ``(Hf, Wf)`` for ``frame_features``.
            feat_hw_fine:        Spatial shape ``(Hf2, Wf2)`` for fine features.
        """
        K, V = self._encode_frame(
            frame_features,
            mask,
            frame_features_fine,
            feat_hw=feat_hw,
            feat_hw_fine=feat_hw_fine,
        )

        # Apply KAN-SSM key adaptation to non-reference frames.
        # The adapter adds a recurrent residual so each key is conditioned on
        # the full appearance history seen so far.
        if self.key_adapter is not None:
            # Adapt only the coarse key portion (first P tokens per frame)
            # The coarse key is at the front of K when dual_scale is enabled.
            P_coarse = frame_features.shape[1]  # number of coarse patches
            K_coarse = K[:, :P_coarse, :]       # [B, P, d_key]
            K_coarse_adapted = K_coarse + self.key_adapter.adapt(K_coarse)
            K = torch.cat([K_coarse_adapted, K[:, P_coarse:, :]], dim=1)

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
        frame_features_fine: Float[torch.Tensor, "B P2 D2"] | None = None,
        feat_hw: tuple[int, int] | None = None,
        feat_hw_fine: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project features to K/V space and inject object ID embeddings.

        For dual-scale mode, coarse (Stage 3) and fine (Stage 2) keys are
        concatenated along the sequence dimension.  Scale-discriminating
        embeddings (``scale_embed_coarse/fine``) are added to the respective
        key tensors before concatenation so PropagationAttention can learn
        to weight them differently.

        Coarse K/V object embedding:

        .. math::
            V_{patch} = \\text{proj}_v(f_{patch})
                        + \\sum_o s_o(patch) \\cdot \\mathbf{e}_o

        where :math:`s_o(patch)` is the soft assignment of patch to object
        :math:`o` and :math:`\\mathbf{e}_o = \\text{id\\_embeddings}[o]`.

        Args:
            frame_features:      Coarse patches ``[B, P, D]``.
            mask:                ``[B, H, W]`` (hard) or ``[B, C, H, W]`` (soft).
            frame_features_fine: Fine patches ``[B, P2, D2]`` (optional).

        Returns:
            K: ``[B, P_total, d_key]``  — P_total = P (or P+P2 dual-scale)
            V: ``[B, P_total, d_value]``
        """
        K_c, V_c = self._project_with_id(
            frame_features,
            mask,
            self.proj_key,
            self.proj_value,
            feat_hw=feat_hw,
        )

        if self.use_dual_scale and frame_features_fine is not None:
            # Add scale tokens to coarse keys.
            K_c = K_c + self.scale_embed_coarse

            # Encode fine-scale with its own projections.
            K_f, V_f = self._project_with_id(
                frame_features_fine,
                mask,
                self.proj_key_fine,
                self.proj_value_fine,
                feat_hw=feat_hw_fine,
            )
            K_f = K_f + self.scale_embed_fine

            # Concatenate along patch (sequence) dimension.
            K = torch.cat([K_c, K_f], dim=1)  # [B, P+P2, d_key]
            V = torch.cat([V_c, V_f], dim=1)  # [B, P+P2, d_value]
        else:
            K, V = K_c, V_c

        return K, V

    def _project_with_id(
        self,
        features: torch.Tensor,          # [B, P, D]
        mask: torch.Tensor,              # [B, H, W] or [B, C, H, W]
        proj_key_fn: nn.Module,
        proj_value_fn: nn.Module,
        feat_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``features`` → (K, V) and inject ID embeddings into V.

        Reusable for both coarse and fine scale; the only difference is which
        projection layers are applied.

        Args:
            features:     ``[B, P, D]`` (any scale).
            mask:         ``[B, H, W]`` (hard) or ``[B, C, H, W]`` (soft).
            proj_key_fn:  Key projection module (Linear + LayerNorm).
            proj_value_fn: Value projection module.

        Returns:
            K: ``[B, P, d_key]``
            V: ``[B, P, d_value]``
        """
        B, P, _ = features.shape
        h, w = self._resolve_patch_hw(P, mask, feat_hw)
        n_cls = self.n_objects + 1  # background + objects

        K = proj_key_fn(features)    # [B, P, d_key]
        V = proj_value_fn(features)  # [B, P, d_value]

        # Build soft assignment matrix: [B, P, n_cls]
        if mask.ndim == 3:
            # Hard integer mask [B, H, W]
            mask_small = F.interpolate(
                mask.unsqueeze(1).float(), size=(h, w), mode="nearest"
            ).long().squeeze(1)                          # [B, h, w]
            void_mask = (mask_small == 255)              # [B, h, w] — track void before zeroing
            mask_small[mask_small == 255] = 0            # void → bg index (for one-hot only)
            mask_small = mask_small.clamp(0, n_cls - 1)
            soft = F.one_hot(mask_small, num_classes=n_cls).float()  # [B, h, w, n_cls]
            soft = soft.reshape(B, P, n_cls)
            soft[void_mask.reshape(B, P)] = 0.0          # suppress ID signal for void patches
        else:
            # Soft probability mask [B, C, H, W]
            C = mask.shape[1]
            mask_small = F.interpolate(mask.float(), size=(h, w), mode="area")  # [B, C, h, w]
            soft = mask_small.reshape(B, C, P).permute(0, 2, 1)                  # [B, P, C]
            if C < n_cls:
                pad = torch.zeros(B, P, n_cls - C, device=mask.device, dtype=soft.dtype)
                soft = torch.cat([soft, pad], dim=-1)
            elif C > n_cls:
                soft = soft[:, :, :n_cls]

        # Weighted ID embedding injection: [B, P, n_cls] @ [n_cls, d_value] → [B, P, d_value]
        id_signal = torch.matmul(soft, self.id_embeddings.weight)  # [B, P, d_value]
        V = V + id_signal

        return K, V

    @staticmethod
    def _resolve_patch_hw(
        p: int,
        mask: torch.Tensor,
        feat_hw: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Return spatial (h, w) for a token sequence length ``p``.

        Priority:
        1) Use explicit ``feat_hw`` from the encoder.
        2) Fallback: infer a factor pair from ``p`` matching mask aspect ratio.
        """
        if feat_hw is not None:
            h, w = int(feat_hw[0]), int(feat_hw[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid feat_hw={feat_hw}; expected positive integers.")
            if h * w != p:
                raise ValueError(
                    f"feat_hw={feat_hw} is incompatible with P={p} (h*w={h*w})."
                )
            return h, w

        return MemoryBank._infer_hw_from_mask(p, mask)

    @staticmethod
    def _infer_hw_from_mask(p: int, mask: torch.Tensor) -> tuple[int, int]:
        """Infer patch grid from sequence length using mask aspect ratio."""
        if p <= 0:
            raise ValueError(f"Expected positive P, got {p}.")

        in_h, in_w = int(mask.shape[-2]), int(mask.shape[-1])
        target_aspect = in_w / max(in_h, 1)

        best_hw: tuple[int, int] | None = None
        best_err = float("inf")

        for h in range(1, int(p ** 0.5) + 1):
            if p % h != 0:
                continue
            w = p // h
            for hh, ww in ((h, w), (w, h)):
                err = abs((ww / max(hh, 1)) - target_aspect)
                if err < best_err:
                    best_err = err
                    best_hw = (hh, ww)

        if best_hw is None:
            # Defensive fallback; mathematically unreachable for integer p>0.
            return p, 1
        return best_hw
