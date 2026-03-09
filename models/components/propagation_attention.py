"""Propagation Attention for Video Object Segmentation.

Performs multi-head cross-attention from current-frame query features to the
explicit memory bank (K, V) built from reference and past predicted frames.

This module resolves spatial correspondence — "where is the object in this
frame?" — before the KAN-modulated SSM performs temporal refinement.

Reference:
    Yang et al., "Associating Objects with Transformers for Video Object
    Segmentation", NeurIPS 2021. (LSTT cross-attention design)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class PropagationAttention(nn.Module):
    """Multi-head cross-attention from current frame to the memory bank.

    Given current-frame DINOv2 patch features and explicit memory keys/values
    from :class:`~models.components.memory_bank.MemoryBank`, this module
    computes an object-conditioned representation of each query patch.

    Architecture (per head):

    .. math::
        Q = W_q(\\mathrm{LN}(f_t))                     &\\in \\mathbb{R}^{P \\times d_k}  \\\\
        A = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k/H}}\\right)
                                                       &\\in \\mathbb{R}^{P \\times MP} \\\\
        R = AV                                         &\\in \\mathbb{R}^{P \\times d_v/H} \\\\
        \\text{out} = \\mathrm{LN}(W_o(R) + f_t)       &\\in \\mathbb{R}^{P \\times D}

    The output is in the same space as :class:`~models.components.kanga_ssm.KangaSSM`
    input (``d_model = D``), so no dimension change is needed.

    Args:
        d_model:   Input/output feature dimension (DINOv2 ViT-B/14 = 768).
        d_key:     Key and query projection dimension.
        d_value:   Value projection dimension.
        n_heads:   Number of attention heads.
        dropout:   Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        d_model: int = 768,
        d_key: int = 256,
        d_value: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_key % n_heads == 0, (
            f"d_key ({d_key}) must be divisible by n_heads ({n_heads})."
        )
        assert d_value % n_heads == 0, (
            f"d_value ({d_value}) must be divisible by n_heads ({n_heads})."
        )

        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_heads = n_heads
        self.head_dim_k = d_key // n_heads
        self.head_dim_v = d_value // n_heads
        self.scale = math.sqrt(self.head_dim_k)

        # Query projection (K and V come pre-projected from MemoryBank)
        self.norm_q = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_key, bias=False)

        # Output projection back to d_model
        self.proj_out = nn.Linear(d_value, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query_feat: Float[torch.Tensor, "B P D"],
        memory_K: Float[torch.Tensor, "B MP dk"],
        memory_V: Float[torch.Tensor, "B MP dv"],
    ) -> Float[torch.Tensor, "B P D"]:
        """Cross-attend current-frame patches to the accumulated memory.

        Args:
            query_feat: Current frame DINOv2 patches ``[B, P, d_model]``.
            memory_K:   Stacked memory keys ``[B, M*P, d_key]`` from
                        :meth:`~models.components.memory_bank.MemoryBank.get_memory`.
            memory_V:   Stacked memory values ``[B, M*P, d_value]`` from
                        :meth:`~models.components.memory_bank.MemoryBank.get_memory`.

        Returns:
            Propagated features ``[B, P, d_model]`` ready for KangaSSM input.
            Object identity from the memory is fused into every query patch
            through the attention readout.
        """
        B, P, _ = query_feat.shape
        MP = memory_K.shape[1]

        # ── Query projection ────────────────────────────────────────────
        Q = self.W_q(self.norm_q(query_feat))  # [B, P, d_key]

        # ── Reshape to multi-head: [B, H, seq, head_dim] ────────────────
        Q = Q.view(B, P, self.n_heads, self.head_dim_k).transpose(1, 2)     # [B, H, P,  dk/H]
        K = memory_K.view(B, MP, self.n_heads, self.head_dim_k).transpose(1, 2)  # [B, H, MP, dk/H]
        V = memory_V.view(B, MP, self.n_heads, self.head_dim_v).transpose(1, 2)  # [B, H, MP, dv/H]

        # ── Scaled dot-product attention ─────────────────────────────────
        # A: [B, H, P, MP]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ── Aggregate values ─────────────────────────────────────────────
        out = torch.matmul(attn, V)  # [B, H, P, dv/H]
        out = out.transpose(1, 2).contiguous().view(B, P, self.d_value)  # [B, P, d_value]

        # ── Project to d_model + residual ───────────────────────────────
        out = self.proj_out(out)               # [B, P, d_model]
        out = self.norm_out(out + query_feat)  # residual

        return out
