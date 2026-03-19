"""Propagation Attention for Video Object Segmentation.

Performs multi-head cross-attention from current-frame query features to the
explicit memory bank (K, V) built from reference and past predicted frames.

This module resolves spatial correspondence — "where is the object in this
frame?" — before the KAN-modulated SSM performs temporal refinement.

Reference:
    Yang et al., "Associating Objects with Transformers for Video Object
    Segmentation", NeurIPS 2021. (LSTT cross-attention design)

Stability fixes applied (Job 65725194 analysis):
    - Xavier initialization for all linear projections
    - QK-Norm post-projection (Dehghani et al., 2023)
    - LayerNorm on incoming memory K/V
    - fp32 softmax for numerical safety under mixed-precision
    - Attention logit clamping as safety net
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
        Q = \\mathrm{QKNorm}(W_q(\\mathrm{LN}(f_t)))  &\\in \\mathbb{R}^{P \\times d_k}  \\\\
        K = \\mathrm{QKNorm}(\\mathrm{LN}(K_{mem}))    &\\in \\mathbb{R}^{MP \\times d_k} \\\\
        A = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k/H}}\\right)
                                                       &\\in \\mathbb{R}^{P \\times MP} \\\\
        R = AV                                         &\\in \\mathbb{R}^{P \\times d_v/H} \\\\
        \\text{out} = \\mathrm{LN}(W_o(R) + f_t)       &\\in \\mathbb{R}^{P \\times D}

    Stability notes:
        - QK-Norm prevents attention logit explosion. See Dehghani et al.,
          "Scaling Vision Transformers to 22 Billion Parameters", 2023.
        - Softmax is computed in fp32 regardless of AMP context.
        - Xavier init ensures projection weight scale matches attention regime.

    Args:
        d_model:   Input/output feature dimension (DINOv2 ViT-B/14 = 768).
        d_key:     Key and query projection dimension.
        d_value:   Value projection dimension.
        n_heads:   Number of attention heads.
        dropout:   Dropout probability applied to attention weights.
        logit_clamp: Maximum absolute value for attention logits (safety net).
    """

    def __init__(
        self,
        d_model: int = 768,
        d_key: int = 256,
        d_value: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        logit_clamp: float = 50.0,
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
        self.logit_clamp = logit_clamp

        # ── Query projection ─────────────────────────────────────────────
        self.norm_q = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_key, bias=False)

        # ── FIX 1: LayerNorm for incoming memory K and V ─────────────────
        # Memory keys/values arrive pre-projected from MemoryBank but
        # without normalisation. This balances magnitude with normalised Q.
        self.norm_k = nn.LayerNorm(d_key)
        self.norm_v = nn.LayerNorm(d_value)

        # ── FIX 2: QK-Norm (post-projection, per-head) ──────────────────
        # Prevents attention logit explosion by normalising Q and K vectors
        # after projection, before the dot product.
        # See: Dehghani et al., "Scaling ViTs to 22B", 2023, Sec. 3.2
        self.qk_norm_q = nn.LayerNorm(self.head_dim_k, elementwise_affine=False)
        self.qk_norm_k = nn.LayerNorm(self.head_dim_k, elementwise_affine=False)

        # ── Output projection back to d_model ────────────────────────────
        self.proj_out = nn.Linear(d_value, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)

        self.attn_drop = nn.Dropout(dropout)

        # ── FIX 3: Proper Xavier initialization ──────────────────────────
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier uniform init for attention projections.

        Kaiming init (PyTorch default for nn.Linear) assumes ReLU
        activations and produces weight norms that are too large for
        the softmax-based attention regime, causing gradient explosion.

        Xavier uniform maintains variance across the projection and is
        the standard choice for Transformer attention layers.

        Additionally, proj_out uses a scaled init (1/sqrt(2*n_layers))
        pattern common in deep residual networks. Since we don't know
        n_layers here, we use a conservative 0.5x scaling.
        """
        nn.init.xavier_uniform_(self.W_q.weight)

        # Output projection: scaled down to prevent residual branch
        # from dominating the skip connection at initialisation.
        nn.init.xavier_uniform_(self.proj_out.weight)
        with torch.no_grad():
            self.proj_out.weight.mul_(0.5)

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

        # ── FIX 1: Normalize incoming memory K/V ────────────────────────
        K = self.norm_k(memory_K)   # [B, MP, d_key]
        V = self.norm_v(memory_V)   # [B, MP, d_value]

        # ── Reshape to multi-head: [B, H, seq, head_dim] ────────────────
        Q = Q.view(B, P, self.n_heads, self.head_dim_k).transpose(1, 2)
        K = K.view(B, MP, self.n_heads, self.head_dim_k).transpose(1, 2)
        V = V.view(B, MP, self.n_heads, self.head_dim_v).transpose(1, 2)

        # ── FIX 2: QK-Norm (per-head normalisation before dot product) ──
        Q = self.qk_norm_q(Q)  # [B, H, P,  dk/H]
        K = self.qk_norm_k(K)  # [B, H, MP, dk/H]

        # ── Scaled dot-product attention ─────────────────────────────────
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ── FIX 5: Clamp logits as safety net against fp16 overflow ─────
        attn_logits = attn_logits.clamp(
            min=-self.logit_clamp, max=self.logit_clamp,
        )

        # ── FIX 4: Force softmax to fp32 for numerical stability ────────
        # Under AMP (16-mixed), large logits cause softmax to saturate in
        # fp16 (max ~65504), producing zero gradients. Casting to fp32
        # before softmax and back afterwards prevents this.
        attn = F.softmax(attn_logits.float(), dim=-1).to(Q.dtype)
        attn = self.attn_drop(attn)

        # ── Aggregate values ─────────────────────────────────────────────
        out = torch.matmul(attn, V)  # [B, H, P, dv/H]
        out = out.transpose(1, 2).contiguous().view(B, P, self.d_value)

        # ── Project to d_model + residual ───────────────────────────────
        out = self.proj_out(out)               # [B, P, d_model]
        out = self.norm_out(out + query_feat)  # residual

        return out