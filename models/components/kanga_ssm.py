import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from jaxtyping import Float

from .kan_ssm_core import IntricateKANSSMCore, DiagonalKANSSMCore


class KangaSSM(nn.Module):
    """Temporal State Space Model for VOS — supports diagonal and dense backends.

    When ``use_diagonal=True`` (default), delegates to :class:`DiagonalKANSSMCore`
    which eliminates ``matrix_exp`` / ``linalg.solve`` and the Python for-loop
    on the T=1 VOS path.  Ablation experiments can set ``use_diagonal=False``
    to recover the original :class:`IntricateKANSSMCore` (B/C modulation only).

    Diagonal mode additionally exposes ``modulate_A`` to ablate KAN on the
    per-channel decay rate — this is the cleanest contribution for publication.

    Args:
        d_model:        Feature dimension D (e.g. 256 for MobileNetV2 encoder).
        d_state:        SSM hidden-state dimension N.
        expand:         Unused; kept for API compatibility.
        num_layers:     Number of stacked SSM layers.
        dropout:        Dropout applied after scan output.
        use_checkpointing: Gradient checkpointing per layer (saves VRAM).
        modulator_type: ``'kan'`` (FastKAN RBF) or ``'mlp'`` (ablation baseline).
        identity_dim:   If not None, concatenate an object-ID vector to modulator
                        input (enables identity-conditioned modulation).
        use_diagonal:   ``True`` → :class:`DiagonalKANSSMCore` (fast, O(N·D));
                        ``False`` → :class:`IntricateKANSSMCore` (legacy, O(N³)).
        modulate_A:     (diagonal only) Enable KAN_A decay-rate modulation.
        modulate_B:     Enable KAN_B state-input modulation.
        modulate_C:     Enable KAN_C readout modulation.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_checkpointing: bool = False,
        modulator_type: str = "kan",
        identity_dim: Optional[int] = None,
        # ── Backend selection ──────────────────────────────────────────
        use_diagonal: bool = True,
        modulate_A: bool = True,   # diagonal only
        modulate_B: bool = True,
        modulate_C: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        self.modulator_type = modulator_type
        self.use_diagonal = use_diagonal

        if use_diagonal:
            self.layers = nn.ModuleList([
                DiagonalKANSSMCore(
                    inner_dim=d_model,
                    state_dim=d_state,
                    modulate_A=modulate_A,
                    modulate_B=modulate_B,
                    modulate_C=modulate_C,
                    modulator_type=modulator_type,
                    identity_dim=identity_dim,
                ) for _ in range(num_layers)
            ])
        else:
            # Legacy dense backend — kept for ablation comparison
            self.layers = nn.ModuleList([
                IntricateKANSSMCore(
                    inner_dim=d_model,
                    state_dim=d_state,
                    modulate_B=modulate_B,
                    modulate_C=modulate_C,
                    modulation_mode="factor",
                    use_fast_kan=True,
                    modulator_type=modulator_type,
                    use_mamba_kernels=True,
                    identity_dim=identity_dim,
                ) for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(d_model)
        # Post-scan normalization: stabilises the unbounded scan output
        # before the output projection. Empirically critical for SSM models
        # (see "Layer-Wise Analysis of Normalization in Mamba", 2025).
        self.post_scan_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gated output projection (SwiGLU, Mamba-2 style).
        # Replaces Linear(D→D) with a split-gate: up * silu(gate).
        # The sigmoid-like gate bounds the output magnitude, which cuts the
        # dL/dW = dL/dy ⊗ y gradient explosion observed in out_proj.weight
        # (peak norm 2.05e+05 in job 65737509 stability_fix_v2).
        # See: Shazeer (2020) "GLU Variants Improve Transformers".
        self.out_proj = nn.Linear(d_model, d_model * 2, bias=False)
        # Small init: 0.02 / sqrt(2 * num_layers) following GPT-2 / Mamba convention
        # for output projections in residual branches — prevents residual branch from
        # dominating the skip connection at step 0.
        nn.init.normal_(
            self.out_proj.weight,
            std=0.02 / math.sqrt(2 * max(num_layers, 1)),
        )

    def forward(
        self,
        x: Float[torch.Tensor, "B T C"],
        prev_states: Optional[list[torch.Tensor]] = None,
        return_last_state: bool = False,
        identity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x:               Input ``[B, T, C]``.
            prev_states:     List of per-layer carry-over states.
                             Diagonal: each ``[B, N]``.
                             Dense:    each ``[B, N, 1]``.
            return_last_state: Return per-layer states after scan.
            identity:        Optional ``[B, identity_dim]`` object-ID context.

        Returns:
            Contextualised ``[B, T, C]`` or ``(output, states)``.
        """
        B, T, C = x.shape
        residual = x
        x = self.norm(x)

        next_states: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            p_state = prev_states[i] if prev_states is not None else None

            if self.use_diagonal:
                # DiagonalKANSSMCore: no external delta; state [B, N]
                if self.use_checkpointing and x.requires_grad:
                    x, n_state = cp.checkpoint(
                        lambda _x, _ps, _id: layer(
                            _x, initial_state=_ps,
                            return_last_state=True, identity=_id
                        ),
                        x, p_state, identity, use_reentrant=False,
                    )
                else:
                    res = layer(
                        x, initial_state=p_state,
                        return_last_state=return_last_state, identity=identity,
                    )
                    if return_last_state:
                        x, n_state = res
                    else:
                        x = res
                        n_state = p_state  # dummy
            else:
                # Legacy IntricateKANSSMCore: needs external delta; state [B, N, 1]
                delta = torch.full((B, T, 1), 0.1, device=x.device, dtype=x.dtype)
                if self.use_checkpointing and x.requires_grad:
                    x, n_state = cp.checkpoint(
                        lambda _x, _d, _ps, _id: layer(
                            _x, _d, initial_state=_ps,
                            return_last_state=True, identity=_id
                        ),
                        x, delta, p_state, identity, use_reentrant=False,
                    )
                else:
                    res = layer(
                        x, delta, initial_state=p_state,
                        return_last_state=return_last_state, identity=identity,
                    )
                    if return_last_state:
                        x, n_state = res
                    else:
                        x = res
                        n_state = p_state  # dummy

            if return_last_state:
                next_states.append(n_state)

        x = self.post_scan_norm(x)
        x = self.dropout(x)
        # SwiGLU gated projection: out_proj outputs [D*2], split into (gate, up).
        # up * silu(gate) bounds output scale — see __init__ for motivation.
        gate, up = self.out_proj(x).chunk(2, dim=-1)
        out = up * F.silu(gate) + residual

        if return_last_state:
            return out, next_states
        return out
