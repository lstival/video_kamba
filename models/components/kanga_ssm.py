from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from jaxtyping import Float

from .kan_ssm_core import IntricateKANSSMCore

class KangaSSM(nn.Module):
    """Temporal State Space Model based on the KANGA IntricateKANSSMCore.

    Uses FastKAN for modulating B and C matrices to integrate
    spatial-temporal dynamics.

    Args:
        d_model: Feature dimension.
        d_state: SSM hidden-state dimension.
        expand: Unused expansion factor (reserved for future use).
        num_layers: Number of stacked :class:`IntricateKANSSMCore` layers.
        dropout: Dropout rate applied after scanning.
        use_checkpointing: Enables gradient checkpointing per layer.
        modulator_type: ``'kan'`` (default, FastKAN RBF) or ``'mlp'``
            (dense two-layer MLP baseline for ablation experiments).
    """

    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        expand: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_checkpointing: bool = False,
        modulator_type: str = "kan",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        self.modulator_type = modulator_type

        # We enforce modulating B and C matrices through FastKAN per the icip_experiment_sota design
        self.layers = nn.ModuleList([
            IntricateKANSSMCore(
                inner_dim=d_model,
                state_dim=d_state,
                modulate_B=True,
                modulate_C=True,
                modulation_mode="factor",
                use_fast_kan=True,
                modulator_type=modulator_type,
                use_mamba_kernels=True,  # silently falls back to Python loop if mamba_ssm absent
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Projection mixing layer standard in Mamba block architectures (out_proj)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Float[torch.Tensor, "B T C"],
        prev_states: Optional[list[torch.Tensor]] = None,
        return_last_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Input: [B, T, C] from DINO sequence.
        Output: Contextualized [B, T, C].
        """
        B, T, C = x.shape
        residual = x
        x = self.norm(x)
        
        delta = torch.full((B, T, 1), 0.1, device=x.device, dtype=x.dtype)
        
        next_states = []
        # Sequence processing via Intricate Modulated SSM
        for i, layer in enumerate(self.layers):
            p_state = prev_states[i] if prev_states is not None else None
            
            if self.use_checkpointing and x.requires_grad:
                # Checkpointing usually doesn't play well with returning extra values 
                # unless handled specifically. For simplicity in VOS loop, we might 
                # disable it or handle the state return.
                x, n_state = cp.checkpoint(
                    lambda _x, _d, _ps: layer(_x, _d, initial_state=_ps, return_last_state=True),
                    x, delta, p_state, use_reentrant=False
                )
            else:
                res = layer(x, delta, initial_state=p_state, return_last_state=return_last_state)
                if return_last_state:
                    x, n_state = res
                else:
                    x = res
                    n_state = p_state # dummy
            
            if return_last_state:
                next_states.append(n_state)
                
        x = self.dropout(x)
        x = self.out_proj(x)
        
        out = x + residual
        
        if return_last_state:
            return out, next_states
        return out
