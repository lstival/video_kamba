import torch
import torch.nn as nn
from jaxtyping import Float

from .kan_ssm_core import IntricateKANSSMCore

class KangaSSM(nn.Module):
    """
    Temporal State Space Model based on the KANGA IntricateKANSSMCore.
    Uses FastKAN for modulating B and C matrices to integrate spatial-temporal dynamics.
    """
    def __init__(self, d_model: int = 768, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # We enforce modulating B and C matrices through FastKAN per the icip_experiment_sota design
        self.core = IntricateKANSSMCore(
            inner_dim=d_model,
            state_dim=d_state,
            modulate_B=True,
            modulate_C=True,
            modulation_mode="factor", 
            use_fast_kan=True,
            use_mamba_kernels=True  # Will silently fallback to naive loop if mamba_ssm not installed
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Projection mixing layer standard in Mamba block architectures (out_proj)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Float[torch.Tensor, "B T C"]) -> Float[torch.Tensor, "B T C"]:
        """
        Input: [B, T, C] from DINO sequence.
        Output: Contextualized [B, T, C].
        """
        B, T, C = x.shape
        residual = x
        x = self.norm(x)
        
        # Assuming a constant delta value for discretization, typical in simplified SSM wrappers 
        # (Alternatively, could be a learned parameter per channel, keeping it simple here)
        delta = torch.full((B, T, 1), 0.1, device=x.device, dtype=x.dtype)
        
        # Sequence processing via Intricate Modulated SSM
        x = self.core(x, delta) 
        
        x = self.out_proj(x)
        
        return x + residual
