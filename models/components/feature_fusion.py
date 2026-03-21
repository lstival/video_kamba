import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """Fuse current and reference features — O(n), no attention."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)   # identity at init (delta=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, curr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # curr: [B, P, D], ref: [B, P, D]
        delta = self.proj(torch.cat([curr, ref], dim=-1))
        return self.norm(curr + delta)     # residual onto curr
