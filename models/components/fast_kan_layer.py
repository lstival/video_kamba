"""FastKAN Layer Implementation using Gaussian Radial Basis Functions.

Based on https://github.com/ZiyaoLi/fast-kan
This implementation uses Gaussian RBFs to approximate B-splines, offering
significant speed improvements over the recursive B-spline implementation.

Key features:
1. Gaussian RBF basis functions: exp(-(x - grid)^2 / h)
2. LayerNorm for input scaling (removing need for grid updates)
3. Vectorized operations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

class FastKANLayer(nn.Module):
    """Fast KAN layer using Gaussian Radial Basis Functions (RBF).
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        grid_size: Number of grid points (default: 8)
        grid_range: Range of the grid (default: [-2, 2])
            Note: FastKAN typically uses a wider range than standard KAN [-1, 1]
            because inputs are normalized with LayerNorm.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.grid_range = grid_range

        # LayerNorm to normalize inputs to the grid range
        self.layernorm = nn.LayerNorm(in_features)

        # Grid points (non-learnable)
        # Shape: [grid_size]
        step = (grid_range[1] - grid_range[0]) / grid_size
        self.register_buffer(
            "grid",
            torch.linspace(grid_range[0], grid_range[1], steps=grid_size),
        )
        self.step = step

        # RBF weights
        # Shape: [out_features, in_features, grid_size]
        self.rbf_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )

        # Base linear weights (residual connection)
        # Shape: [out_features, in_features]
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Base bias
        self.base_bias = nn.Parameter(torch.zeros(out_features))

        self._init_weights()

    def _init_weights(self):
        # std=0.05: reduced from 0.1 so KAN modulators start closer to identity
        # (softplus(small) ≈ 1), preventing early gradient explosion through the
        # SSM multiplicative chain (A_bar * h + alpha_B * B @ u).
        nn.init.normal_(self.rbf_weight, mean=0.0, std=0.05)

        # Xavier uniform: matches the SiLU activation used in the base path
        # (Kaiming with a=sqrt(5) assumes ReLU, causing variance inflation).
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        
        # 1. Normalize input
        x_norm = self.layernorm(x_flat)
        
        # 2. Compute Gaussian RBF basis
        # x_norm: [Batch, In, 1]
        # grid:   [1,    1, Grid]
        # basis:  [Batch, In, Grid]
        x_expanded = x_norm.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        
        # Gaussian RBF: exp(-((x - mu) / h)^2)
        # Using a fixed bandwidth relative to grid step
        basis = torch.exp(-((x_expanded - grid_expanded) / (self.step)).pow(2))
        
        # 3. Apply RBF weights
        # basis:      [Batch, In, Grid]
        # rbf_weight: [Out, In, Grid]
        # Output:     [Batch, Out]
        #
        # Computation: sum_{in, grid} (basis_{b,in,g} * weight_{out,in,g})
        # Map to einsum: 'big,oig->bo'
        rbf_output = torch.einsum('big,oig->bo', basis, self.rbf_weight)
        
        # 4. Apply Base linear transformation (SiLU activation for smoothness)
        base_output = F.linear(F.silu(x_flat), self.base_weight, self.base_bias)
        
        # 5. Combine
        output = base_output + rbf_output
        
        return output.view(*original_shape, self.out_features)

    def regularization_loss(self) -> torch.Tensor:
        """L1 regularization on RBF weights."""
        return self.rbf_weight.abs().mean()
