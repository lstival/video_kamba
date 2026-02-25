import torch
import torch.nn as nn
from jaxtyping import Float

class ClassificationHead(nn.Module):
    """
    Video-level classification head.
    Pools over the temporal dimension.
    """
    def __init__(self, dim_in: int = 768, num_classes: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_in // 2),
            nn.ReLU(),
            nn.Linear(dim_in // 2, num_classes)
        )

    def forward(self, x: Float[torch.Tensor, "B T C"]) -> Float[torch.Tensor, "B num_classes"]:
        # Temporal mean pooling
        pooled = x.mean(dim=1)
        return self.head(pooled)
