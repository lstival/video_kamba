import torch
import torch.nn as nn
from jaxtyping import Float

class SegmentationDecoder(nn.Module):
    """
    Per-frame segmentation decoder.
    Takes SSM outputs and projects to mask logits.
    """
    def __init__(self, dim_in: int = 768, num_classes: int = 10, target_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.target_size = target_size
        
        # Simulating a simple transposed conv decoder path
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim_in, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )
        
    def forward(self, patch_tokens: Float[torch.Tensor, "B T D P"]) -> Float[torch.Tensor, "B T num_classes H W"]:
        """
        Accepts patch tokens. Returns dense segmentations.
        P is spatial patches (e.g., 14*14=196)
        """
        B, T, D, P = patch_tokens.shape
        # Assuming P = 14x14
        h = w = int(P ** 0.5)
        
        # [B*T, D, h, w]
        x = patch_tokens.reshape(B * T, D, h, w)
        
        # [B*T, num_classes, H', W']
        logits = self.decoder(x)
        
        # Interpolate to target size if needed
        if logits.shape[-1] != self.target_size:
            logits = nn.functional.interpolate(logits, size=(self.target_size, self.target_size), 
                                               mode='bilinear', align_corners=False)
            
        _, C, H, W = logits.shape
        return logits.reshape(B, T, C, H, W)
