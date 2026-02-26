import torch
import torch.nn as nn
from jaxtyping import Float
from typing import Tuple

class DinoV3Wrapper(nn.Module):
    """
    Wrapper for DINOv2 (often referred to as DINOv3 in this context) to extract spatial features.
    Uses 'dinov2_vits14' as a solid, lightweight baseline.
    """
    def __init__(self, use_cls_token: bool = True, freeze: bool = True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.freeze = freeze
        
        # Load real pretrained DINOv2 from torch.hub
        # Note: 'dinov2_vits14' has a feature dim of 384. 
        # If your SSM expects 768, we should use 'dinov2_vitb14'.
        # Let's use 'vitb14' (768) to match the existing config.
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.feature_dim = 768
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: Float[torch.Tensor, "B T C H W"]) -> Tuple[Float[torch.Tensor, "B T D"], Float[torch.Tensor, "B T D P"]]:
        """
        Process sequences of frames.
        Returns:
        
            cls_tokens: shape [B, T, D]
            patch_tokens: shape [B, T, D, P]
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # Get features from DINOv2
        # get_intermediate_layers returns patch tokens (and optionally cls token)
        # For simplicity, we can use the forward pass to get cls token and then extract patches
        
        # Get intermediate layers for patch tokens
        layers = self.backbone.get_intermediate_layers(x_flat, n=1, return_class_token=True)
        # layers[0] is (patch_tokens, cls_token)
        patches_flat, cls_flat = layers[0]
        
        # cls_flat: [BT, D]
        # patches_flat: [BT, P, D] -> [BT, D, P]
        D = self.feature_dim
        P = patches_flat.shape[1]
        
        cls_tokens = cls_flat.view(B, T, D)
        patch_tokens = patches_flat.transpose(1, 2).view(B, T, D, P)
        
        return cls_tokens, patch_tokens
