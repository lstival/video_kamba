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

    def forward(self, x: Float[torch.Tensor, "B T C H W"]) -> Tuple[torch.Tensor, dict]:
        """
        Process sequences of frames and return multi-scale features.
        
        Returns:
            cls_token: shape [B, T, D] (from last layer)
            features: dictionary of {scale: tensor} where tensor is [B, T, D, P]
                      Scales typically include layers 3, 6, 9, 11 for hierarchical fusion.
        """
        import torch.nn.functional as F
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        D = self.feature_dim
        
        # Ensure image dimensions are multiples of 14 for DINOv2 patch size
        patch_size = 14
        if H % patch_size != 0 or W % patch_size != 0:
            new_H = (H // patch_size) * patch_size
            new_W = (W // patch_size) * patch_size
            x_flat = F.interpolate(x_flat, size=(new_H, new_W), mode='bilinear', align_corners=False)
        h_tokens = x_flat.shape[-2] // patch_size
        w_tokens = x_flat.shape[-1] // patch_size
            
        # Extract features from multiple intermediate layers (3, 6, 9, 11)
        # This allows the decoder to have high-res spatial details (early layers) 
        # and deep semantic context (later layers).
        layers_idxs = [3, 6, 9, 11]
        multi_scale_out = self.backbone.get_intermediate_layers(x_flat, n=layers_idxs, return_class_token=True)
        
        # multi_scale_out is a list of (patch_tokens, cls_token) for each layer requested
        features = {}
        for idx, (patches_flat, cls_flat) in zip(layers_idxs, multi_scale_out):
            # patches_flat: [BT, P, D] -> [B, T, D, P]
            P = patches_flat.shape[1]
            features[f"layer_{idx}"] = patches_flat.transpose(1, 2).view(B, T, D, P)
            features[f"layer_{idx}_hw"] = (h_tokens, w_tokens)
        
        # Use the CLS token from the last layer (idx 11) for global features
        last_cls = multi_scale_out[-1][1].view(B, T, D)
        
        return last_cls, features
