import torch
import torch.nn as nn
from jaxtyping import Float
import torchvision.ops
import numpy as np

class ConcatUpBlock(nn.Module):
    """Original upsampling block with naive concatenation."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class FPNUpBlock(nn.Module):
    """FPN-style addition instead of concatenation."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip = self.skip_proj(skip)
        return self.conv(x + skip)

class CrossAttentionUpBlock(nn.Module):
    """Cross-attention: Semantic features (Query) attend to Spatial Features (Key/Value)."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        skip = self.skip_proj(skip)
        
        B, C, H, W = x.shape
        q = x.view(B, C, -1).permute(0, 2, 1)
        kv = skip.view(B, C, -1).permute(0, 2, 1)
        
        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        x_att = self.norm1(q + attn_out)
        
        ffn_out = self.ffn(x_att)
        out = self.norm2(x_att + ffn_out)
        
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.final_conv(out)

class DeformableUpBlock(nn.Module):
    """Uses Deformable Convolution v2 to align features before fusion."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        
        self.offset_mask_conv = nn.Conv2d(out_channels * 2, 3 * 3 * 3, kernel_size=3, padding=1)
        
        self.dcn_weight = nn.Parameter(torch.Tensor(out_channels, out_channels, 3, 3))
        nn.init.kaiming_uniform_(self.dcn_weight, a=np.sqrt(5))
        self.dcn_bias = nn.Parameter(torch.zeros(out_channels))
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        skip = self.skip_proj(skip)
        
        concat_feat = torch.cat([x, skip], dim=1)
        offset_mask = self.offset_mask_conv(concat_feat)
        
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        aligned_skip = torchvision.ops.deform_conv2d(
            input=skip, 
            offset=offset, 
            weight=self.dcn_weight, 
            bias=self.dcn_bias, 
            padding=1, 
            mask=mask
        )
        
        return self.conv(torch.cat([x, aligned_skip], dim=1))

class CombinedUpBlock(nn.Module):
    """Uses Cross-Attention to refine skip features, then FPN-style addition instead of concatenation."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(out_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        skip = self.skip_proj(skip)
        
        B, C, H, W = x.shape
        q = x.view(B, C, -1).permute(0, 2, 1)
        kv = skip.view(B, C, -1).permute(0, 2, 1)
        
        # Use SSM to query DINO spatial features
        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        
        # Combine DINO with Attended DINO, then FPN add to SSM
        aligned_skip = self.norm(kv + attn_out).permute(0, 2, 1).view(B, C, H, W)
        
        return self.conv(x + aligned_skip)

class SegmentationDecoder(nn.Module):
    """
    Hierarchical Feature Fusion Decoder.
    Fuses deep SSM semantics with high-res intermediate DinoV2 features.
    """
    def __init__(self, dim_ssm: int = 768, dim_dinov2: int = 768, num_classes: int = 11, target_size: int = 224, compress_skip: bool = False, fusion_mode: str = "concat"):
        super().__init__()
        self.num_classes = num_classes
        self.target_size = target_size
        self.compress_skip = compress_skip
        self.fusion_mode = fusion_mode
        
        skip_channels = 128 if compress_skip else dim_dinov2
        
        if compress_skip:
            self.skip_conv1 = nn.Conv2d(dim_dinov2, 128, kernel_size=1)
            self.skip_conv2 = nn.Conv2d(dim_dinov2, 128, kernel_size=1)
            self.skip_conv3 = nn.Conv2d(dim_dinov2, 128, kernel_size=1)

        # Allow dynamic selection of UpBlock
        BlockClass = {
            "concat": ConcatUpBlock,
            "fpn": FPNUpBlock,
            "cross_attn": CrossAttentionUpBlock,
            "deformable": DeformableUpBlock,
            "combined": CombinedUpBlock
        }.get(self.fusion_mode, ConcatUpBlock)

        # Top-down hierarchy: Layer 11 (SSM) -> 9 -> 6 -> 3
        # dim_ssm: Input from SSM (e.g., 1536 in concat mode)
        # dim_dinov2: Input from Dino skip connections (e.g., 768)
        self.up1 = BlockClass(dim_ssm, skip_channels, 256) # From SSM/L11 to L9
        self.up2 = BlockClass(256, skip_channels, 128) # From L9 to L6
        self.up3 = BlockClass(128, skip_channels, 64)  # From L6 to L3
        
        self.final_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(
        self, 
        ssm_output: Float[torch.Tensor, "B T D P"], 
        dino_features: dict
    ) -> Float[torch.Tensor, "B T num_classes H W"]:
        """
        Args:
            ssm_output: Temporal context from Mamba [B, T, D, P]
            dino_features: Multi-scale dictionary from DinoV3Wrapper
        """
        B, T, D_ssm, P = ssm_output.shape
        h = w = int(P ** 0.5)
        
        # Flatten time for spatial ops
        x = ssm_output.reshape(B * T, D_ssm, h, w)
        
        # Intermediate Dino features
        # dino_features: {"layer_9": [B, T, D_dino, P], ...}
        D_dino = dino_features["layer_9"].shape[2]
        l9 = dino_features["layer_9"].reshape(B * T, D_dino, h, w)
        l6 = dino_features["layer_6"].reshape(B * T, D_dino, h, w)
        l3 = dino_features["layer_3"].reshape(B * T, D_dino, h, w)
        
        if self.compress_skip:
            l9 = self.skip_conv1(l9)
            l6 = self.skip_conv2(l6)
            l3 = self.skip_conv3(l3)
        
        # Recursive fusion
        x = self.up1(x, l9) # -> [BT, 256, 28, 28] roughly (Dino patches are 14x14)
        x = self.up2(x, l6) # -> [BT, 128, 56, 56]
        x = self.up3(x, l3) # -> [BT, 64, 112, 112]
        
        logits = self.final_head(x) # [BT, num_classes, 112, 112]
        
        # Final upscale to target (e.g., 224)
        if logits.shape[-1] != self.target_size:
            logits = nn.functional.interpolate(
                logits, size=(self.target_size, self.target_size), 
                mode='bilinear', align_corners=False
            )
            
        _, C, H, W = logits.shape
        return logits.reshape(B, T, C, H, W)
