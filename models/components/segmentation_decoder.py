from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float
import torchvision.ops
import numpy as np
from .fast_kan_layer import FastKANLayer

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

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_heads: int = 4,
        return_gate: bool = False,
    ):
        super().__init__()
        self.return_gate = return_gate
        # Stores averaged attention weights [B, H*W, H*W] when return_gate=True.
        self._last_attn: Optional[torch.Tensor] = None
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
        
        attn_out, attn_weights = self.attn(query=q, key=kv, value=kv)
        if self.return_gate:
            # attn_weights: [B, tgt_len, src_len] (averaged over heads by default)
            self._last_attn = attn_weights.detach() if attn_weights is not None else None
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


class KANSpatialGatingUpBlock(nn.Module):
    """KAN-Modulated Spatial Alignment decoder block.

    Fuses temporal (SSM) context with high-resolution DINOv2 skip features at
    strictly O(N·D) cost — no N×N pixel-to-pixel similarity matrix is computed.

    Three operations per pyramid level
    -----------------------------------
    1. Contextual Projection  (1×1 conv):
       x̃ = Conv_1x1(upsample(x))          [BT, skip_ch, H, W]

    2. KAN Spatial Gating (FastKANLayer applied pixel-wise):
       G = σ(FastKAN(x̃))                  [BT, skip_ch, H, W]   ∈ [0,1]

    3. Modulated Fusion (Hadamard product + residual):
       out = out_conv(x̃ + G ⊙ skip)       [BT, out_ch, H, W]
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, return_gate: bool = False):
        super().__init__()
        self.return_gate = return_gate
        self._last_gate: Optional[torch.Tensor] = None

        # Step 0 — upsample temporal stream (preserves in_channels so proj can see full dim)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # Step 1 — lightweight 1×1 contextual projection
        self.proj = nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(skip_channels)

        # Step 2 — FastKAN gate (pixel-wise, O(N·D))
        # Input/output: [N_pixels, skip_channels] — no spatial pair interactions
        self.gate_kan = FastKANLayer(skip_channels, skip_channels)

        # Step 3 — lightweight output refinement (single 3×3 conv instead of two)
        self.out_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, prev_mask: Optional[torch.Tensor] = None, ref_skip: Optional[torch.Tensor] = None, ref_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [BT, in_channels,   H/2, W/2]  — upsampled temporal feature
            skip: [BT, skip_channels, H,   W  ]  — DINOv2 high-res skip connection
            prev_mask: Not used by this block, kept for API consistency.
            ref_skip: Not used by this block, kept for API consistency.
        Returns:
            [BT, out_channels, H, W]
        """
        # --- Upsample temporal stream ×2 spatially ---
        x_up = self.upsample(x)                                      # [BT, in_ch, H, W]
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = nn.functional.interpolate(
                x_up, size=skip.shape[-2:], mode='bilinear', align_corners=False
            )

        # --- Contextual Projection ---
        x_proj = self.proj_bn(self.proj(x_up))                       # [BT, skip_ch, H, W]

        # --- FastKAN Spatial Gating ---
        BT, C_s, H, W = x_proj.shape
        flat_proj = x_proj.permute(0, 2, 3, 1).reshape(BT * H * W, C_s)
        flat_gate = torch.sigmoid(self.gate_kan(flat_proj))
        gate = flat_gate.reshape(BT, H, W, C_s).permute(0, 3, 1, 2)   # [BT, skip_ch, H, W]

        if self.return_gate:
            self._last_gate = gate.detach()

        # --- Modulated Fusion ---
        # Hadamard product gates the DINOv2 skip features
        fused = x_proj + (gate * skip)

        return self.out_conv(fused)

class KANRefinedCrossAttentionUpBlock(nn.Module):
    """
    Advanced Cross-Attention Bridge with KAN-based Refinement.
    Uses reference mask-weighted attention to retrieve global identity context.
    
    1. Cross-Attention: Current (Q) attends to [Ref * RefMask] (K, V).
    2. KAN Gate: Learns spatial mixing between Attention (Global) and Skip (Local).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_heads: int = 4,
        return_gate: bool = False,
    ):
        super().__init__()
        self.return_gate = return_gate
        self._last_attn = None
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        
        # Attention components
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # KAN-based Fusion Gate (Mixing Local and Global)
        self.fusion_kan = FastKANLayer(out_channels * 2, out_channels)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor, 
        prev_mask: Optional[torch.Tensor] = None,
        ref_skip: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
           x: Current temporal features [BT, in, H/2, W/2]
           skip: Current high-res skip [BT, skip, H, W]
           ref_skip: Reference high-res features [B, skip, H, W] (broadcasted to BT)
           ref_mask: Reference ground-truth mask [B, 1, H, W] (broadcasted to BT)
        """
        # 1. Upsample and project
        x_up = self.upsample(x)
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = nn.functional.interpolate(x_up, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        skip_p = self.skip_proj(skip)
        
        # 2. Cross-Attention Matching (Global Identity)
        if ref_skip is not None and ref_mask is not None:
            B_T, C, H, W = x_up.shape
            
            # Modulate Reference Features with Mask
            # This makes the Key/Value "object-aware"
            if ref_mask.shape[-2:] != (H, W):
                ref_mask = nn.functional.interpolate(ref_mask, size=(H, W), mode='bilinear', align_corners=False)
            
            m_weight = self.mask_proj(ref_mask)
            ref_feat = self.skip_proj(ref_skip) * m_weight
            
            # Prepare for MultiheadAttention [B, Seq, Dim]
            q = x_up.reshape(B_T, C, -1).permute(0, 2, 1)
            kv = ref_feat.reshape(B_T, C, -1).permute(0, 2, 1)
            
            global_context, attn_weights = self.attn(q, kv, kv)
            global_context = global_context.permute(0, 2, 1).reshape(B_T, C, H, W)
            
            if self.return_gate:
                self._last_attn = attn_weights.detach()
                
            # 3. KAN-based Hybrid Fusion (Global Context + Local Detail)
            # Concatenate Global (Attention) and Local (Current Skip)
            combined = torch.cat([global_context, skip_p], dim=1)
            BT_HW, C2 = B_T * H * W, C * 2
            
            fused_flat = self.fusion_kan(combined.permute(0, 2, 3, 1).reshape(BT_HW, C2))
            fused = fused_flat.reshape(B_T, H, W, C).permute(0, 3, 1, 2)
            
            # Final mixing with current features
            x_out = x_up + fused
        else:
            # Fallback to simple fusion
            x_out = x_up + skip_p
            
        return self.out_conv(x_out)


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
            "combined": CombinedUpBlock,
            "kan_spatial": KANSpatialGatingUpBlock,
            "kan_cross_attn": KANRefinedCrossAttentionUpBlock,
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
        dino_features: dict,
        prev_mask: Optional[torch.Tensor] = None,
        ref_features: Optional[dict] = None,
        ref_mask: Optional[torch.Tensor] = None
    ) -> Float[torch.Tensor, "B T num_classes H W"]:
        """
        Args:
            ssm_output: Temporal context from Mamba [B, T, D, P]
            dino_features: Multi-scale dictionary from DinoV3Wrapper
            prev_mask: Optional previous frame mask [B, 1, H_orig, W_orig] or [B, T, 1, H, W]
            ref_features: Optional reference frame multi-scale dictionary
            ref_mask: Optional reference ground-truth mask [B, 1, H_orig, W_orig]
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

        # Reference anchors
        ref_l9 = ref_l6 = ref_l3 = None
        if ref_features is not None:
            # Broadcast reference features across all time steps in the batch
            ref_l9 = ref_features["layer_9"].reshape(B, D_dino, h, w).repeat_interleave(T, dim=0)
            ref_l6 = ref_features["layer_6"].reshape(B, D_dino, h, w).repeat_interleave(T, dim=0)
            ref_l3 = ref_features["layer_3"].reshape(B, D_dino, h, w).repeat_interleave(T, dim=0)
        
        if self.compress_skip:
            l9 = self.skip_conv1(l9)
            l6 = self.skip_conv2(l6)
            l3 = self.skip_conv3(l3)
        
        # Prepare auxiliary masks
        m_guidance = prev_mask
        if m_guidance is not None:
            # Ensure 4D [BT, 1, H, W]
            if m_guidance.ndim == 5:
                # Could be [B, T, C, H, W] or [B, T, 1, H, W]
                m_guidance = m_guidance.reshape(B * T, -1, *m_guidance.shape[-2:])
                if m_guidance.shape[1] > 1: # multi-channel prob mask
                    m_guidance = m_guidance[:, 1:2] # take first object channel
            elif m_guidance.ndim == 3:
                m_guidance = m_guidance.unsqueeze(1).repeat_interleave(T, dim=0)
            m_guidance = m_guidance.float()
            
        r_mask = ref_mask
        if r_mask is not None:
            # Ensure 4D [N, 1, H, W]
            if r_mask.ndim == 5: # [B, 1, 1, H, W]
                r_mask = r_mask.reshape(B, 1, *r_mask.shape[-2:])
            elif r_mask.ndim == 3: # [B, H, W]
                r_mask = r_mask.unsqueeze(1)
            r_mask = r_mask.float()
            r_mask = r_mask.repeat_interleave(T, dim=0) # [B*T, 1, H, W]

        # Recursive fusion
        # Only KANSpatialGatingUpBlock and KANRefinedCrossAttentionUpBlock currently support ref_context
        def execute_up(block, current_x, skip_feat, r_skip, r_mask_g):
            if isinstance(block, KANRefinedCrossAttentionUpBlock):
                return block(current_x, skip_feat, prev_mask=m_guidance, ref_skip=r_skip, ref_mask=r_mask_g)
            elif isinstance(block, KANSpatialGatingUpBlock):
                return block(current_x, skip_feat, prev_mask=m_guidance, ref_skip=r_skip)
            else:
                return block(current_x, skip_feat)

        x = execute_up(self.up1, x, l9, ref_l9, r_mask)
        x = execute_up(self.up2, x, l6, ref_l6, r_mask)
        x = execute_up(self.up3, x, l3, ref_l3, r_mask)
        
        logits = self.final_head(x) # [BT, num_classes, 112, 112]
        
        # Final upscale to target (e.g., 224)
        if logits.shape[-1] != self.target_size:
            logits = nn.functional.interpolate(
                logits, size=(self.target_size, self.target_size), 
                mode='bilinear', align_corners=False
            )
            
        _, C, H, W = logits.shape
        return logits.reshape(B, T, C, H, W)
