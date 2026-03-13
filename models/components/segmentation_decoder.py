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

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, return_gate: bool = False, use_mask_guidance: bool = True):
        super().__init__()
        self.return_gate = return_gate
        self.use_mask_guidance = use_mask_guidance
        self._last_gate: Optional[torch.Tensor] = None

        # Step 0 — upsample temporal stream
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # Step 1 — lightweight 1×1 contextual projection
        self.proj = nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(skip_channels)

        # Step 2 — FastKAN gate (pixel-wise, O(N·D))
        # If mask-guided, we add 1 channel for the previous mask
        kan_in = skip_channels + 1 if use_mask_guidance else skip_channels
        self.gate_kan = FastKANLayer(kan_in, skip_channels)

        # Step 3 — output refinement
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
            prev_mask: [BT, 1, H, W] — previous soft-mask guidance
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
        
        if self.use_mask_guidance and prev_mask is not None:
            # Resize mask to current resolution if needed
            if prev_mask.shape[-2:] != (H, W):
                m_guide = nn.functional.interpolate(prev_mask, size=(H, W), mode='bilinear', align_corners=False)
            else:
                m_guide = prev_mask
            
            # Concatenate mask as an extra feature channel for the KAN gate
            kan_input = torch.cat([x_proj, m_guide], dim=1) # [BT, C_s+1, H, W]
        else:
            kan_input = x_proj

        flat_in = kan_input.permute(0, 2, 3, 1).reshape(-1, kan_input.shape[1])
        flat_gate = torch.sigmoid(self.gate_kan(flat_in))
        gate = flat_gate.reshape(BT, H, W, C_s).permute(0, 3, 1, 2)   # [BT, skip_ch, H, W]

        if self.return_gate:
            self._last_gate = gate.detach()

        # --- Modulated Fusion ---
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
    """Hierarchical Feature Fusion Decoder — genuine FPN with Hiera skip connections.

    Fuses SSM temporal semantics with native multi-scale Hiera features:

    +---------+------------------+------------------+
    | Level   | Input resolution | Skip source      |
    +=========+==================+==================+
    | up1     | 14×14 → 14×14*   | Stage 3, 384-dim |
    +---------+------------------+------------------+
    | up2     | 14×14 → 28×28    | Stage 2, 192-dim |
    +---------+------------------+------------------+
    | up3     | 28×28 → 56×56    | Stage 1,  96-dim |
    +---------+------------------+------------------+
    | head    | 56×56 → 224×224  | —                |
    +---------+------------------+------------------+

    (*) up1 upsample ×2 then resizes back to skip's resolution; this acts as
    a semantic refinement step at the same 14×14 scale.

    Args:
        dim_ssm:    SSM output channel dimension (384 for Hiera-B+).
        skip_dims:  Tuple of (skip1, skip2, skip3) channel dims for up1/up2/up3.
                    Defaults to Hiera-B+ native dims ``(384, 192, 96)``.
        decoder_dims: Tuple of decoder channel widths for up1/up2/up3.
                     Defaults to ``(256, 128, 64)``.
        num_classes: Segmentation output channels (background + n_id).
        target_size: Final spatial output resolution (default 224).
        fusion_mode: Up-block variant — see block class map below.
    """
    def __init__(
        self,
        dim_ssm: int = 384,
        skip_dims: tuple[int, int, int] = (384, 192, 96),
        decoder_dims: tuple[int, int, int] = (256, 128, 64),
        num_classes: int = 11,
        target_size: int = 224,
        fusion_mode: str = "kan_spatial",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_size = target_size
        self.fusion_mode = fusion_mode

        skip1, skip2, skip3 = skip_dims  # per-level channel dims
        dec1, dec2, dec3 = decoder_dims

        BlockClass = {
            "concat": ConcatUpBlock,
            "fpn": FPNUpBlock,
            "cross_attn": CrossAttentionUpBlock,
            "deformable": DeformableUpBlock,
            "combined": CombinedUpBlock,
            "kan_spatial": KANSpatialGatingUpBlock,
            "kan_cross_attn": KANRefinedCrossAttentionUpBlock,
        }.get(self.fusion_mode, ConcatUpBlock)

        use_g = self.fusion_mode in ("kan_spatial", "kan_cross_attn")

        def _make_up_block(in_ch: int, skip_ch: int, out_ch: int) -> nn.Module:
            if BlockClass is KANSpatialGatingUpBlock:
                return BlockClass(in_ch, skip_ch, out_ch, use_mask_guidance=use_g)
            return BlockClass(in_ch, skip_ch, out_ch)

        # Genuine top-down pyramid — each skip has a different spatial resolution.
        # up1: SSM (14×14) fused with Stage 3 (14×14)  → 256-ch  (semantic refinement)
        # up2: 256 (14×14) fused with Stage 2 (28×28)  → 128-ch  (first real upsample)
        # up3: 128 (28×28) fused with Stage 1 (56×56)  →  64-ch  (second real upsample)
        self.up1 = _make_up_block(dim_ssm, skip1, dec1)
        self.up2 = _make_up_block(dec1, skip2, dec2)
        self.up3 = _make_up_block(dec2, skip3, dec3)

        self.final_head = nn.Sequential(
            nn.Conv2d(dec3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self,
        ssm_output: Float[torch.Tensor, "B T D P"],
        skip_features: dict,
        prev_mask: Optional[torch.Tensor] = None,
        ref_features: Optional[dict] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "B T num_classes H W"]:
        """Decode SSM temporal features into segmentation logits.

        Args:
            ssm_output:    Temporal context ``[B, T, D, P]`` from KangaSSM.
                           P = 196 (14×14) for Hiera-B+.
            skip_features: Multi-scale feature dict from HieraWrapper::

                               "stage_3": [B, T, 384, 196]   # 14×14
                               "stage_2": [B, T, 192, 784]   # 28×28
                               "stage_1": [B, T,  96, 3136]  # 56×56

            prev_mask:     Optional objectness prior ``[B, 1, H, W]``.
            ref_features:  Optional reference multi-scale dict (same keys).
            ref_mask:      Optional reference GT mask ``[B, 1, H, W]``.

        Returns:
            ``[B, T, num_classes, H, W]`` segmentation logits at *target_size*.
        """
        B, T, D_ssm, P = ssm_output.shape
        h3 = w3 = int(P ** 0.5)  # 14 for Hiera Stage 3

        # ── Flatten time for 2-D spatial ops ────────────────────────────
        x = ssm_output.reshape(B * T, D_ssm, h3, w3)  # [BT, 384, 14, 14]

        # ── Hiera skip features ─────────────────────────────────────────
        # Stage 3  14×14  384-ch  (same resolution as SSM output → semantic refinement)
        s3_feat = skip_features["stage_3"]
        h_s3 = w_s3 = int(s3_feat.shape[3] ** 0.5)        # 14
        s3 = s3_feat.reshape(B * T, s3_feat.shape[2], h_s3, w_s3)  # [BT, 384, 14, 14]

        # Stage 2  28×28  192-ch  (genuine upsample target for up2)
        s2_feat = skip_features["stage_2"]
        h_s2 = w_s2 = int(s2_feat.shape[3] ** 0.5)        # 28
        s2 = s2_feat.reshape(B * T, s2_feat.shape[2], h_s2, w_s2)  # [BT, 192, 28, 28]

        # Stage 1  56×56   96-ch  (genuine upsample target for up3)
        s1_feat = skip_features["stage_1"]
        h_s1 = w_s1 = int(s1_feat.shape[3] ** 0.5)        # 56
        s1 = s1_feat.reshape(B * T, s1_feat.shape[2], h_s1, w_s1)  # [BT,  96, 56, 56]

        # ── Reference anchors (optional, broadcast over T) ────────────────
        ref_s3 = ref_s2 = ref_s1 = None
        if ref_features is not None:
            def _broadcast_ref(feat):
                D_r = feat.shape[2]
                P_r = feat.shape[3]
                hr = wr = int(P_r ** 0.5)
                return feat.squeeze(1).reshape(B, D_r, hr, wr).repeat_interleave(T, dim=0)

            ref_s3 = _broadcast_ref(ref_features["stage_3"])
            ref_s2 = _broadcast_ref(ref_features["stage_2"])
            ref_s1 = _broadcast_ref(ref_features["stage_1"])

        # ── Spatial guide mask ──────────────────────────────────────────
        m_guidance = prev_mask
        if m_guidance is not None:
            if m_guidance.ndim == 5:
                m_guidance = m_guidance.reshape(B * T, -1, *m_guidance.shape[-2:])
                if m_guidance.shape[1] > 1:
                    m_guidance = m_guidance[:, 1:2]   # first object channel
            elif m_guidance.ndim == 3:
                m_guidance = m_guidance.unsqueeze(1).repeat_interleave(T, dim=0)
            m_guidance = m_guidance.float()

        r_mask = ref_mask
        if r_mask is not None:
            if r_mask.ndim == 5:
                r_mask = r_mask.reshape(B, 1, *r_mask.shape[-2:])
            elif r_mask.ndim == 3:
                r_mask = r_mask.unsqueeze(1)
            r_mask = r_mask.float().repeat_interleave(T, dim=0)

        # ── Top-down FPN fusion ─────────────────────────────────────────
        def execute_up(block, current_x, skip_feat, r_skip, r_mask_g):
            if isinstance(block, KANRefinedCrossAttentionUpBlock):
                return block(current_x, skip_feat, prev_mask=m_guidance, ref_skip=r_skip, ref_mask=r_mask_g)
            elif isinstance(block, KANSpatialGatingUpBlock):
                return block(current_x, skip_feat, prev_mask=m_guidance, ref_skip=r_skip)
            else:
                return block(current_x, skip_feat)

        # up1: 14×14 semantics + Stage-3 skip  → [BT, 256, 14×14]
        # up2: 14→28 first genuine upsample   + Stage-2 skip  → [BT, 128, 28×28]
        # up3: 28→56 second genuine upsample  + Stage-1 skip  → [BT,  64, 56×56]
        x = execute_up(self.up1, x, s3, ref_s3, r_mask)
        x = execute_up(self.up2, x, s2, ref_s2, r_mask)
        x = execute_up(self.up3, x, s1, ref_s1, r_mask)

        logits = self.final_head(x)   # [BT, num_classes, 56, 56]

        # Final upscale to target_size (e.g. 56→224).
        if logits.shape[-1] != self.target_size:
            logits = nn.functional.interpolate(
                logits,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        _, C, H, W = logits.shape
        return logits.reshape(B, T, C, H, W)

