"""Lightweight KAN-guided Vision-Mamba style encoder.

This encoder keeps the interface used by ``VideoMambaSystem``:

- Input:  ``[B, T, C, H, W]``
- Output:
  - ``cls_token``: ``[B, T, dim_out]``
  - ``features``: dict with tokenized multi-scale maps

    - ``stage_3``: projected main features (stride-16)
    - ``stage_2``: fine stride-16 features (pre-projection)
    - ``stage_1``: stride-8 features
    - ``stage_0``: stride-4 features

KAN guidance is provided by ``KangaSSM`` blocks (which internally use the
KAN-modulated SSM core) applied over spatial token sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kanga_ssm import KangaSSM


class _ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class _DepthwiseDownsample(nn.Module):
    """Efficient downsampling block used by the tiny encoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class _SpatialKangaBlock(nn.Module):
    """Apply KAN-SSM over spatial tokens of a single feature map."""

    def __init__(
        self,
        dim: int,
        d_state: int,
        layers: int,
        modulator_type: str,
    ) -> None:
        super().__init__()
        self.ssm = KangaSSM(
            d_model=dim,
            d_state=d_state,
            num_layers=layers,
            dropout=0.0,
            modulator_type=modulator_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        tokens = x.reshape(bt, c, h * w).transpose(1, 2)  # [BT, HW, C]
        tokens = self.ssm(tokens)
        return tokens.transpose(1, 2).reshape(bt, c, h, w)


class VisionMambaTinyWrapper(nn.Module):
    """Lightweight trainable Vision-Mamba encoder for VOS.

    The model keeps compute low by using depthwise-conv pyramid extraction and
    only two spatial KAN-SSM refinements at stride-16.
    """

    def __init__(
        self,
        out_dim: int = 192,
        *,
        stem_dim: int = 24,
        stage0_dim: int = 48,
        stage1_dim: int = 64,
        stage2_dim: int = 96,
        ssm_d_state: int = 8,
        ssm_layers: int = 1,
        modulator_type: str = "kan",
    ) -> None:
        super().__init__()

        if min(out_dim, stem_dim, stage0_dim, stage1_dim, stage2_dim) <= 0:
            raise ValueError("VisionMambaTinyWrapper dims must be positive.")

        # Public channel attributes used by configuration code.
        self.DIM_4X = stage0_dim
        self.DIM_8X = stage1_dim
        self.DIM_16X = stage2_dim
        self.DIM_OUT = out_dim

        self.stem = _ConvBNAct(3, stem_dim, stride=2)  # /2
        self.stage0 = _DepthwiseDownsample(stem_dim, stage0_dim)  # /4
        self.stage1 = _DepthwiseDownsample(stage0_dim, stage1_dim)  # /8
        self.stage2 = _DepthwiseDownsample(stage1_dim, stage2_dim)  # /16

        self.stage2_ssm = _SpatialKangaBlock(
            dim=stage2_dim,
            d_state=ssm_d_state,
            layers=ssm_layers,
            modulator_type=modulator_type,
        )

        self.project = nn.Sequential(
            nn.Conv2d(stage2_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

        self.stage3_ssm = _SpatialKangaBlock(
            dim=out_dim,
            d_state=ssm_d_state,
            layers=ssm_layers,
            modulator_type=modulator_type,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extract lightweight multi-scale features from clip frames."""
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)

        x = self.stem(x_flat)
        s4x = self.stage0(x)
        s8x = self.stage1(s4x)
        s16x = self.stage2(s8x)

        s16x = self.stage2_ssm(s16x)
        s_out = self.project(s16x)
        s_out = self.stage3_ssm(s_out)

        cls_token = s_out.mean(dim=(2, 3)).reshape(b, t, self.DIM_OUT)

        def _to_tokens_and_hw(feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
            bt, ch, hf, wf = feat.shape
            return feat.reshape(b, t, ch, hf * wf), (hf, wf)

        stage_3, stage_3_hw = _to_tokens_and_hw(s_out)
        stage_2, stage_2_hw = _to_tokens_and_hw(s16x)
        stage_1, stage_1_hw = _to_tokens_and_hw(s8x)
        stage_0, stage_0_hw = _to_tokens_and_hw(s4x)

        features = {
            "stage_3": stage_3,
            "stage_2": stage_2,
            "stage_1": stage_1,
            "stage_0": stage_0,
            "stage_3_hw": stage_3_hw,
            "stage_2_hw": stage_2_hw,
            "stage_1_hw": stage_1_hw,
            "stage_0_hw": stage_0_hw,
        }
        return cls_token, features
