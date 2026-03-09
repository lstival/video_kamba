"""MobileNetV2 encoder wrapper — AOT-compatible backbone.

Wraps a self-contained MobileNetV2 implementation (matching AOT's encoder with
``output_stride`` dilation support) and returns multi-scale spatial features
that are compatible with our downstream modules (MemoryBank,
PropagationAttention, KangaSSM, SegmentationDecoder).

Architecture (output_stride=16, 224x224 input -> 14x14 tokens at stride-16):

    Stage  Stride  Channels  Tokens (224p)   Key
    -----  ------  --------  -------------   --------
    4x       4       24       56x56 = 3136   "stage_0"
    8x       8       32       28x28 =  784   "stage_1"
    16x     16       96       14x14 =  196   "stage_2"  (pre-projection)
    proj    16      256       14x14 =  196   "stage_3"  <- SSM input

Usage in VideoMambaSystem:
  - SSM / PropagationAttention input  -> "stage_3"  (256-ch)
  - Dual-scale fine memory-bank key   -> "stage_2"  ( 96-ch, same spatial as stage_3)
  - SegmentationDecoder up2 skip      -> "stage_1"  ( 32-ch, 28x28 genuine upsample)
  - SegmentationDecoder up3 skip      -> "stage_0"  ( 24-ch, 56x56 genuine upsample)

Note: The MobileNetV2 building blocks are inlined here rather than imported from
sota/aot-benchmark/ to avoid a name collision between our own utils/ package and
the aot-benchmark/utils/ package.

Reference:
    Yang et al., "AOT: Associating Objects with Transformers for VOS",
    NeurIPS 2021.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MobileNetV2 building blocks (adapted from AOT's encoder implementation)
# ---------------------------------------------------------------------------

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class _ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: int = -1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=False,
            ),
            (norm_layer or nn.BatchNorm2d)(out_planes),
            (activation_layer or nn.ReLU6)(inplace=True),
        )
        self.out_channels = out_planes


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        dilation: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert stride in (1, 2)
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(_ConvBNAct(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            _ConvBNAct(
                hidden_dim, hidden_dim,
                stride=stride, dilation=dilation,
                groups=hidden_dim, norm_layer=norm_layer,
            ),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            (norm_layer or nn.BatchNorm2d)(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class _AOTMobileNetV2(nn.Module):
    """MobileNetV2 with selectable output_stride (mirrors AOT's encoder)."""

    def __init__(
        self,
        output_stride: int = 16,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        width_mult: float = 1.0,
        freeze_at: int = 0,
    ) -> None:
        super().__init__()

        inverted_residual_setting = [
            # t,   c,  n, s
            [1,   16,  1, 1],
            [6,   24,  2, 2],
            [6,   32,  3, 2],
            [6,   64,  4, 2],
            [6,   96,  3, 1],
            [6,  160,  3, 2],
            [6,  320,  1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 8)
        current_stride = 1
        rate = 1

        features: List[nn.Module] = [
            _ConvBNAct(3, input_channel, stride=2, norm_layer=norm_layer)
        ]
        current_stride *= 2

        for t, c, n, s in inverted_residual_setting:
            if current_stride == output_stride:
                blk_stride, blk_dil = 1, rate
                rate *= s
            else:
                blk_stride, blk_dil = s, 1
                current_stride *= s

            out_ch = _make_divisible(c * width_mult, 8)
            for i in range(n):
                features.append(
                    _InvertedResidual(
                        input_channel, out_ch,
                        stride=(blk_stride if i == 0 else 1),
                        dilation=(blk_dil if i == 0 else rate),
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = out_ch

        last_ch = _make_divisible(int(1280 * max(1.0, width_mult)), 8)
        features.append(_ConvBNAct(input_channel, last_ch, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)

        # AOT-style stage splits
        self.stages = [
            self.features[0:4],    # 24-ch, stride 4
            self.features[4:7],    # 32-ch, stride 8
            self.features[7:14],   # 96-ch, stride 16
            self.features[14:],    # 1280-ch, stride 16 (dilated when output_stride=16)
        ]

        self._init_weights()
        self._apply_freeze(freeze_at)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _apply_freeze(self, freeze_at: int) -> None:
        if freeze_at >= 1:
            _freeze(self.stages[0])
        for i, stage in enumerate(self.stages[1:], start=2):
            if freeze_at >= i:
                _freeze(stage)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns [s4x, s8x, s16x, s32x(dilated)]."""
        xs = []
        for stage in self.stages:
            x = stage(x)
            xs.append(x)
        return xs


# ---------------------------------------------------------------------------
# Public Wrapper
# ---------------------------------------------------------------------------

class MobileNetV2Wrapper(nn.Module):
    """AOT-style MobileNetV2 encoder wrapper for our VOS pipeline.

    Args:
        output_stride: Spatial output stride of the last feature map.
                       16 (AOT default) keeps main features at stride 16
                       via dilated convolutions, giving 14x14 tokens at 224p.
        freeze_at:     Freeze backbone stages 1..freeze_at (0 = fully trainable).
        pretrained:    Bootstrap backbone from torchvision ImageNet-1K weights.
    """

    DIM_4X  = 24    # stride-4  feature channels
    DIM_8X  = 32    # stride-8  feature channels
    DIM_16X = 96    # stride-16 feature channels (pre-projection)
    DIM_OUT = 256   # projected dim (matches AOT MODEL_ENCODER_EMBEDDING_DIM)

    def __init__(
        self,
        output_stride: int = 16,
        freeze_at: int = 0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = _AOTMobileNetV2(output_stride=output_stride, freeze_at=freeze_at)
        # Project deepest stage (1280-ch) to 256-ch — mirrors AOT encoder_projector
        self.encoder_projector = nn.Conv2d(1280, self.DIM_OUT, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.encoder_projector.weight)

        if pretrained:
            self._load_imagenet_weights()

    def _load_imagenet_weights(self) -> None:
        """Bootstrap backbone weights from torchvision ImageNet-1K MobileNetV2."""
        try:
            import torchvision.models as tv_models
            tv_mv2 = tv_models.mobilenet_v2(weights="IMAGENET1K_V1")
            tv_state = tv_mv2.features.state_dict()
            bb_state = self.backbone.features.state_dict()
            # Load every tensor whose name AND shape matches.
            # Dilation / stride changes don't alter parameter shapes so all
            # pre-trained weights load correctly even with output_stride=16.
            matched = {
                k: v for k, v in tv_state.items()
                if k in bb_state and v.shape == bb_state[k].shape
            }
            bb_state.update(matched)
            self.backbone.features.load_state_dict(bb_state, strict=False)
            print(
                f"[MobileNetV2Wrapper] ImageNet pretrained: "
                f"{len(matched)}/{len(bb_state)} param tensors loaded."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[MobileNetV2Wrapper] Warning -- pretrained weights not loaded: {exc}")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extract multi-scale features from a batch of video clips.

        Args:
            x: [B, T, C, H, W] float tensor (ImageNet normalisation recommended).

        Returns:
            cls_token:  [B, T, 256]  global-average-pooled stage_3.
            features:   dict of [B, T, Ch, P] token tensors::

                "stage_3" -> [B, T, 256, (H/16)*(W/16)]  projected  (SSM input)
                "stage_2" -> [B, T,  96, (H/16)*(W/16)]  pre-proj   (fine mem / up1 skip)
                "stage_1" -> [B, T,  32, (H/ 8)*(W/ 8)]             (decoder up2 skip)
                "stage_0" -> [B, T,  24, (H/ 4)*(W/ 4)]             (decoder up3 skip)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        s4x, s8x, s16x, s32x = self.backbone(x_flat)
        s256 = self.encoder_projector(s32x)  # [BT, 256, H/16, W/16]

        cls_token = s256.mean(dim=[2, 3]).view(B, T, self.DIM_OUT)

        def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
            BT_, Ch, Hf, Wf = feat.shape
            return feat.view(B, T, Ch, Hf * Wf)

        features = {
            "stage_3": _to_tokens(s256),
            "stage_2": _to_tokens(s16x),
            "stage_1": _to_tokens(s8x),
            "stage_0": _to_tokens(s4x),
        }
        return cls_token, features
