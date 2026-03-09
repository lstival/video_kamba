# MobileNetV2 Encoder — Change Summary

> **Branch**: `fix/mobilenetv2-encoder`  
> **Date**: March 9, 2026  
> **Goal**: Replace the frozen DINOv2 ViT-B/14 backbone with a trainable
> AOT-style MobileNetV2 encoder to align our architecture more closely with
> the AOTT SOTA baseline (J&F = 83.3% on DAVIS 2017 val).

---

## Motivation

From `docs/vos_propagation_analysis.md` §11.3 and §5, the two primary structural
gaps between our model and AOTT are:

1. **Single-scale decoder skips** — DINOv2 outputs all three "skip" tensors at
   the same 14×14 resolution; the FPN decoder never sees genuine higher-resolution
   features.
2. **Frozen backbone** — DINOv2 weights cannot be fine-tuned for VOS;
   MobileNetV2 is fully trainable end-to-end.

AOT's actual decoder receives shortcut features at 4×, 8×, and 16× strides
(`[24, 32, 96]` channels). This change brings our pipeline to parity on both
counts.

---

## Files Changed

### 1. `models/components/mobilenetv2_wrapper.py` *(new)*

Self-contained AOT-compatible MobileNetV2 encoder. **Does NOT import from
`sota/aot-benchmark/`** (avoided due to a `utils` package name collision).

**Key design:**
- `_AOTMobileNetV2(output_stride=16)` — inverted-residual backbone with
  dilation applied once `current_stride == output_stride`, so all main
  features stay at stride 16 (14×14 at 224p, 30×30 at 480p — matching AOTT).
- `encoder_projector` — `Conv2d(1280 → 256, k=1)` identical to AOT's
  `encoder_projector`.
- Loads **torchvision ImageNet-1K** pretrained weights automatically on init.

**Output tensor map:**

| Key | Channels | Spatial (224p) | Downstream use |
|---|---|---|---|
| `stage_3` | 256 | 14×14 = 196 tokens | SSM input, PropagationAttention Q/K/V |
| `stage_2` | 96  | 14×14 = 196 tokens | Dual-scale MemoryBank fine key + decoder up1 skip |
| `stage_1` | 32  | 28×28 = 784 tokens | Decoder up2 skip (first genuine spatial upsample) |
| `stage_0` | 24  | 56×56 = 3136 tokens| Decoder up3 skip (second genuine spatial upsample) |

Forward signature:
```python
cls_token, features = MobileNetV2Wrapper()(x)  # x: [B, T, C, H, W]
# cls_token: [B, T, 256]
# features: dict as above
```

---

### 2. `models/video_mamba.py` *(updated)*

#### New constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `encoder_type` | str | `"dino"` | `"dino"` or `"mobilenetv2"` |
| `dim_in_fine` | int | `-1` (→ `dim_in`) | Fine-scale dim for MemoryBank dual-scale key |
| `dim_in_s2` | int | `-1` (→ `dim_in`) | Decoder up2 skip channels |
| `dim_in_s1` | int | `-1` (→ `dim_in`) | Decoder up3 skip channels |
| `mv2_output_stride` | int | `16` | MobileNetV2 output stride |
| `mv2_freeze_at` | int | `0` | Freeze first N backbone stages |
| `mv2_pretrained` | bool | `True` | Load ImageNet weights |

#### New helper methods

- `_map_ms_features(raw)` — maps raw encoder output to canonical
  `{"stage_3", "stage_2", "stage_1"}` keys for SSM/decoder:
  - DINOv2: `layer_11 / layer_9 / layer_6`
  - MobileNetV2: `stage_3 / stage_1 / stage_0`

- `_get_fine_features(raw, t)` — returns fine-scale tokens for MemoryBank
  dual-scale key (DINOv2: `layer_9`; MobileNetV2: `stage_2`).

- `_build_dec_ms(raw, t)` — builds decoder skip dict for frame `t`;
  for MobileNetV2 this routes MV2-native multi-scale features:
  - up1 skip: `stage_2` (96-ch, 14×14 — same-scale semantic refinement)
  - up2 skip: `stage_1` (32-ch, 28×28 — first genuine spatial upsample)
  - up3 skip: `stage_0` (24-ch, 56×56 — second genuine spatial upsample)

#### `forward()` changes
All encoder-specific feature routing is now handled through the three helpers
above, making the main loop fully encoder-agnostic. No logic changes to the
propagation / SSM / memory-bank flow.

**Backward compatibility**: `encoder_type="dino"` (the default) is 100%
identical to the previous behaviour — no existing checkpoints are affected.

---

### 3. `configs/model/mobilenetv2.yaml` *(new)*

```yaml
encoder_type: mobilenetv2
dim_in: 256
dim_in_fine: 96
dim_in_s2: 32
dim_in_s1: 24
mv2_output_stride: 16
mv2_freeze_at: 0
mv2_pretrained: true
target_size: 480        # AOT native resolution → 30×30 = 900 tokens
```

---

## How to Use

```bash
# Train with MobileNetV2 encoder (AOT-style)
python train.py model=mobilenetv2

# Train at 224p (faster, debugging)
python train.py model=mobilenetv2 model.target_size=224

# Train at 448p (good quality/speed balance)
python train.py model=mobilenetv2 model.target_size=448

# Keep DINO (unchanged)
python train.py model=default
```

---

## Architecture Alignment vs AOTT

| Dimension | AOTT | Our DINOv2 | Our MobileNetV2 (this branch) |
|---|---|---|---|
| Backbone | MobileNetV2 (trainable) | DINOv2 (frozen) | MobileNetV2 (trainable) ✓ |
| Feature dim | 256 | 768 | 256 ✓ |
| Stride-16 tokens | 900 (30×30 at 480p) | 196 (14×14 at 224p) | 900 at 480p ✓ |
| Decoder skip scales | 3 genuine (4×/8×/16×) | 3 same-scale (14×14) | 3 genuine ✓ |
| Trainable encoder | Yes | No | Yes ✓ |
| Pretrained init | ImageNet | Yes (hub) | ImageNet ✓ |

---

## Remaining Gaps to Close

From `docs/vos_propagation_analysis.md` §10 (Action Plan):

| Priority | Fix | Expected Gain |
|---|---|---|
| **Next** | Train on YouTube-VOS + DAVIS (30× more data) | +10–15% J&F |
| Next | Stack 2–3 PropagationAttention layers | +2–3% J&F |
| Next | Increase SSM state_dim 16 → 64 | +1–2% J&F |
| Later | Object-aware KAN conditioning | novel contribution |

---

## Validation

Tested locally (CPU):
```
MobileNetV2Wrapper: stage_3(1,2,256,196) stage_2(1,2,96,196) stage_1(1,2,32,784) stage_0(1,2,24,3136)
VideoMambaSystem(encoder_type='mobilenetv2'): clf(1,51) | seg(1,3,11,224,224)  OK
VideoMambaSystem(encoder_type='dino'):        clf(1,51) | seg(1,2,11,224,224)  OK (backward compat)
```
