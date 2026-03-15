# Video-Kamba: Lightweight KAN-Modulated State-Space Video Object Segmentation

> Technical overview for ACM Multimedia submission.

---

## 1. Problem Statement

**Video Object Segmentation (VOS)** requires tracking and precisely segmenting specific object
instances across video frames — given only a single annotated reference frame at test time.
Unlike semantic segmentation (which assigns class labels), VOS must distinguish *object A*
from *object B* even when they belong to the same semantic category.

**The core challenge**: propagating fine-grained instance masks through complex motion, occlusion,
and appearance change, without retraining per video — and doing so efficiently enough for
practical deployment.

**Our approach**: replace the large Vision Transformer backbone (ViT-B/16, ~86M parameters)
used in current SOTA methods (Cutie, DeAOT) with a fully-trainable **<3M parameter pipeline**
that combines depthwise-separable convolutions, spatial state-space models, and
KAN-modulated temporal memory — without sacrificing mask propagation quality.

---

## 2. Architecture Overview

```
Input Clip [B, T, 3, H, W]
         │
         ▼
┌─────────────────────────┐
│  Vision-Mamba Tiny      │  ← Depthwise pyramid + KAN-SSM refinement
│  Encoder (~2.0M params) │    Outputs 4 feature scales
└────────┬────────────────┘
         │  stage_3 [/16]  stage_2 [/16]  stage_1 [/8]  stage_0 [/4]
         │
         ▼
┌─────────────────────────┐
│  Memory Bank (K/V)      │  ← AOT-style explicit memory
│  + KAN Key Adapter      │    Coarse + Fine dual-scale keys
└────────┬────────────────┘
         │ cross-attention per frame
         ▼
┌─────────────────────────┐
│  Propagation Attention  │  ← Multi-head cross-attention (6 heads)
│  + KangaSSM (2 layers)  │    DiagonalKANSSMCore: O(N·D) recurrence
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Segmentation Decoder   │  ← 3-level FPN with KAN spatial gating
│  (~0.4M params)         │    Upsamples /16 → /2 → target resolution
└────────┬────────────────┘
         │
         ▼
  Mask Logits [B, T, K, H, W]    (K = num tracked objects)
```

**Total parameters**: ~2.8M  (cf. Cutie: ~42M, DeAOT: ~42M, MobileVOS: ~5M)

---

## 3. Key Components

### 3.1 Vision-Mamba Tiny Encoder

A fully-trainable, lightweight feature extractor built from scratch — no ImageNet pretrained
weights required.

**Architecture:**
```
Input [3, H, W]
  │
  ├─ Stem: Conv2d(3→24, stride=2)                    → /2
  ├─ Stage 0: DepthwiseSepConv(24→48, stride=2)      → /4   (48ch)
  ├─ Stage 1: DepthwiseSepConv(48→64, stride=2)      → /8   (64ch)
  ├─ Stage 2: DepthwiseSepConv(64→96, stride=2)      → /16  (96ch)
  ├─ Spatial KangaSSM (at /16)                        ← SSM refinement
  ├─ Projection: Conv1d(96→192)
  └─ Final Spatial KangaSSM                           → 192ch at /16
```

Multi-scale outputs:
| Scale | Resolution | Channels | Usage |
|-------|-----------|----------|-------|
| stage_3 | H/16 | 192 | Memory keys (coarse), SSM input |
| stage_2 | H/16 | 96  | Memory keys (fine), decoder skip |
| stage_1 | H/8  | 64  | Decoder skip (level 2) |
| stage_0 | H/4  | 48  | Decoder skip (level 3) |

---

### 3.2 DiagonalKANSSMCore — The Temporal Model

The central innovation. A **diagonal state-space model** where all three SSM matrices
(A, B, C) are **input-dependently modulated** by KAN (Kolmogorov–Arnold Network) functions.

**State update (per timestep, vectorized over T):**
```
delta_t  = softplus(W_δ · u_t)                          ← per-channel step size  [B, N]
alpha_A  = KAN_A(u_t)                                   ← decay modulation        [B, N]
A_bar_t  = exp(−exp(log_A) · delta_t · alpha_A)         ← diagonal decay          [B, N]

alpha_B  = KAN_B(u_t)                                   ← input-gate modulation   [B, N]
b_t      = alpha_B ⊙ (B · u_t)                         ← modulated state input    [B, N]

h_t      = A_bar_t ⊙ h_{t−1} + b_t                    ← recurrent update         [B, N]

alpha_C  = KAN_C(u_t)                                   ← readout modulation       [B, D]
y_t      = alpha_C ⊙ (C · h_t) + D_skip ⊙ u_t         ← output + skip connection [B, D]
```

**Why diagonal?**
- Classic Mamba uses dense [N×N] A matrices: O(N³) per update
- Diagonal A: O(N·D) per update — practical for video (T=16 frames)
- Expressiveness recovered via KAN modulators (data-driven gating on A, B, C)

**Why KAN modulators?**
- Standard SSMs use fixed input-independent A, B, C → poor at appearance change
- KAN (FastKAN with RBF grid) learns smooth, non-linear modulation functions
- A_modulator: controls how fast the memory *decays* (slow objects → slow decay)
- B_modulator: controls *what* gets written to state (focus on moving regions)
- C_modulator: controls *what* gets read from state (query-dependent readout)

**Learnable parameters:**
| Parameter | Shape | Role |
|-----------|-------|------|
| `log_A` | [N] | Diagonal decay rates (init: log(0.5)) |
| `B` | [N, D] | State-input projection |
| `C` | [D, N] | State-to-output projection |
| `D_skip` | [D] | Per-channel skip weights |
| `delta_proj` | Linear(D→N) | Step-size generator |
| `A/B/C_modulator` | FastKANLayer | Input-dependent gating |

---

### 3.3 Memory Bank & Propagation

AOT-style explicit memory with **dual-scale keys** and **KAN key adaptation**.

**Memory structure:**
- Stores up to `max_mem_frames=7` key-value pairs
- Reference frame is **anchored** (never evicted)
- Non-reference frames evict oldest when full (FIFO)

**Dual-scale keys** (novel contribution):
- Coarse keys at /16 (stage_3): semantic-level matching
- Fine keys at /16 from pre-projection (stage_2): spatial detail
- Concatenated for attention: [B, P_coarse + P_fine, d_key]
- Learnable **scale discriminators** prevent confusion between scales

**KAN Key Adapter:**
- Lightweight recurrent correction on non-reference keys
- Models cumulative appearance drift across frames
- Parameters: ~30k (KangaSSM in d_key=192 space only)
- Formula: `K_adapted = K + α · KangaSSM(K, state_t)`

**Propagation Attention:**
- Query: current frame features [B, P, D]
- Keys/Values: memory bank [B, M·P, d_key/d_val]
- Standard multi-head dot-product attention (6 heads, d_key=192)
- Output: context-enriched features, same shape as query

---

### 3.4 Segmentation Decoder

Three-level FPN decoder with **KAN spatial gating** at each level.

```
prop_feat [B·T, 192, H/16, W/16]
     │
     ├─ up1: KANSpatialGatingUpBlock  → [B·T, 160, H/16, W/16]  (semantic refinement)
     │        skip: stage_3 (192ch)
     ├─ up2: KANSpatialGatingUpBlock  → [B·T,  96, H/8,  W/8 ]  (first upsample ×2)
     │        skip: stage_2 (96ch)
     ├─ up3: KANSpatialGatingUpBlock  → [B·T,  48, H/4,  W/4 ]  (second upsample ×2)
     │        skip: stage_1 (64ch)
     └─ head: Conv + bilinear upsample → [B·T, K, H, W]
```

**KANSpatialGatingUpBlock:**
1. ConvTranspose2d ×2 (or identity at up1)
2. 1×1 contextual projection on skip features
3. FastKAN pixel-wise gating: `gate = KAN(x)` ∈ [0,1]
4. Fusion: `out = conv(x_proj + gate ⊙ skip)`

---

### 3.5 Scheduled Sampling (Curriculum Learning)

During training, the memory bank can be updated with either:
- **Ground-truth masks** (teacher forcing): stable gradients early in training
- **Predicted soft-masks** (self-teaching): forces robustness to own errors

**Schedule:**
```
rate(epoch) = linear_ramp(start=0.0, end=0.25, over=50 epochs, warmup=5)
```
At epoch 5 (post-warmup): always use GT masks (rate=0.0)
At epoch 50: 25% of frames use predicted masks as memory input

This mirrors how the model behaves at inference — where GT masks are unavailable
after the reference frame.

---

## 4. Training Protocol

### 4.1 Three-Phase Pipeline

Following Cutie (CVPR 2024): eliminate large synthetic pre-training (BL30K, ~700GB),
use only real instance masks.

```
Phase 1 — COCO Static Pre-training           (~15h, 1 GPU)
  Script  : scripts/pretrain_coco_sota_light.sh
  Data    : COCO 2017 (118k images, instance masks)
  Epochs  : 20  |  limit_train_batches=3000
  LR      : 1e-4 (AdamW, cosine schedule)
  Batch   : 4 × accum=2 = eff. 8
  Goal    : Learn instance-mask understanding from static images
  Output  : checkpoints/best_coco_sota_light.ckpt

              ↓ (shape-safe partial weight transfer)

Phase 2 — DAVIS Fast Validation Gate         (~16h, 1 GPU)
  Script  : scripts/ft_davis_fast_validate.sh
  Data    : DAVIS 2017 (60 train sequences, ~301 clips)
  Epochs  : 100  |  limit_train_batches=500 (~12 min/epoch)
  LR      : 4e-5 (AdamW, cosine schedule)
  Goal    : Early J&F signal to validate architecture before full training
  GATE    : val_J_and_F > 0.55 @ epoch 50 → proceed to Phase 3

              ↓ (if gate passed)

Phase 3 — YouTube-VOS + DAVIS Joint Fine-tune (~71h, 1 GPU)
  Script  : scripts/ft_coco_ytb_dav_50ep.sh
  Data    : YouTube-VOS (3471 seqs) + DAVIS (60 seqs), 25%/75% weighted sampler
  Epochs  : 50  |  limit_train_batches=3500 (~1.4h/epoch)
  LR      : 4e-5 (AdamW, cosine schedule)
  Goal    : DAVIS J&F > 0.65, competitive with MobileVOS (~78% J&F)
  Output  : checkpoints/best_ytbdav_sota_light.ckpt
```

### 4.2 Data Curriculum Rationale

| Stage | Dataset | Mask Type | Why |
|-------|---------|-----------|-----|
| COCO | 118k static images | Instance (polygon/RLE) | Align feature space to instance-level segmentation |
| DAVIS | 60 dense sequences | Semi-supervised VOS | Fast convergence signal; dense annotations, hard scenes |
| YTB+DAV joint | 3531 sequences | Semi-supervised VOS | Scale + diversity for generalization |

### 4.3 Optimizer Configuration

- **Optimizer**: AdamW, `weight_decay=1e-2`
- **LR groups**:
  - Propagation/SSM modules: `lr × 5.0` (faster adaptation)
  - Trainable backbone: `lr × 0.1`
- **Schedule**: Cosine annealing, `eta_min=1e-6`
- **Gradient clipping**: `grad_norm=1.0` (prevents SSM state explosion in T=16 loops)
- **Mixed precision**: `16-mixed` (AMP)

---

## 5. Benchmark Targets

| Model | Params | DAVIS J&F | YTB-VOS J&F | Notes |
|-------|--------|-----------|------------|-------|
| DeAOT (NIPS 2022) | ~42M | 79.4 | 80.4 | Heavy ViT backbone |
| Cutie (CVPR 2024) | ~42M | 84.3 | 82.5 | COCO→YTB+DAV, no BL30K |
| MobileVOS (CVPR 2023) | ~5M | 77.8 | — | Best published lightweight VOS |
| **Video-Kamba (ours)** | **~2.8M** | *target: >0.65* | *TBD* | DiagonalKAN-SSM + KAN memory |

---

## 6. Key Contributions Summary

1. **DiagonalKANSSMCore**: Input-dependent KAN modulation of all three SSM matrices
   (A, B, C) in a diagonal state-space model — balances expressiveness and O(N·D) efficiency.

2. **KAN Key Adapter**: Recurrent per-frame correction of memory-bank keys
   using a lightweight KangaSSM in key-space only (~30k parameters).

3. **Dual-scale memory keys**: Coarse (semantic) + Fine (spatial) key concatenation
   with learnable scale discriminators — better spatial localization without ViT.

4. **Fully trainable <3M pipeline**: No frozen ImageNet/DINOv2 backbone required —
   entire network trained end-to-end from COCO → YTB+DAV in two SLURM jobs.

5. **Scheduled sampling curriculum**: Linear ramp from teacher-forcing to self-teaching
   during YTB+DAV fine-tune — closes train/test domain gap in memory updates.

---

## 7. Repository Structure

```
video_kamba/
├── configs/
│   ├── model/
│   │   └── vision_mamba_tiny_sota_light.yaml   ← Main model config
│   ├── datamodule/
│   │   ├── coco_pretrain.yaml
│   │   ├── davis.yaml
│   │   └── ytv_dav_joint.yaml
│   └── trainer/default.yaml
├── models/
│   ├── video_mamba.py                          ← VideoMambaSystem (LightningModule)
│   └── components/
│       ├── vision_mamba_tiny_wrapper.py        ← Encoder
│       ├── kan_ssm_core.py                     ← DiagonalKANSSMCore
│       ├── kanga_ssm.py                        ← KangaSSM (stacked wrapper)
│       ├── memory_bank.py                      ← Dual-scale K/V memory
│       ├── kan_key_adapter.py                  ← Recurrent key correction
│       ├── propagation_attention.py            ← Cross-attention to memory
│       └── segmentation_decoder.py             ← FPN + KAN gating
├── data/
│   ├── coco_pretrain.py
│   ├── davis.py
│   └── vos_datamodule.py                       ← YouTubeVOS + DAVIS joint
├── utils/
│   └── vos_loss.py                             ← HybridVOSLoss (CE + SoftDice)
├── train.py                                    ← Hydra entry point
├── scripts/
│   ├── pretrain_coco_sota_light.sh             ← Phase 1 SLURM job
│   ├── ft_davis_fast_validate.sh               ← Phase 2 gate job
│   └── ft_coco_ytb_dav_50ep.sh                 ← Phase 3 full fine-tune
└── TRAINING_PROTOCOL.md                        ← Submission sequence + gates
```
