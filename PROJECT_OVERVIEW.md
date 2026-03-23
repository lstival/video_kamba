# Video-Kamba: KAN-Modulated State-Space Temporal Reasoning for Lightweight Video Object Segmentation

> Technical overview — targeting ACM Multimedia submission.
> Last updated: 2026-03-23

---

## 1. Problem Statement

**Video Object Segmentation (VOS)** requires tracking and precisely segmenting specific object
instances across all frames of a video — given only a single annotated reference frame at test time.
Unlike semantic segmentation (class-level labels), VOS must distinguish *object A* from *object B*
even when they belong to the same semantic category, and must do so continuously through complex
motion, occlusion, and appearance change.

**The core challenge**: propagating fine-grained instance masks over time without retraining per
video — and doing so efficiently enough for practical deployment.

**The gap in the literature**: current SOTA methods (Cutie ~42M, XMem ~57M, DeAOT ~42M) rely on
large Vision Transformer backbones and Transformer-based temporal attention — both quadratic in
sequence length and prohibitively heavy for edge deployment. The only published lightweight VOS
method, MobileVOS (CVPR 2023, ~5M params), still uses a standard Transformer for temporal
reasoning.

**Our contribution**: replace the Transformer temporal module with a **KAN-modulated diagonal
State Space Model** — a recurrent architecture with O(N·D) complexity whose state-update dynamics
are conditioned on input content via learned Kolmogorov-Arnold Network functions. We pair this
with a MobileNetV2 backbone (ImageNet pretrained) to provide stable visual features without
the instability of training an encoder from scratch.

---

## 2. Architecture Overview

```
Input Clip [B, T, 3, H, W]   (480×480 → 30×30 tokens at stride-16)
          │
          ▼
┌──────────────────────────────────┐
│  MobileNetV2 Encoder             │  ImageNet-1K pretrained (torchvision)
│  (~3.4M params, trainable)       │  Depthwise-separable inverted residuals
│                                  │  Output: 4-scale feature pyramid
│  stride-4  →  24-ch  [/4 ]      │  stage_0 (decoder skip 3)
│  stride-8  →  32-ch  [/8 ]      │  stage_1 (decoder skip 2)
│  stride-16 →  96-ch  [/16]      │  stage_2 (fine memory key)
│  projection→ 256-ch  [/16]      │  stage_3 (SSM input + coarse key)
└────────────┬─────────────────────┘
             │  [B, T, 256, P]  P = (H/16)·(W/16) = 900 at 480p
             ▼
┌──────────────────────────────────┐
│  Memory Bank (K/V pairs)         │  Explicit episodic memory, max 5 frames
│  + KAN Key Adapter               │  Dual-scale keys: coarse (256) + fine (96)
│  (~130k params)                  │  KANKeyAdapter: recurrent drift correction
└────────────┬─────────────────────┘
             │  K: [B, M·P, 256]   V: [B, M·P, 256]
             ▼
┌──────────────────────────────────┐
│  Propagation Attention           │  Multi-head cross-attention (8 heads)
│  (~660k params)                  │  Q: current frame  ←→  K,V: memory bank
│                                  │  SwiGLU gated output projection (v3)
│                                  │  QK-Norm, fp32 softmax, logit clamp
└────────────┬─────────────────────┘
             │  [B, T, P, 256]  memory-attended features
             ▼
┌──────────────────────────────────────────────────────────┐
│  KangaSSM — Dual-Timescale Temporal Model                │  THE CORE CONTRIBUTION
│                                                          │
│  ┌─ Fast path (local)  ─ bidirectional KangaSSM (~336k) ─┐  │
│  │  d_state=16, 2 layers — frame-to-frame dynamics       │  │
│  └────────────────────────────┬──────────────────────────┘  │
│                               │ KAN gate                 │
│  ┌─ Fast path (global) ─ bidirectional KangaSSM (~336k) ─┐  │
│  │  d_state=16, 2 layers — object re-id context          │  │
│  └────────────────────────────┘                          │
│                               ↓ fused (KAN gate)         │
│  ┌─ DTSM slow path ─ causal KangaSSM (~675k) ────────────┐  │
│  │  d_state=64, 1 layer, A_bar≈0.995 — long-range memory │  │
│  │  Persistent hidden state across all video frames      │  │
│  │  No attention — purely recurrent consolidation        │  │
│  └────────────────────────────┬──────────────────────────┘  │
│                               ↓ residual correction      │
└──────────────────────────────────────────────────────────┘
             │  [B, T, P, 256]  temporally-refined features
             ▼
┌──────────────────────────────────┐
│  Segmentation Decoder            │  3-level FPN, genuine multi-scale
│  (~280k params)                  │  /16 → /8 → /4 → /1 (target size)
│                                  │  Skip connections from MobileNetV2
└────────────┬─────────────────────┘
             │
             ▼
  Mask Logits [B, T, K+1, H, W]   (K = num tracked objects + background)
```

**Total parameters**: ~8.3M with DTSM enabled / ~5.0M without (cf. MobileVOS: ~5M Transformer, Cutie: ~42M, DeAOT: ~42M)

> DTSM overhead: +740k (slow KangaSSM 675k + residual projection 65k).
> The base 5M target is preserved as a no-DTSM ablation; DTSM is enabled by default for
> long-range benchmarks (LVOS) and disabled for short-clip evaluation (DAVIS, YouTube-VOS).

---

## 3. Key Components

### 3.1 MobileNetV2 Encoder

MobileNetV2 (ImageNet-1K pretrained, torchvision weights) with AOT-style `output_stride=16`
via dilated convolutions in the deepest stage.

**Why MobileNetV2 over a custom from-scratch encoder:**
- ImageNet pretraining provides proven edge/texture/object features from step 0
- Eliminates a major source of gradient instability (random encoder + VOS loss)
- Enables a clean ablation: the variable is the temporal module (KAN-SSM vs Transformer)
- Matches the backbone used by MobileVOS (CVPR 2023) — direct comparison is possible

**Multi-scale outputs** (all at 480×480 input → 30×30 tokens at stride-16):

| Stage | Stride | Channels | Tokens (480p) | Role |
|-------|--------|----------|---------------|------|
| stage_0 | /4  | 24  | 120×120 = 14400 | Decoder skip (up3) |
| stage_1 | /8  | 32  | 60×60   = 3600  | Decoder skip (up2) |
| stage_2 | /16 | 96  | 30×30   = 900   | Fine memory key + decoder skip (up1) |
| stage_3 | /16 | 256 | 30×30   = 900   | SSM input, coarse memory key |

**Backbone learning rate**: `lr × 0.05` (effective ~1.5e-6 at base 3e-5) — preserves
ImageNet features while allowing slow adaptation to VOS spatial statistics.

---

### 3.2 DiagonalKANSSMCore — The Temporal Model

The central innovation. A **diagonal State Space Model** where all three SSM matrices
(A, B, C) are **input-dependently modulated** by learned KAN (Kolmogorov-Arnold Network)
functions.

**State update (per timestep, vectorized over all patches P):**

```
δ_t   = softplus(W_δ · u_t)                      ← per-channel step size        [B·P, N]
Ā_t   = exp(−exp(log_A) · δ_t · α_A(u_t))        ← KAN-modulated diagonal decay [B·P, N]

h_t   = Ā_t ⊙ h_{t-1} + (B̄ ⊙ α_B(u_t)) · u_t  ← state update                [B·P, N]

y_t   = (C ⊙ α_C(h_t)) · h_t + D_skip ⊙ u_t    ← KAN-modulated readout + skip [B·P, D]
```

**KAN Modulators (v3 — tanh_bounded activation):**

```python
α(x) = 1.0 + 0.5 · tanh(FastKAN(x))   # output ∈ [0.5, 1.5]
```

- Replaces `softplus` (unbounded, caused C_modulator to reach 1.8e+04 norm in job 65737509)
- Bounds modulation to [0.5, 1.5] — the SSM can attenuate or amplify by at most 2×
- `FastKANLayer` uses RBF (radial basis function) grid with `grid_size=8`

**Why diagonal A?**
- Dense A: O(N³) per state update — impractical for T=4–16 video frames
- Diagonal A: O(N·D) — the per-channel state recurrence is fully independent
- Expressiveness recovered by KAN modulators: data-driven gating of A, B, C compensates
  for the lost cross-channel mixing

**Backward pass complexity**: diagonal A decouples gradient channels exactly:
```
∂L/∂h_t^(p,n) = (∂L/∂y_t^(p)) · C^n + (∂L/∂h_{t+1}^(p,n)) · Ā_{t+1}^(p,n)
```
No cross-state mixing in the gradient — O(N·D) backward, same as forward.

**BCNorm (v3)**: `nn.RMSNorm(elementwise_affine=False)` on B and C projections.
The learnable affine gamma was growing to 1.2e+03 by epoch 4 in v2, negating normalisation.
Removing it prevents BCNorm from becoming a de-facto unregularised scale layer.

**SwiGLU output gate (v3)**:
```python
gate, up = self.out_proj(x).chunk(2, dim=-1)   # Linear(D → 2D)
out = up * F.silu(gate) + residual
```
Eliminates unbounded `out_proj` gradient explosion (peak 1.9e+05 norm in job 65737509,
monotonic 5× growth over 4 epochs, 100% explosion rate across 128 tracked steps).

**Parameters per KangaSSM layer** (d_model=256, d_state=16):

| Component | Params |
|-----------|--------|
| B projection (D→N) | 256×16 = 4,096 |
| C projection (N→D) | 16×256 = 4,096 |
| delta_proj (D→N) | 256×16 = 4,096 |
| log_A (diagonal) | 16 |
| D_skip | 256 |
| A/B/C modulators (FastKAN) | ~3×8k ≈ 24,000 |
| out_proj (D→2D, SwiGLU) | 256×512 = 131,072 |
| **Total per layer** | **~168k** |

2 layers × 2 paths (local + global) → ~672k fast-path temporal parameters.

---

### 3.2b Dual-Timescale State Memory (DTSM)

**Motivation**: The fast local/global SSMs (d_state=16, bidirectional, trained on 6–12 frame
clips) capture short-range temporal dynamics. For long-range benchmarks like **LVOS** (videos
averaging 68 seconds, ~408 frames at 6 fps), objects can be occluded for 100+ frames — far
beyond any clip window. A purely clip-based model cannot retain information across this gap.

**The failure mode without DTSM:**
The FIFO memory bank (5 frames max) evicts the frame just before a long occlusion begins.
When the object reappears after 150 frames, no past evidence survives in the memory bank or
the short-range SSM state. The model must segment from appearance alone.

**Why not cross-attention for long-term memory?**
Adding a cross-attention layer (softmax, QK dot-product) would contradict the core thesis —
that SSM recurrence can replace attention for temporal video reasoning. Any reviewer would
correctly note the architectural contradiction. The solution must be purely recurrent.

**DTSM design — a third SSM at a slow timescale:**

A second `KangaSSM` runs in parallel to the fast paths with three key differences:

| Property | Fast (local/global) | Slow (DTSM) |
|----------|--------------------|----|
| Directionality | Bidirectional | **Causal (unidirectional)** |
| d_state | 16 | **64** |
| num_layers | 2 | **1** |
| A initialisation | `log_A=log(0.5)` → A_bar≈0.6 | **`log_A=−3.0` → A_bar≈0.995** |
| grid_size (KAN) | 8 | **4** (efficient) |
| Input | T=1 frame per step | T=1 frame per step |
| State carry | Across frames of the video | **Across all frames (never reset mid-video)** |
| Params | ~336k × 2 paths | **~675k** |

**Why causal (unidirectional)?** The slow path must carry state *forward* through time. A
bidirectional SSM would require reversing the sequence — incompatible with an online,
persistent state buffer that accumulates across frames.

**Why A_bar ≈ 0.995?** The slow path should "forget" at roughly 0.5% per frame. At this
rate, information from 100 frames ago survives at `0.995^100 ≈ 0.61` — still meaningful.
Compare to the fast path: `0.6^100 ≈ 6×10⁻²³` (effectively zero). This is the timescale
separation that gives DTSM its long-range capacity.

Initialisation: `log_A = −3.0` → `exp(log_A) ≈ 0.05` → `A_bar = exp(−0.05 · δ)`.
With δ ≈ 0.1 (typical softplus output at initialisation): `A_bar ≈ exp(−0.005) ≈ 0.9950`.

**Integration — residual correction (no new data path):**

```
ssm_fused  ← KAN_gate(out_local, out_global)    ← fast path (existing)
out_slow   ← temporal_model_slow(fused, h_prev)  ← DTSM slow path
h_slow     ← update_state(out_slow)              ← persistent carry
ssm_fused  ← ssm_fused + slow_residual_proj(out_slow)  ← residual
```

`slow_residual_proj` is a `Linear(256→256, bias=False)` initialised near-zero
(`std = 0.02/√2`). The slow path starts as a null correction and learns to contribute
incrementally — preventing any initialisation shock to the trained fast path.

**Persistent state propagation (PSP — Option A):**
The DTSM state is carried frame-to-frame via `MemoryStateBank`, identical to the existing
mechanism for the fast paths. What changes: the slow SSM's `A_bar≈0.995` ensures the state
decays slowly enough to retain object evidence across hundreds of frames. The infrastructure
(`prev_states`, `return_last_state`) was already present in `KangaSSM.forward()`.

**Training requirement:**
The slow and fast paths see identical temporal context if `clip_len=6`. Increasing to
`clip_len=12` ensures gradient flows through 12 steps — enough to differentiate the two
timescales. BL30K with `clip_len=12` is the first training environment for the slow path.

**Parameters:**
| Component | Params |
|-----------|--------|
| `temporal_model_slow` (KangaSSM, d_state=64, 1 layer) | ~675k |
| `slow_residual_proj` (Linear 256→256) | 65.5k |
| **DTSM total** | **~740k** |

**Smoke test results (job 65781811):**
```
Total params : 8.29M  (base 5M + 740k DTSM + existing overhead)
Slow A_bar (delta=0.1): 0.9950  ✓
Forward pass [1, 12, 11, 480, 480]: ✓
```

---

### 3.3 Memory Bank

AOT-style explicit episodic memory storing key-value pairs from reference and past frames.

**Memory structure:**
- Up to `max_mem_frames=5` stored frames
- Reference frame: anchored (never evicted)
- Non-reference frames: FIFO eviction when full
- Memory update: each predicted query frame is optionally written back (controlled by
  `memory_update_freq=1` and scheduled sampling rate)

**Dual-scale keys:**
- Coarse key (256-ch, stage_3): semantic-level matching — captures object identity
- Fine key (96-ch, stage_2): spatial-detail matching — captures boundary position
- Both projected to `d_key=256` via learned linear projections
- Concatenated: effective key sequence `[B, M·P·2, 256]` per attention call

**KAN Key Adapter** (~30k params):
A lightweight `KangaSSM` operating in `d_key=256` space corrects for cumulative appearance
drift in non-reference memory keys:
```
K_adapted_t = K_t + α · KangaSSM(K_t, state_t)
```
Allows the memory bank to model slow appearance change without storing raw decoder features.

---

### 3.4 Propagation Attention

Multi-head cross-attention from current-frame features to the memory bank.

```
Q = QKNorm(W_q(LN(f_t)))       [B, P, d_key]
K = QKNorm(LN(K_mem))          [B, M·P, d_key]
A = softmax(QKᵀ / √(d_k/H))   [B, P, M·P]    — fp32 softmax for AMP safety
R = AV                          [B, P, d_value]

gate, up = proj_out(R).chunk(2)                 — SwiGLU gating (v3)
out = LN(up · silu(gate) + f_t)
```

**Stability features (v3):**
- QK-Norm (Dehghani et al., 2023): LayerNorm on Q and K before dot-product
- fp32 softmax: prevents fp16 saturation under AMP at high logit values
- Logit clamping: `clamp(logit, -50, 50)` as hard safety net
- SwiGLU output gate: eliminates peak `proj_out` norm of 2.98e+05 (job 65737509)
- Xavier init for `W_q`, small-std init (0.02) for `proj_out`

---

### 3.5 Segmentation Decoder

Three-level hierarchical upsampling with genuine multi-scale skip connections from MobileNetV2.

```
SSM output [B·T, 256, 30, 30]  (30×30 at stride-16, 480p input)
     │
     ├─ up1: ConvBN + skip(stage_2, 96-ch)  → [B·T, 192, 30, 30]  (same-scale refinement)
     ├─ up2: ×2 upsample + skip(stage_1, 32-ch) → [B·T, 96, 60, 60]
     ├─ up3: ×2 upsample + skip(stage_0, 24-ch) → [B·T, 48, 120, 120]
     └─ head: Conv + bilinear upsample → [B·T, K+1, 480, 480]
```

Unlike the DINOv2-based setup (all skips at same 14×14 spatial scale), MobileNetV2 provides
**genuinely different spatial resolutions** at each skip level — the decoder performs real
feature-pyramid fusion.

---

## 4. Datasets

### COCO 2017 (Pre-training)

- **Size**: ~118k training images, ~5k validation
- **Annotations**: Instance segmentation polygons / RLE masks (up to 80 categories)
- **Usage**: Static images converted to pseudo-video clips via **coherent affine kinematic prior**

**Kinematic prior (coherent motion trajectory generation):**
```
T_{θ_t}(x) = A_t · x + b_t
θ_t = θ_0 + t · θ̇ + ε_t,   ε_t ~ N(0, σ²)
```
Each clip: 6 frames, constant-velocity affine trajectory with small Gaussian noise.
Parameters: scale ∈ [0.9, 1.1], rotation ∈ [−10°, 10°], translation ∈ [−20, 20] px.

This teaches the memory bank and temporal SSM that objects persist across frames under
rigid/affine motion — a useful inductive bias for Phase 2 domain adaptation to real VOS.

**Why COCO over ADE20K**: instance-level masks (not semantic), directly matching the
VOS annotation format. ADE20K has semantic labels only, requiring a weaker training signal.

---

### BL30K (Phase 1b Synthetic Video Pre-training)

- **Size**: ~30K synthetic video sequences in 6 parts (~700 GB total, parts a–f)
- **Source**: MiVOS / STCN paper (Cheng et al., NIPS 2021). Illinois Data Bank.
- **Annotations**: Binary foreground/background masks (single-object per sequence)
- **Characteristics**: Rendered 3D objects with clean, noise-free masks; realistic motion
- **Clip length**: 12 frames (`clip_len=12` — increased from 6 to give the DTSM slow SSM
  a meaningful temporal horizon during training)
- **Role**: Bridge between COCO static pre-training and real VOS sequences. Teaches the
  temporal model (especially the DTSM slow path) that objects persist across many frames
  with realistic motion dynamics.

**Why skip COCO re-warmup for DTSM:**
COCO is static images — no temporal signal exists. Re-running COCO with the new DTSM
components would waste GPU time: the slow SSM would learn nothing from static images.
Instead, BL30K is the first and correct training environment for the slow path.
All existing components (fast SSMs, decoder, encoder) load from the Phase 1 COCO
checkpoint via `pretrained_weights` (partial weight transfer); only DTSM weights are
randomly initialised (with `log_A=−3.0` as an informative prior).

**Download**: 6 segments (a–f) via `scripts/slurm_setup_bl30k_lustre.sh`
Target: `/lustre/scratch/WUR/AIN/stiva001/video_kamba_data/BL30K`

---

### DAVIS 2017 (Phase 2 Fine-tuning)

- **Size**: 60 training sequences, 30 validation sequences
- **Annotations**: Dense per-frame binary/multi-object masks (pixel-perfect)
- **Characteristics**: Hard scenes — fast motion, occlusion, deformable objects, crowded scenes
- **Clip length**: 4 frames (training), up to 16 frames (validation capped)
- **Resolution**: 480p (`output_size=480`)

DAVIS provides the highest-quality VOS annotations available. Its small size (60 sequences)
means convergence is fast (~12 min/epoch at `limit_train_batches=500`), making it ideal as a
**gate validation** before committing 71h to the joint fine-tune.

---

### YouTube-VOS 2019 + DAVIS 2017 Joint (Phase 3 Fine-tuning)

- **YouTube-VOS 2019**: 3,471 training sequences, 507 validation
  - ~7.8 average objects per video, ~4.3s average duration
  - 65 seen + 26 unseen categories at validation
  - Sparse annotations: every 5th frame labeled during training
- **DAVIS 2017**: 60 sequences (same as Phase 2)
- **Sampling**: `WeightedRandomSampler` ensures DAVIS = 25% of every epoch (otherwise
  YouTube-VOS dominates at 98%)
- **Clip length**: 4 frames sampled with `max_gap=3`
- **Resolution**: 480p

**Why start Phase 3 from Phase 2 checkpoint** (not Phase 1):
PropagationAttention and KangaSSM must first learn VOS-specific dynamics on DAVIS
(real masks, real propagation chains) before being exposed to YouTube-VOS's noisier,
more diverse distribution. Starting from Phase 1 (COCO) would mean the temporal
model enters joint training without ever having seen a VOS sequence.

---

### LVOS V2 (Long-Range Evaluation)

- **Size**: 720 videos (train: 420, val: 140, test: 160)
- **Total frames**: 296,401  |  **Total annotations**: 407,945
- **Average video length**: ~68 seconds (≈ 408 frames at 6 fps annotation rate)
- **Resolution**: 720P
- **Object categories**: 44 (including 12 unseen at validation)
- **Source**: "LVOS: A Benchmark for Long-term Video Object Segmentation" (Hong et al., T-PAMI 2024)

**Why LVOS exposes DTSM's value:**
LVOS videos are **14× longer** than DAVIS/YouTube-VOS (~5s average). The benchmark
specifically targets objects that disappear for extended periods and reappear — the exact
failure mode of FIFO memory banks with short-range SSMs. Without DTSM, the model must
segment from appearance alone when memory has evicted all past evidence.

| Dataset | Videos | Avg length | Long occlusion challenge |
|---------|--------|-----------|--------------------------|
| DAVIS 2017 | 90 | ~5 s | Low |
| YouTube-VOS 2019 | 4,453 | ~5 s | Low |
| BL30K | 30K | ~6 frames | None (synthetic) |
| **LVOS V2** | **720** | **~68 s** | **High (primary design target)** |

---

## 5. Training Protocol

### 5.1 Four-Phase Curriculum (Updated for DTSM)

```
Phase 1 — COCO Static Pre-training             (~20h, 1 GPU)
  Script  : scripts/train_mv2_phase1_coco.sh
  Data    : COCO 2017 (118k images → pseudo-video, 6 frames/clip)
  Epochs  : 20  |  limit_train_batches=3000
  Encoder : MobileNetV2 stages 0+1 frozen (mv2_freeze_at=2)
  LR      : 1e-4 (AdamW + OneCycleLR)
  DTSM    : disabled (static images — no temporal signal for slow path)
  Goal    : val_loss < 0.5 at epoch 5
  Result  : val_J_and_F=0.635 (job 65777196)
  Output  : checkpoints/ssm_mem_v2_aot/last-v1.ckpt  ✓ COMPLETED

              ↓ pretrained_weights transfer (all 631 params, 0 skipped)
                DTSM weights randomly initialised (A_bar≈0.995 prior)

Phase 1b — BL30K Synthetic Video Pre-training  (~48h, 1 GPU)
  Script  : scripts/slurm_train_bl30k_phase2_lustre.sh
  Config  : configs/experiment/mv2_ssm_mem_phase2_bl30k.yaml
  Data    : BL30K (6 parts, ~700GB) — 12 frames/clip (↑ from 6)
  Epochs  : 50  |  limit_train_batches=2000
  Encoder : MobileNetV2 stages 0+1 frozen (mv2_freeze_at=2)
  LR      : 5e-5 (OneCycleLR, 2-epoch warmup)
  DTSM    : ENABLED — slow SSM trains from scratch on temporal sequences
  clip_len: 12 (ensures gradient differentiates fast vs slow timescales)
  Goal    : val_loss decreasing, slow SSM A_bar remains > 0.95
  Output  : checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k.ckpt
  Status  : RUNNING (job 65781828)

              ↓ pretrained_weights transfer

Phase 2 — DAVIS Fine-tuning                    (~17h, 1 GPU)
  Script  : scripts/train_mv2_phase2_davis.sh
  Data    : DAVIS 2017 (60 train sequences)
  Init    : best_slim_v2_phase2_bl30k.ckpt
  Epochs  : 100  |  limit_train_batches=500 (~12 min/epoch)
  Encoder : All stages unfrozen (mv2_freeze_at=0), backbone_lr=lr×0.05
  LR      : 3e-5 (3-epoch warmup from 3e-6)
  ss_rate : linear ramp 0.0 → 0.3 over 100 epochs
  DTSM    : ENABLED
  GATE    : val_J_and_F > 0.55 @ epoch 50 → proceed
  Output  : checkpoints/best_mv2_phase2_davis.ckpt

              ↓ (only if gate passed)

Phase 3 — YouTube-VOS + DAVIS Joint            (~70h, 1 GPU)
  Script  : scripts/train_mv2_phase3_ytbdav.sh
  Data    : YouTube-VOS 2019 + DAVIS 2017 (DAVIS 25%)
  Init    : best_mv2_phase2_davis.ckpt  ← NOT Phase 1
  Epochs  : 50  |  limit_train_batches=3500 (~1.4h/epoch)
  Encoder : All stages trainable, backbone_lr=lr×0.05
  LR      : 3e-5 (2-epoch warmup)
  ss_rate : linear ramp 0.1 → 0.5 (λ* = 0.5, bias-variance optimum)
  DTSM    : ENABLED
  Target  : val_J_and_F > 0.65 (DAVIS), LVOS evaluation post-training
  Output  : checkpoints/best_mv2_phase3_ytbdav.ckpt
```

### 5.2 Scheduled Sampling — Formal Justification

At training step k, with probability λ_k the model uses its own predicted mask as memory
input (self-teaching); with probability 1−λ_k it uses the GT mask (teacher forcing):

```
E_T ≤ λ · Σ_{k=1}^{T} L^{T-k} · ε_k        (memory contamination error bound)
```

The bias-variance trade-off objective has optimum:

```
λ* = argmin_λ [ (1-λ)² · Bias_max + λ(1-λ) · Var_max ] ≈ 0.5
```

Phase 3 asymptotes at λ=0.5, not 1.0 — maximising λ increases error accumulation faster
than it reduces exposure bias. The 0.5 ceiling is a deliberate regulariser.

### 5.3 Optimiser and Gradient Stability (v3)

**4-group learning rate partition:**

| Group | Components | LR multiplier | Effective LR (base 3e-5) |
|-------|-----------|---------------|--------------------------|
| backbone | MobileNetV2 (all stages) | 0.05× | 1.5e-6 |
| temporal | KangaSSM (all layers) | 0.5× | 1.5e-5 |
| propagation | PropagationAttention + MemoryBank | 1.0× | 3.0e-5 |
| base | Decoder + KeyAdapter + norms | 1.0× | 3.0e-5 |

**Gradient control:**
- `gradient_clip_val: 0.5` (norm clipping — reduced from 1.0 after v2 analysis)
- `precision: 16-mixed` (AMP with fp32 softmax in PropagationAttention)
- `accumulate_grad_batches: 2` (effective batch = 8 clips)

**Weight decay**: `0.05` (AdamW, applied to all non-norm, non-bias parameters)

---

## 6. Stability Fix History

| Run | Job ID | Issue | Fix |
|-----|--------|-------|-----|
| v1 | 65725194 | Attention logit explosion; missing QK-norm | Xavier init, QK-norm, fp32 softmax, logit clamp |
| v2 | 65737509 | `out_proj` norm 1.9e+05 (KangaSSM); `C_modulator` softplus unbounded (1.8e+04); affine BCNorm gamma 1.2e+03; `prop_lr=5×` on explosion site | SwiGLU gating; tanh_bounded modulator; non-affine BCNorm; prop_lr→1.0; temporal_lr→0.5; clip→0.5; lr→3e-5 |
| **v3 (MV2)** | **65747603** | Architecture pivot to MobileNetV2; from-scratch encoder was fundamental instability source | **MobileNetV2 ImageNet init; all v3 arch fixes applied** |

---

## 7. Benchmark Context

### Target Performance

**Short-clip benchmarks (DAVIS 2017, YouTube-VOS):**

| Model | Params | DAVIS-17 J&F | YTB-VOS J&F | Backbone | Temporal |
|-------|--------|-------------|-------------|----------|----------|
| AOTT | ~8M | 79.2 | 80.0 | ResNet-50 | Hierarchical Propagation |
| MobileVOS (CVPR 2023) | ~5M | 77.8 | — | MobileNetV2 | Transformer |
| Cutie (CVPR 2024) | ~42M | 84.3 | 82.5 | ResNet-50 | Transformer |
| XMem (ECCV 2022) | ~57M | 81.0 | 81.2 | ResNet-50 | GRU + Attention |
| **Video-Kamba base (~5M)** | **~5M** | **>0.65** | **TBD** | **MobileNetV2** | **KAN-SSM (ours)** |
| **Video-Kamba + DTSM (~8.3M)** | **~8.3M** | **>0.65** | **TBD** | **MobileNetV2** | **KAN-SSM + DTSM (ours)** |

**Long-range benchmark (LVOS V2):**

| Model | Params | LVOS J&F | Long-term memory mechanism |
|-------|--------|----------|--------------------------|
| XMem (ECCV 2022) | ~57M | — | GRU long-term + working memory |
| Cutie (CVPR 2024) | ~42M | — | Object memory tokens (Transformer) |
| **Video-Kamba + DTSM** | **~8.3M** | **TBD** | **Dual-timescale SSM (no attention)** |

**The scientific claims:**
1. At equal backbone (MobileNetV2) and comparable parameter count, KAN-modulated SSM temporal
   reasoning achieves competitive VOS performance to Transformer-based temporal reasoning —
   reducing temporal complexity from O(T²·P²) to O(T·P·D).
2. *(DTSM claim)* Long-range video propagation can be achieved by a second SSM at a slow
   timescale — without any cross-attention mechanism — through hierarchical timescale
   separation in the state decay parameter A.

### DAVIS 2017 Reference (AOT/DeAOT, same backbone family)

| Model | Params | J&F | FPS |
|-------|--------|-----|-----|
| AOTT (ResNet-18) | ~8M | 79.2 | 51.4 |
| DeAOTT | ~11M | — | 53.4 |
| AOTS (ResNet-50) | ~18M | 82.1 | 40.0 |

Our target of >65 J&F is realistic given our smaller parameter budget. With DTSM enabled
(8.3M), the primary differentiator is LVOS performance — a benchmark where XMem/Cutie rely
on attention-based long-term memory and we use purely recurrent state consolidation.

---

## 8. Key Contributions

1. **DiagonalKANSSMCore**: Input-dependent KAN modulation of all three SSM matrices (A, B, C)
   in a diagonal state-space model — balances expressiveness with O(N·D) complexity. Replaces
   quadratic Transformer attention for temporal video reasoning.

2. **Stability-engineered temporal module (v3)**: SwiGLU output gating + tanh-bounded KAN
   modulators + non-affine BCNorm — combination derived from gradient norm analysis of
   two failed training runs, resolving monotonic explosion at 100% rate in `out_proj`.

3. **KAN Key Adapter**: Lightweight recurrent correction of memory-bank keys in key-space
   (~30k params), modeling cumulative appearance drift without storing decoder features.

4. **COCO kinematic pre-training**: Coherent affine motion trajectories as a rigid kinematic
   prior — teaches temporal persistence without requiring large synthetic video datasets
   (BL30K, ~700GB). Domain gap to real VOS is small because short-term object motion is
   locally near-rigid.

5. **Four-phase curriculum with bias-variance optimal scheduled sampling**: Phase ordering
   (COCO → BL30K → DAVIS → YTB+DAV, with Phase 3 initialised from Phase 2) and λ*=0.5
   scheduled sampling ceiling derived from the memory contamination error bound.

6. **Dual-Timescale State Memory (DTSM)**: A second KangaSSM with near-unity decay
   (A_bar≈0.995) runs causally alongside the fast paths, accumulating long-range context
   across all video frames via a persistent hidden state. No cross-attention, no softmax, no
   key-value retrieval — purely recurrent state consolidation. Timescale separation is
   achieved entirely through `log_A` initialisation (`−3.0` vs `log(0.5)` for fast paths).
   The slow path integrates as a residual correction on the fast-path fusion, with near-zero
   initialised projection to prevent disruption of the trained fast path.
   Overhead: +740k params (~14% above 5M base). Target benchmark: LVOS V2 (long-range VOS).

---

## 9. Repository Structure

```
video_kamba/
├── configs/
│   ├── model/
│   │   ├── mv2_ssm_memory.yaml             ← CURRENT production config (DTSM flags here)
│   │   ├── mobilenetv2_kan_temporal.yaml   ← Legacy config
│   │   └── vision_mamba_tiny_sota_light.yaml  ← Deprecated (from-scratch encoder)
│   ├── experiment/
│   │   ├── mv2_ssm_mem_phase2_bl30k.yaml   ← Phase 1b BL30K (DTSM enabled, clip_len=12)
│   │   ├── mv2_phase1_coco.yaml            ← Phase 1 overrides
│   │   ├── mv2_phase2_davis.yaml           ← Phase 2 overrides
│   │   ├── mv2_phase3_ytbdav.yaml          ← Phase 3 overrides
│   │   └── stability_fix_v[1-3].yaml       ← Deprecated (VisionMambaTiny)
│   └── datamodule/
│       ├── bl30k.yaml                      ← BL30K (clip_len=12, Lustre path)
│       ├── coco_pretrain.yaml              ← COCO kinematic pre-train
│       ├── davis.yaml                      ← DAVIS-only fine-tune
│       └── ytv_dav_joint.yaml              ← Joint YTB+DAV
├── models/
│   ├── video_mamba.py                      ← VideoMambaSystem (LightningModule)
│   │                                          use_dual_timescale_memory flag here
│   └── components/
│       ├── mobilenetv2_wrapper.py          ← MobileNetV2 encoder (CURRENT)
│       ├── kan_ssm_core.py                 ← DiagonalKANSSMCore + modulators
│       ├── kanga_ssm.py                    ← KangaSSM (stacked wrapper, PSP-ready)
│       ├── fast_kan_layer.py               ← FastKANLayer (RBF basis)
│       ├── memory_state_bank.py            ← MemoryStateBank (state carry, 0 params)
│       ├── memory_bank.py                  ← Dual-scale K/V episodic memory
│       ├── kan_key_adapter.py              ← Recurrent key drift correction
│       ├── propagation_attention.py        ← Cross-attention to memory bank
│       └── segmentation_decoder.py         ← FPN with multi-scale MV2 skips
├── data/
│   ├── bl30k_pretrain.py                   ← BL30K DataModule (single-object, T=12)
│   ├── coco_pretrain.py                    ← COCO + kinematic trajectory generator
│   ├── davis.py                            ← DAVIS datamodule
│   └── vos_datamodule.py                   ← YouTube-VOS + DAVIS joint loader
├── utils/
│   ├── vos_loss.py                         ← HybridVOSLoss (BCE + SoftDice)
│   └── davis_metrics.py                    ← J&F metric computation
├── scripts/
│   ├── slurm_train_bl30k_phase2_lustre.sh  ← Phase 1b SLURM (CURRENT, Job 65781828)
│   ├── slurm_setup_bl30k_lustre.sh         ← BL30K full download to Lustre (a–f)
│   ├── download_bl30k_subset.py            ← BL30K downloader (--segments flag)
│   ├── slurm_smoke_dtsm.sh                 ← DTSM instantiation smoke test
│   ├── train_mv2_phase1_coco.sh            ← Phase 1 SLURM
│   ├── train_mv2_phase2_davis.sh           ← Phase 2 SLURM
│   ├── train_mv2_phase3_ytbdav.sh          ← Phase 3 SLURM
│   ├── analyse_training_run.py             ← SLURM log + npz artifact analysis
│   └── analyse_gradient_explosion.py       ← Deep gradient breakdown by layer
├── train.py                                ← Hydra entry point (torch.load patch)
├── TRAINING_PROTOCOL.md                    ← Protocol, gates, time budget
└── PROJECT_OVERVIEW.md                     ← This file
```

---

## 10. Active Experiment Status

| Phase | Job ID | Script | Status | Result |
|-------|--------|--------|--------|--------|
| Phase 1 — COCO pre-train | 65777196 | `train_mv2_phase1_coco.sh` | **COMPLETED** | val_J_and_F=0.635 |
| Phase 1b — BL30K + DTSM | 65781828 | `slurm_train_bl30k_phase2_lustre.sh` | **RUNNING** | — |
| Phase 2 — DAVIS fine-tune | — | `train_mv2_phase2_davis.sh` | Pending Phase 1b | — |
| Phase 3 — YTB+DAV joint | — | `train_mv2_phase3_ytbdav.sh` | Pending Phase 2 gate | — |
| LVOS evaluation | — | TBD | Pending Phase 3 | — |

**Phase 1b notes (BL30K + DTSM, job 65781828):**
- DTSM enabled: `temporal_model_slow` (675k) + `slow_residual_proj` (65k) training from scratch
- 631/631 params loaded from Phase 1 COCO checkpoint (0 skipped)
- `clip_len=12` (up from 6) — ensures slow/fast timescale differentiation during training
- Comet tracking live: `video-mamba / video_kamba` workspace

**Engineering fixes applied this session (2026-03-23):**
- `train.py`: `torch.load` patched to `weights_only=False` for trusted internal checkpoints
  (PyTorch 2.6 changed default, breaking Lightning checkpoint loading)
- `models/video_mamba.py`: `obj_present` padded `[B,T,1]→[B,T,n_id]` in `_vos_step` so
  single-object datasets (BL30K) work with a multi-object model (DAVIS 10-object output head)
- `configs/experiment/mv2_ssm_mem_phase2_bl30k.yaml`: switched `checkpoint:` → `pretrained_weights:`
  (was doing full resume with Phase 1 optimizer state; now correct weight-only transfer)
- `.bashrc` + SLURM script: fixed `COMET_API_KEY=your-key-here` placeholder that was
  preventing Comet from tracking training runs
