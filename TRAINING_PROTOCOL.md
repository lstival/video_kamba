# Training Protocol — vision_mamba_tiny_sota_light

> **Last updated**: 2026-03-19
> **Current active job**: `65746167` — `vim_stability_fix_v3` (stability fixes validated)
> **Status**: Stability Fix v3 running on ADE20K pre-train phase

---

## 1. Architecture Overview

### Backbone (frozen)
- **Encoder**: DINOv2 ViT-B/14 — 768-dim patch features, 14×14 spatial grid (196 patches)
- Multi-scale extraction: Layer 6 (coarse), Layer 9 (fine), Layer 11 (semantic)
- Frozen throughout all training phases

### Memory Bank
- Dual-scale keys: Layer 11 (768-dim, semantic) + Layer 9 (768-dim, fine-grained)
- 5 memory frames × 196 patches = 980 key-value pairs (single scale) / 1,960 (dual)
- Values: 256-dim projected via `KANKeyAdapter`

### Propagation Attention
Cross-attend current-frame DINOv2 patches to the memory bank:

```
Q = QKNorm(W_q(LN(f_t)))        [B, P, d_key]
K = QKNorm(LN(K_mem))           [B, M·P, d_key]
A = softmax(QKᵀ / √(d_k/H))    [B, P, M·P]   (fp32 for stability)
R = AV                           [B, P, d_value]
out = LN(SwiGLU(R) + f_t)      [B, P, D]
```

**Stability v3 change**: `proj_out: Linear(d_value → D)` replaced with
`proj_out: Linear(d_value → 2D)`, `gate, up = chunk(2); out = up * silu(gate)`.
Bounds output magnitude — eliminated peak grad norm 2.98e+05 (job 65737509).

### KAN-Modulated SSM (KangaSSM / DiagonalKANSSMCore)

Standard diagonal SSM with content-adaptive modulation:

```
h_t = Ā ⊙ h_{t-1} + (B̄ ⊙ α(u_t)) · u_t
y_t = (C ⊙ β(h_t)) · h_t
```

where:
- `α(u_t) = 1.0 + 0.5·tanh(KAN_B(u_t))  ∈ [0.5, 1.5]` — B modulator
- `β(h_t) = 1.0 + 0.5·tanh(KAN_C(h_t))  ∈ [0.5, 1.5]` — C modulator
- `KAN_*` are `FastKANLayer` (RBF basis, `grid_size=8`)

**Stability v3 changes**:
1. `tanh_bounded` activation replaces `softplus` — modulators bounded to [0.5, 1.5],
   eliminating peak `C_modulator` norm of 1.8e+04 (job 65737509)
2. `BCNorm` (RMSNorm on B/C projections): `elementwise_affine=False` — removes learnable
   gamma that grew to 1.2e+03, negating normalisation
3. `out_proj: Linear(D → D)` → `Linear(D → 2D)` with SwiGLU gating — eliminated peak
   `out_proj.weight` norm 1.9e+05 in 4 epochs (100% explosion rate, all 128 steps)

**Backward pass complexity**: The diagonal structure of Ā decouples state channels.
For each patch p and state channel n:

```
∂L/∂h_t^(p,n) = (∂L/∂y_t^(p)) · C^n + (∂L/∂h_{t+1}^(p,n)) · Ā_{t+1}^(p,n)
```

No cross-state mixing occurs in the gradient — the per-channel recurrence is
independent. Total backward complexity: **O(N · D)** where N = sequence length
(frames) and D = state dimension. When the DINOv2 backbone is frozen,
`∇_{Θ_backbone} L = 0` exactly, halving the effective parameter count for the
backward pass.

### Segmentation Decoder
Hierarchical upsampling with DINO skip connections (Layers 6/9/11):
- Stage 1: 14×14 → 28×28 (Layer 11 skip)
- Stage 2: 28×28 → 56×56 (Layer 9 skip)
- Stage 3: 56×56 → 224×224 (Layer 6 skip)
- Output: binary/multi-object mask logits `[B, n_obj+1, H, W]`

---

## 2. Three-Phase Training Pipeline

```
Phase 1 (ADE20K/COCO)          Phase 2 (DAVIS)          Phase 3 (YTB+DAV)
──────────────────────    ─────────────────────────    ──────────────────────
Static image pre-train → Single-dataset VOS fine-tune → Joint multi-dataset
Coherent motion prior    DAVIS 301 clips               YouTube-VOS + DAVIS
~15–20h (ADE20K/COCO)    ~16h (~12 min/epoch)          ~71h
```

### Phase 1 — Static Image Pre-training

**Datasets**: ADE20K (scene parsing) + COCO (instance segmentation)

**Coherent Motion Prior (Kinematic Synthetic Videos)**

Static images are converted to pseudo-video clips via a **rigid first-order
Markov kinematic prior**. Each clip's affine trajectory is:

```
T_{θ_t}(x) = A_t · x + b_t
θ_t = θ_0 + t·θ̇ + ε_t,    ε_t ~ N(0, σ²)
```

where `θ = (scale, rotation, tx, ty)` and `θ̇` is sampled once per clip.
This generates smooth camera-like motion with small stochastic perturbations.

**Motivation**: This constitutes a deliberate inductive bias — the model learns
that objects persist across frames under rigid (or near-rigid) transformations.
Phase 2 fine-tuning then performs domain adaptation from `D_rigid` (synthetic
affine transforms) to `D_real` (real deformable motion). The residual
gap is small because real short-term object motion is locally near-rigid.

**Key config**:
- `limit_train_batches=3000`, `lr=1e-4`, `max_epochs=20`
- `scheduled_sampling_rate=0.0` (always GT — no exposure bias during pre-train)
- Success criterion: `val_loss < 0.5` at epoch 5

**Scripts**:
```bash
sbatch scripts/pretrain_coco_sota_light.sh   # COCO variant
sbatch scripts/pretrain_ade20k.sh            # ADE20K variant
```

**Stability Fix v3 pre-train** (current, job 65746167):
```bash
sbatch scripts/train_vim_stability_fix_v3.sh
# Calls: train.py model=vision_mamba_tiny datamodule=ade20k_pretrain
#        +experiment=stability_fix_v3 ++model.use_identity_modulation=True
```

---

### Phase 2 — DAVIS Fine-tuning

**Dataset**: DAVIS 2017 semi-supervised (301 training clips)

- Initialise from Phase 1 checkpoint (`best_coco_sota_light.ckpt`)
- 100 epochs, `limit_train_batches=500` (~12 min/epoch)
- Scheduled sampling ramp: `ss_rate: 0.0 → 0.3` over 100 epochs

**Gate @ epoch 50**: check Comet experiment `ft_davis_fast_validate_v1`
- `val_J_and_F > 0.55` → proceed to Phase 3
- `val_J_and_F < 0.55` → investigate before continuing

```bash
sbatch scripts/ft_davis_fast_validate.sh
```

---

### Phase 3 — Joint YouTube-VOS + DAVIS Fine-tuning

**Dataset**: YouTube-VOS 2019 + DAVIS 2017 (joint loader, balanced sampling)

- Initialise from Phase 1 checkpoint (`best_coco_sota_light.ckpt`)
- 50 epochs, `limit_train_batches=3500`
- Scheduled sampling ramp: `ss_rate: 0.1 → 0.5` over 50 epochs

```bash
sbatch scripts/ft_coco_ytb_dav_50ep.sh
```

**Target**: `val_J_and_F > 0.65`
(Reference: MobileVOS CVPR 2023 achieves ~78% J&F at <5M params)

---

## 3. Scheduled Sampling Curriculum

### Formal Bias-Variance Framing

At step k, with probability `λ_k` the model uses its own predicted mask
`m̂_{t-1}` as memory input (model-generated), and with probability `1 - λ_k`
uses the ground-truth mask `m_{t-1}` (teacher-forced):

```
memory_input = { m̂_{t-1}   with probability λ_k
               { m_{t-1}    with probability 1 - λ_k
```

The accumulated memory contamination error bound is:

```
E_T ≤ λ · Σ_{k=1}^{T} L^{T-k} · ε_k
```

where `L` is the Lipschitz constant of the memory update function and
`ε_k` is the prediction error at step k.

The bias-variance trade-off objective:

```
min_λ [ Bias(λ) + α · Var(λ) ]
    = min_λ [ (1 - λ)² · Bias_max + α · λ(1-λ) · Var_max ]
```

The optimal `λ* ≈ 0.5` when `α ≈ 1` (equal weighting of bias and variance),
which is why Phase 3 asymptotes at `ss_rate=0.5` rather than 1.0.
Setting `λ=1.0` would maximise exposure to inference-time distribution but
also maximise error accumulation — the 0.5 ceiling is a deliberate regulariser.

### Schedule by Phase

| Phase | Epochs | ss_rate start | ss_rate end | Note |
|-------|--------|---------------|-------------|------|
| 1 (pre-train) | 20 | 0.0 | 0.0 | Always GT — learn features first |
| 2 (DAVIS) | 100 | 0.0 | 0.3 | Conservative ramp |
| 3 (YTB+DAV) | 50 | 0.1 | 0.5 | Full curriculum to λ* |

---

## 4. Optimiser Configuration (v3)

### Learning Rate Groups

The parameter space is partitioned into 4 groups with distinct learning rates:

```
Θ = Θ_prop ∪ Θ_temporal ∪ Θ_base ∪ Θ_backbone
```

| Group | Parameters | LR multiplier | Effective LR (base 3e-5) |
|-------|-----------|---------------|--------------------------|
| `backbone` | DINOv2 ViT-B/14 (trainable layers) | 0.05× | 1.5e-6 |
| `temporal` | KangaSSM (all KAN-SSM layers) | 0.5× | 1.5e-5 |
| `base` | Decoder, key adapter, norms | 1.0× | 3.0e-5 |
| `propagation` | PropagationAttention W_q, proj_out | 1.0× | 3.0e-5 |

**Rationale for `propagation_lr_multiplier = 1.0`** (was 5.0 in v2):
PropagationAttention `out_proj` was explosion site #2 (peak 2.98e+05). A 5×
multiplier on an already-exploding component was the direct cause. Reducing to
1.0× removes the amplification.

**Rationale for `temporal_lr_multiplier = 0.5`**:
KangaSSM `out_proj` was explosion site #1 (3.9e+04 → 1.9e+05, 5× in 4 epochs,
100% explosion rate). A separate lower LR for the temporal stream reduces the
magnitude of each update, giving SwiGLU gating time to establish stable dynamics
at the start of training.

### Optimiser

```yaml
optimizer: AdamW
  lr: 3.0e-5          # was 5e-5 in v2
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.05

lr_scheduler: OneCycleLR
  max_lr: 3.0e-5
  pct_start: 0.1      # 10% warmup
  anneal_strategy: cos
  div_factor: 10.0
  final_div_factor: 100.0
```

### Gradient Clipping

```yaml
gradient_clip_val: 0.5        # was 1.0 in v2 — more aggressive as safety net
gradient_clip_algorithm: norm
```

---

## 5. Stability Fix History

| Version | Job ID | Root Cause | Fix |
|---------|--------|------------|-----|
| v1 | 65725194 | Xavier init missing; no QK-norm | Xavier init; QK-norm; fp32 softmax; logit clamp |
| v2 | 65737509 | `out_proj` unbounded; softplus KAN; affine BCNorm; prop_lr=5× | (identified, not yet fixed in v2) |
| **v3** | **65746167** | — | SwiGLU gating on `out_proj` + `proj_out`; `tanh_bounded` KAN modulators; non-affine BCNorm; `prop_lr=1.0`; `temporal_lr=0.5`; `clip=0.5`; `lr=3e-5` |

### v3 Architecture Changes (code-level)

**KangaSSM** (`models/components/kanga_ssm.py`):
```python
# Before: Linear(D → D) → residual add
# After:  Linear(D → 2D) → SwiGLU → residual add
gate, up = self.out_proj(x).chunk(2, dim=-1)
out = up * F.silu(gate) + residual
# Init: std = 0.02 / sqrt(2 * n_layers)  [GPT-2/Mamba convention]
```

**PropagationAttention** (`models/components/propagation_attention.py`):
```python
# Before: Linear(d_value → D) → LN(out + query_feat)
# After:  Linear(d_value → 2D) → SwiGLU → LN(out + query_feat)
gate, up = self.proj_out(out).chunk(2, dim=-1)
out = self.norm_out(up * F.silu(gate) + query_feat)
# Init: std = 0.02  [standard residual branch init]
```

**DiagonalKANSSMCore** (`models/components/kan_ssm_core.py`):
```python
# Modulator activation: softplus → tanh_bounded
return 1.0 + 0.5 * torch.tanh(out)   # output ∈ [0.5, 1.5]

# BCNorm: elementwise_affine=True → False
self.B_norm = nn.RMSNorm(state_dim, elementwise_affine=False)
self.C_norm = nn.RMSNorm(inner_dim, elementwise_affine=False)
```

---

## 6. Submission Sequence

### Current: Stability Fix v3 Pre-training
```bash
sbatch scripts/train_vim_stability_fix_v3.sh   # Job 65746167 — running
```

### Full Pipeline (when v3 pre-train converges)

```bash
# Phase 1: ADE20K pre-train (stability v3 or COCO variant)
sbatch scripts/train_vim_stability_fix_v3.sh

# Phase 2: DAVIS fine-tune (gate decision @ epoch 50)
sbatch scripts/ft_davis_fast_validate.sh

# Phase 3: Joint fine-tune (only if gate passed)
sbatch scripts/ft_coco_ytb_dav_50ep.sh
```

---

## 7. Success Criteria

| Phase | Epoch | Metric | Threshold | Action if failed |
|-------|-------|--------|-----------|-----------------|
| ADE20K/COCO pre-train | 5 | `val_loss` | < 0.5 | Check data pipeline, LR |
| Stability fix v3 | 3 | `grad_norm (out_proj)` | < 1e+03 | Check SwiGLU init |
| DAVIS fine-tune | 20 | `val_J_and_F` | > 0.25 | Check Phase 1 ckpt quality |
| **DAVIS gate** | **50** | **`val_J_and_F`** | **> 0.55** | **Stop — re-run Phase 1** |
| YTB+DAV | 20 | `val_J_and_F` | > 0.60 | Check ss_rate schedule |
| YTB+DAV | 50 | `val_J_and_F` | > 0.65 | Target (MobileVOS ref: 0.78) |

---

## 8. Time Budget

| Phase | Script | Est. Time |
|-------|--------|-----------|
| ADE20K/COCO pre-train (v3) | `train_vim_stability_fix_v3.sh` | ~20h |
| DAVIS fine-tune | `ft_davis_fast_validate.sh` | ~16h |
| Gate decision (Comet check) | — | — |
| YTB+DAV joint fine-tune | `ft_coco_ytb_dav_50ep.sh` | ~71h |
| **Total without Phase 3** | | **~36h** |
| **Total with Phase 3** | | **~107h** |

---

## 9. Experiment Tracking

All runs logged to **Comet** under project `video_kamba`.

| Job ID | Comet run name | Phase | Status |
|--------|---------------|-------|--------|
| 65725194 | `vim_stability_fix_v1` | ADE20K pre-train | Gradient explosion (QK-norm missing) |
| 65737509 | `vim_stability_fix_v2` | ADE20K pre-train | Gradient explosion (`out_proj` unbounded) |
| 65746167 | `vim_stability_fix_v3` | ADE20K pre-train | **Running** |

Monitored parameters per run:
- `grad_norm/out_proj`, `grad_norm/proj_out`, `grad_norm/C_modulator`
- `train_loss`, `val_loss`, `val_J_and_F`
- `lr/temporal`, `lr/propagation`, `lr/backbone`
- GPU memory utilisation, step time

---

## 10. Architecture Compatibility Notes

- Model: `vision_mamba_tiny_sota_light` (`DiagonalKANSSMCore`, `ssm_layers=2`)
- All checkpoints from `IntricateKANSSMCore` are **incompatible**
- v3 SwiGLU `out_proj` doubles the output dimension of that layer — checkpoints from
  v1/v2 (pre-SwiGLU) are **incompatible** with v3 models
- The `temporal_lr_multiplier` param must exist in `VideoMambaSystem.__init__`; loading
  v1/v2 configs without it will use the default `0.5`
