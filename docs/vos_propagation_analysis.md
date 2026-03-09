# VOS Propagation: AOT vs SSM-KAN — Deep Technical Analysis

> *Branch*: `eval/aot-sota-baseline` | AOTT baseline: **J&F = 83.3%** (DAVIS 2017 val)

---

## 1. Information Propagation: The Core Problem

Video Object Segmentation (VOS) requires answering **"where is object X in frame t?"**
given only its mask annotation at frame 0. Propagation is how past knowledge
flows to future frames.

Both methods solve this, but differ fundamentally in *how* memory is stored and
*how* queries are resolved.

---

## 2. AOT Propagation — Detailed Tensor Flow

### 2.1 Architecture Overview

```
Frame 0 (reference)                    Frame t (query, t > 0)
┌─────────────────────┐                ┌───────────────────────┐
│ MobileNetV2 encoder │                │ MobileNetV2 encoder   │
│ [B,3,960,960]→      │                │ [B,3,960,960]→        │
│ [B,256,30,30]       │                │ [B,256,30,30]         │
└────────┬────────────┘                └──────────┬────────────┘
         │ + ID Embeddings                         │
         │ (patch_wise_id_bank)                    │
         ▼                                         ▼
   LSTT Layer 1..L                           LSTT Layer 1..L
   ┌────────────────┐                        ┌────────────────┐
   │ 1. Self-Attn   │←───── long-term K,V ───│ 1. Self-Attn   │
   │ 2. Long-range  │        (900*T, B, 256)  │ 2. Long-range  │
   │ 3. Local 15×15 │←───── short-term K,V ──│ 3. Local 15×15 │
   │ 4. FFN (DWConv)│                        │ 4. FFN (DWConv)│
   └───────┬────────┘                        └───────┬────────┘
           │                                         │
           ▼                                         ▼
   FPN Decoder                              FPN Decoder
   [B,11,960,960]                           [B,11,960,960]
           │                                         │
           ▼                                         ▼
   Store K,V in                             Update long/short
   long-term memory                         term memory with
   (with ID embeddings!)                    predicted mask ID
```

### 2.2 Object ID Embedding — The Core Mechanism

```python
# networks/models/aot.py — patch_wise_id_bank
self.patch_wise_id_bank = nn.Conv2d(
    in_channels=11,        # 0=bg + 10 objects (one-hot)
    out_channels=256,      # same as feature dim
    kernel_size=17,        # Large receptive field (covers ~17×17 pixels at stride 16)
    stride=16,             # Produces 30×30 output for 480p input
    padding=8,
)
# Input : (B, 11, 960, 960) — one-hot mask per object ID
# Output: (B, 256, 30, 30) — patch-level ID features
```

**Why `kernel_size=17, stride=16`?**
Each output position aggregates a 17×17 region; since the mask has hard boundaries,
this smooths the ID signal across a patch rather than relying on a single pixel.

**How ID embeddings enter the memory:**

```python
# networks/layers/transformer.py — fuse_key_value_id
def fuse_key_value_id(self, key, value, id_emb):
    K = key                             # (900, B, 256)  ← SPATIAL (unchanged)
    V = self.linear_V(value + id_emb)  # (900, B, 256)  ← ID-TAGGED
    return K, V
```

Keys stay spatial (used to *find* where the object is).  
Values become ID-tagged (used to *return what* the object is).

### 2.3 Long-Term vs Short-Term Memory

| Property       | Long-Term Memory                     | Short-Term Memory              |
|----------------|--------------------------------------|--------------------------------|
| **Content**    | K, V from keyframes (ID-tagged)      | K, V from frame t-1 (local)   |
| **Shape**      | (900·K, B, 256) grows over time      | Fixed: (900, B, 256)           |
| **Access pattern** | Global softmax over all past       | Local 15×15 window per position|
| **Complexity** | O(N · M) → O(810K) for long seqs     | O(N · W²) → O(200K) constant  |
| **Update rate**| Every `long_term_mem_gap` frames     | Every frame                    |
| **Purpose**    | Object identity association          | Temporal coherence/smoothness  |

### 2.4 Propagation at Frame t

```
NO ground truth mask at frame t. Object identity comes entirely from memory.

Q = current_frame_features           # (900, B, 256)
K = long_term_keys                   # (900·keyframes, B, 256) — spatial
V = long_term_values_with_id         # (900·keyframes, B, 256) — ID-enriched

Attention:
  softmax(Q·K^T / √256)·V  →  output (900, B, 256)
  ↑ current pixels query:
    "Which past frame position looks like me?"
  ↑ returned values say:
    "That position belongs to object ID X"

Result: current features are now conditioned on object identities
→ Decoder argmax over 11 channels gives per-pixel object ID
```

---

## 3. Our SSM-KAN Propagation — Detailed Tensor Flow

### 3.1 Architecture Overview (fix/claude_decoder branch)

```
Frame 0 (reference)                    Frame t (query, t = 1..T)
┌─────────────────────┐                ┌───────────────────────────────────────┐
│ DINOv2 ViT-B/14     │                │ DINOv2 ViT-B/14 (frozen)              │
│ [B,3,224,224]→      │                │ [B,T,3,224,224]→                      │
│ Layer 11: [B,1,768,196] (stage_3)   │ Layer 11: [B,T,768,196] (stage_3)     │
│ Layer  9: [B,1,768,196] (stage_2)   │ Layer  9: [B,T,768,196] (stage_2)     │
│ Layer  6: [B,1,768,196] (stage_1)   │ Layer  6: [B,T,768,196] (stage_1)     │
└────────┬────────────┘                └──────────┬────────────────────────────┘
         │ + ref_mask                             │ (per frame t)
         ▼                                        ▼
   MemoryBank.encode_reference()         PropagationAttention (cross-attn)
   ┌────────────────────────────┐        ┌────────────────────────────────────┐
   │ proj_key([B,P,768])→K_ref  │        │ Q = W_q(LN(curr_patch)) [B,P,256] │
   │ proj_value([B,P,768])→V_ref│        │ K = memory_bank.K [B,MP,256]      │
   │ + ID embeddings in V_ref   │        │ V = memory_bank.V [B,MP,256]      │
   │   (ID Conv weighted by mask)│       │ out = softmax(QK^T/√32)·V         │
   └────────────────────────────┘        │     = [B, P, 256]→[B, P, 768]    │
                                         └──────────────┬─────────────────────┘
                                                        │ prop_feat [B,P,768]
                                                        ▼
                                         KAN-SSM (per-patch temporal)
                                         ┌────────────────────────────────────┐
                                         │ [B*P, 1, 768] →                   │
                                         │ A_{t} = exp(Δ·A)        [B*P,16,16]│
                                         │ B_{t} = KAN(u_t)·B_disc [B*P,16,768]│
                                         │ h_t = A_{t}·h_{t-1} + B_{t}·u_t  │
                                         │ y_t = C_{t}·h_t        [B*P, 768] │
                                         └──────────────┬─────────────────────┘
                                                        │ ssm_feat [B,1,768,196]
                                                        ▼
                                         SegmentationDecoder (FPN-style)
                                         ┌────────────────────────────────────┐
                                         │ Up1: 14→28 + skip stage_2 (768)   │
                                         │ Up2: 28→56 + skip stage_1 (768)   │
                                         │ Up3: 56→224 + conv                │
                                         │ → [B,1,num_classes,224,224]       │
                                         └──────────────────────────────────-─┘
```

### 3.2 MemoryBank ID Embedding — Our Approach

```python
# models/components/memory_bank.py
# ID embeddings: a learnable per-object weight inserted into values
self.id_embeddings = nn.Embedding(
    num_embeddings=n_objects + 1,  # +1 for background
    embedding_dim=d_value,         # 256
)

def encode_reference(self, ref_patch, ref_mask, ref_patch_fine=None):
    # ref_patch: [B, P, 768]  — DINO patch features
    # ref_mask:  [B, H, W]    — integer object IDs (0=bg, 1-10=objects)
    
    # 1. Project spatial features to key
    K = self.proj_key(ref_patch)   # [B, P, d_key=256]
    V = self.proj_value(ref_patch) # [B, P, d_value=256]
    
    # 2. Compute soft per-patch object assignment
    #    by avg-pooling the mask down to patch resolution
    mask_soft = F.adaptive_avg_pool2d(one_hot_mask, (14, 14))  # [B, N+1, 14, 14]
    mask_soft = mask_soft.flatten(2).permute(0, 2, 1)           # [B, P, N+1]
    
    # 3. Weighted ID embedding sum — per AOT's "values are ID-tagged" design
    id_emb = self.id_embeddings.weight  # [N+1, d_value]
    V = V + torch.einsum('bpn,nd->bpd', mask_soft, id_emb)      # [B, P, d_value]
    # ^ This is equivalent to AOT's fuse_key_value_id(V + id_emb)
    
    self._push_to_bank(K, V)
```

### 3.3 KAN-SSM Temporal Refinement — Why It Matters

After PropagationAttention retrieves "what object is here", the KAN-SSM refines
over time using the sequence of propagated features:

```
KAN-SSM state update per patch position p:

  h_t^p ∈ ℝ^16  ← compressed temporal memory for patch p

  B_t^p = B_base ⊙ KAN_B(prop_feat_t^p)  ← input-dependent coupling
  C_t^p = C_base ⊙ KAN_C(prop_feat_t^p)  ← input-dependent readout

  h_t^p = exp(Δ·A) · h_{t-1}^p + B_disc_t^p · prop_feat_t^p
  y_t^p = C_t^p · h_t^p

Key property: KAN modulates B and C per-input.
→ When a patch has high PropagationAttention confidence (object clearly found),
  KAN can gate a strong state update.
→ When propagation is ambiguous (e.g., occlusion), KAN can down-modulate the
  input coupling and rely more on the previous hidden state.
```

This is a form of **content-dependent temporal gating** that AOT lacks entirely.

---

## 4. Critical Comparison Table

| Dimension                  | AOT (SOTA)                           | SSM-KAN (Ours)                          | Gap / Opportunity            |
|----------------------------|--------------------------------------|-----------------------------------------|------------------------------|
| **Backbone**               | MobileNetV2 (960p input)            | DINOv2 ViT-B/14 (224p input)           | DINO features richer         |
| **Feature dim**            | 256                                  | 768 → projected to 256                 | ← we project down, same K/V  |
| **Spatial tokens (N)**     | 30×30 = 900                          | 14×14 = 196                             | ← 4.6× fewer tokens!         |
| **Attention complexity**   | O(900²) long + O(900×225) local      | O(196 × MP) single-scale               | We're cheaper but less spatial|
| **ID embedding**           | Conv2d(11→256, k=17, s=16) on mask  | Embedding(N+1, 256) + avg-pool         | ≈ equivalent, different impl  |
| **Memory structure**       | LSTT KV per layer (L=1-3 layers)    | Single MemoryBank KV store             | AOT has layered hierarchy     |
| **Long-term memory**       | Grows unbounded (all keyframes)      | Fixed sliding window (max_mem_frames)  | AOT keeps all history         |
| **Temporal model**         | None (pure attention)                | KAN-SSM (state_dim=16, per patch)      | **Our unique contribution**   |
| **Short-term coherence**   | Local 15×15 attention                | SSM hidden state h_t (implicit)        | AOT more explicit             |
| **Input resolution**       | 480p → feature 30×30                | 224p → feature 14×14                   | Lower res = less detail       |
| **Training data**          | PRE_YTB_DAV (YouTube+DAVIS)          | DAVIS 2017 only (currently)            | AOT trained on 10× more data  |
| **Parameters**             | 22-35M (AOTT-AOTL)                   | ~85M (DINO frozen + propagation)       | Larger but DINO is frozen     |
| **J&F DAVIS 2017 val**     | 83.3% (AOTT, ours measured)          | TBD                                    |                               |

---

## 5. Root Cause Analysis: Why Our Results Will Differ

### 5.1 Spatial Resolution Problem

```
AOT:  480p → MobileNetV2 stride 16 → 30×30 = 900 tokens
Ours: 224p → DINOv2 patch 14 → 14×14 = 196 tokens

Resolution loss: (224/480)² × (196/900) ≈ 21% spatial coverage

Impact:
  - Object boundaries at 224p are blurry vs 480p
  - Fine structures (e.g., person's hands, thin objects) lose detail
  - Border accuracy (F-measure) will be lower than AOT
```

**Fix**: Use DINOv2 at 448p or 672p → patches become 32×32=1024 or 48×48=2304 tokens
(DINOv2 patch size = 14px, so input 448=14×32 → 32×32 patches)

### 5.2 Memory Depth Problem

AOT's LSTT with L=3 layers means information flows through 3 successive
attention refinements at each forward step. Each layer's KV memory independently
specializes (early layers → spatial matching, later layers → object identity).

Our single MemoryBank flattens this to one cross-attention + one SSM step.

**Fix**: Stack PropagationAttention blocks (2-3 layers) between DINO encoding
and KAN-SSM, mirroring AOT's L=1,2,3 variants.

### 5.3 Training Data Scarcity

```
AOT pretrained on:
  - YouTube-VOS 2018 train: 3,471 video sequences
  - DAVIS 2017 train: 60 sequences
  Total: ~80K annotated frames

Our training:
  - DAVIS 2017 train only: 60 sequences → ~2,600 frames
  
Data multiplier: AOT has ~30× more training data.
```

**Fix 1**: Download and use YouTube-VOS data with our AOT-compatible dataloader.  
**Fix 2**: Static image pretraining (COCO, PASCAL) using AOT's StaticTrain protocol.

### 5.4 KAN-SSM State Dimensionality

```
SSM state: h_t ∈ ℝ^16  (per patch)
Total temporal memory: B × P × 16 = 2 × 196 × 16 = 6,272 scalars

AOT long-term memory: B × N_keyframes × 900 × 256
For N=5 keyframes: 2 × 5 × 900 × 256 = 2,304,000 scalars

Ratio: AOT stores 367× more temporal information
```

The SSM state is highly compressed. This is by design (fast, efficient), but
it means less historical context for long sequences.

**Fix**: Increase state_dim from 16 → 64 or 128 (moderate cost increase).
Or add SSM output to MemoryBank instead of only using it for decoding.

---

## 6. Alignment Strategies — Ranked by Expected Impact

### Strategy 1: Higher Input Resolution (Expected gain: +3-5% J&F)

```python
# configs/datamodule/vos.yaml
img_size: 448   # was 224  →  14×14→32×32 patches (1024 tokens)

# data/vos_datamodule.py
_DINO_INPUT_SIZES = {224: 16, 448: 32, 672: 48}  # px per token
```

DINOv2 was trained with positional interpolation, handles arbitrary multiples of 14.

### Strategy 2: Multi-Scale Memory Keys — Already Implemented!

The `fix/claude_decoder` branch already has `use_dual_scale=True` which stores
both Stage 3 (14×14 semantic) and Stage 2 (28×28 fine-grained) DINO features
as memory keys. This directly parallels AOT's short-term (local) attention that
captures fine-grained spatial correspondence.

To activate: ensure `prop_use_dual_scale: true` in config.

### Strategy 3: Stacked Propagation Layers (Expected gain: +2-3% J&F)

```python
# Currently: one PropagationAttention → one KAN-SSM
# Proposed: N_prop_layers stacked blocks

class PropagationStack(nn.Module):
    def __init__(self, d_model, d_key, d_value, n_heads, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            PropagationAttention(d_model, d_key, d_value, n_heads)
            for _ in range(n_layers)
        ])
    
    def forward(self, query_feat, K_mem, V_mem):
        for layer in self.layers:
            query_feat = layer(query_feat, K_mem, V_mem)
        return query_feat

# In VideoMambaSystem.forward():
# prop_feat = self.propagation_stack(curr_patch, K_mem, V_mem)  # N=3 layers
```

### Strategy 4: AOT-Style Training Data (Expected gain: +5-8% J&F)

The biggest gap is training data. Use `data/vos_datamodule.py` (built below) with:
- `dataset: youtube_vos` → YouTube-VOS 2019 (2,883 train sequences)
- `dynamic_merge: true` → AOT's augmentation that overlays two sequences
  (creates harder multi-object scenarios)

### Strategy 5: KAN-Conditioned ID Injection

**Novel idea**: instead of injecting fixed ID embeddings into memory values,
use the KAN modulator to dynamically adjust B and C matrices conditioned on
*which object* the current patch is attending to.

```python
# In KangaSSM.forward():
# After PropagationAttention, we know the weighted object mixture:
obj_mixture = attention_weights @ id_embeddings  # [B, P, d_value]

# Condition KAN modulation on object identity
B_mod = KAN_B(cat([prop_feat, obj_mixture]))  # richer conditioning
C_mod = KAN_C(cat([prop_feat, obj_mixture]))
```

This would make the SSM's temporal dynamics **object-aware** — the state update
rule would differ for "background" patches vs "moving object" patches.

---

## 7. Forward Pass Alignment — Side-by-Side

```
                AOT (AOTT, L=1)                    Ours (fix/claude_decoder)
                ───────────────────                ──────────────────────────
ENCODE REF:
  features    backbone(ref_img)                    dino(ref_img)["layer_11"]
  shape       [B, 256, 30, 30]                     [B, 1, 768, 196]
  id_emb      patch_wise_id_bank(one_hot_mask)      MemoryBank.id_embeddings (Emb)
  id shape    [B, 256, 30, 30]                     [B, P, d_value]
  store       LSTT.long_term_memories[0]=K,V       memory_bank._bank = {K, V}

PROPAGATE (frame t):
  extract     backbone(curr_img)                   dino(curr_imgs[:,t])["layer_11"]
  query       (900, B, 256)                        [B, 196, 768]
  cross-attn  long_term_attn(Q, LT_K, LT_V)        propagation_attention(Q, K, V)
  local-attn  short_term_attn(local_Q, ST_K, ST_V) (dual-scale keys serve similar role)
  temporal    ─ (none)                             KAN-SSM per patch position
  decode      FPN(LSTT_out + shortcuts)             SegDecoder(ssm_out + skip[6,9,11])
  output      [B, 11, 960, 960]                    [B, T, num_classes, 224, 224]

MEMORY UPDATE:
  with GT?    teacher forcing (use predicted mask) scheduled_sampling_rate
  update K,V  fuse predicted mask ID → V           memory_bank.add_frame(patch, mask)
  LT update   every long_term_mem_gap frames       every memory_update_freq frames
```

---

## 8. Key Files Reference

| File | Role | Key Tensors |
|------|------|-------------|
| `sota/aot-benchmark/networks/engines/aot_engine.py` | AOT propagation loop | `long_term_memories`, `short_term_memories` |
| `sota/aot-benchmark/networks/layers/transformer.py` | LSTT blocks | K,V fusion, fuse_key_value_id |
| `sota/aot-benchmark/networks/layers/attention.py` | Multi-head + local attention | MultiheadLocalAttentionV3 |
| `sota/aot-benchmark/networks/models/aot.py` | Model: patch_wise_id_bank | ID embedding Conv2d |
| `models/components/memory_bank.py` | Our MemoryBank | K/V store, ID embeddings |
| `models/components/propagation_attention.py` | Our cross-attention | Q,K,V projections |
| `models/components/kanga_ssm.py` | KAN-SSM temporal model | h_t state, KAN B/C |
| `models/components/kan_ssm_core.py` | SSM core with KAN modulation | IntricateKANSSMCore |
| `models/components/segmentation_decoder.py` | FPN-style decoder | Skip connections |
| `models/video_mamba.py` | Full VOS system | Frame-by-frame loop |
| `data/vos_datamodule.py` | AOT-style training data | ref+query protocol |
| `eval/run_our_vos_eval.py` | Evaluate on DAVIS 2017 val | J&F computation |

---

## 9. Empirical Predictions

Based on the architectural analysis, before any fixes:

| Component  | Expected issue                        | Predicted impact on J&F |
|------------|---------------------------------------|-------------------------|
| 224p input | Boundary blurring, less spatial detail | -4 to -6% F-measure     |
| Single-scale memory | Misses fine-grained correspondence | -2 to -3% J   |
| DAVIS-only training | Overfitting, poor generalization   | -5 to -10% overall      |
| 1× PropagationAttention | Under-resolved ambiguous regions | -2 to -3% J |
| SSM state_dim=16 | Long-sequence drift              | -1 to -2% on long seqs  |

**Realistic baseline estimate**: 60-70% J&F before any targeted fixes.

**After Strategy 1+2+4** (resolution + multi-scale + YouTube-VOS data):  
**Target**: 75-80% J&F — closing most of the gap with AOTT (83.3%).

**After Strategy 3+5** (stacked layers + object-aware KAN):  
**Target**: 80-85% J&F — potentially surpassing AOTT with a hybrid model.

---

## 10. Next Steps — Ordered Action Plan

```
Priority  Action                                    File
────────  ──────────────────────────────────────    ──────────────────────────
1 (done)  Build AOT-style VOS dataloader            data/vos_datamodule.py
2 (done)  Build evaluation script for our model     eval/run_our_vos_eval.py
3 (done)  Run DAVIS 2017 val evaluation              eval/run_our_vos_eval.py
4         Switch to 448p input                       configs/datamodule/vos.yaml
5         Retrain with YouTube-VOS + DAVIS           data/vos_datamodule.py (yt-vos)
6         Stack 2 PropagationAttention layers        models/video_mamba.py
7         Fix multi-object ID loss (see §11)         models/video_mamba.py
8         Implement object-aware KAN conditioning    models/components/kanga_ssm.py
9         Full ablation study                        scripts/exp4_propagation_ablation.py
```

---

## 11. Empirical Results — SSM-KAN vs AOTT (DAVIS 2017 Val)

> Checkpoint: `epoch=49-step=11050` | Resolution: 224×224 | Device: CPU  
> Branch: `fix/claude_decoder` | Date: June 2025

### 11.1 Global Metrics

| Model             | J&F   | J     | F     | Notes                          |
|-------------------|-------|-------|-------|--------------------------------|
| **SSM-KAN (ours)**| 28.4% | 35.1% | 21.6% | epoch=49, DAVIS-only training  |
| AOTT (our run)    | 83.3% | 81.8% | 84.7% | YT-VOS+DAVIS training          |
| AOTT (paper)      | 79.2% | 76.5% | 81.9% |                                |
| AOTS (paper)      | 82.1% | 79.3% | 84.8% |                                |
| AOTB (paper)      | 83.3% | 80.6% | 85.9% |                                |

**Gap vs AOTT: -54.9% J&F**

### 11.2 Per-Sequence Results

| Sequence           | Ours J | Ours F | Ours J&F | AOTT J&F | Δ      |
|--------------------|--------|--------|----------|---------|--------|
| bike-packing       | 0.266  | 0.212  | 0.239    | 0.814   | -57.5% |
| blackswan          | 0.627  | 0.251  | 0.439    | 0.972   | -53.3% |
| bmx-trees          | 0.245  | 0.261  | 0.253    | 0.710   | -45.7% |
| breakdance         | 0.567  | 0.321  | 0.444    | 0.895   | -45.1% |
| camel              | 0.550  | 0.279  | 0.415    | 0.972   | -55.7% |
| **car-roundabout** | 0.722  | 0.482  | **0.602**| 0.968   | -36.6% |
| **car-shadow**     | 0.760  | 0.420  | **0.590**| 0.974   | -38.4% |
| cows               | 0.689  | 0.383  | 0.536    | 0.960   | -42.4% |
| dance-twirl        | 0.562  | 0.242  | 0.402    | 0.881   | -47.9% |
| dog                | 0.702  | 0.290  | 0.496    | 0.966   | -47.0% |
| dogs-jump          | 0.112  | 0.099  | 0.105    | 0.942   | -83.7% |
| drift-chicane      | 0.440  | 0.241  | 0.341    | 0.767   | -42.6% |
| **drift-straight** | 0.764  | 0.459  | **0.612**| 0.929   | -31.7% |
| goat               | 0.665  | 0.290  | 0.477    | 0.899   | -42.2% |
| gold-fish          | 0.050  | 0.061  | 0.056    | 0.879   | -82.4% |
| horsejump-high     | 0.439  | 0.299  | 0.369    | 0.870   | -50.1% |
| india              | 0.065  | 0.059  | 0.062    | 0.680   | -61.8% |
| judo               | 0.103  | 0.147  | 0.125    | 0.873   | -74.8% |
| kite-surf          | 0.054  | 0.058  | 0.056    | 0.542   | -48.6% |
| lab-coat           | 0.031  | 0.043  | 0.037    | 0.536   | -49.9% |
| libby              | 0.405  | 0.227  | 0.316    | 0.925   | -60.9% |
| loading            | 0.120  | 0.130  | 0.125    | 0.867   | -74.2% |
| mbike-trick        | 0.109  | 0.058  | 0.084    | 0.730   | -64.6% |
| motocross-jump     | 0.369  | 0.215  | 0.292    | 0.693   | -40.1% |
| paragliding-launch | 0.107  | 0.044  | 0.075    | 0.542   | -46.7% |
| parkour            | 0.389  | 0.283  | 0.336    | 0.956   | -62.0% |
| pigs               | 0.330  | 0.252  | 0.291    | 0.847   | -55.6% |
| scooter-black      | 0.065  | 0.107  | 0.086    | 0.821   | -73.5% |
| shooting           | 0.113  | 0.104  | 0.109    | 0.792   | -68.3% |
| soapbox            | 0.123  | 0.166  | 0.144    | 0.782   | -63.8% |

### 11.3 Root Cause Analysis (Empirical)

**Observation 1: J >> F (35.1% vs 21.6%)**
- All sequences have J substantially larger than F
- Root cause: DINOv2 patch size = 14×14 px. At 224×224 input, each patch covers 14px².
  After upsampling (×16) to full resolution, boundaries are aliased.
- Fix: Use higher res input (448×448 → 8px patches after upsampling ×2 only)
   OR add a boundary refinement head.

**Observation 2: Catastrophic failure on multi-object (J ≈ 0.05)**
- gold-fish (J=0.050), india (J=0.065), dogs-jump (J=0.112): all multi-object hard cases
- Root cause: The model's class IDs (0..10) may not persist correctly across frames
  in multi-object sequences. The propagation attention must assign stable IDs.
- Fix: Add object-ID injection into memory bank values (as in AOT's `patch_wise_id_bank`).

**Observation 3: Good performance on simple single-object (J ≈ 0.6-0.76)**
- car-roundabout (J=0.722), car-shadow (J=0.760), drift-straight (J=0.764): simple tracking
- The MemoryBank + PropagationAttention IS working for coarse tracking
- Fix: Improve boundary quality (higher res) and multi-object ID handling

**Observation 4: Large gap even on "easy" sequences**
- Even our best sequence (drift-straight 0.612) is -32% below AOTT (0.929)
- AOTT is trained on YouTube-VOS (3471 videos) vs our DAVIS-only (60 videos)
- Fix: Incorporate YouTube-VOS training data

### 11.4 Expected Gains from Fixes (cumulative estimate)

| Fix                          | Expected Δ J&F | New Total |
|------------------------------|----------------|-----------|
| Baseline (current)           | —              | 28.4%     |
| Higher res (448×448)         | +5 to +8%      | ~34-36%   |
| YouTube-VOS training         | +10 to +15%    | ~45-50%   |
| Object-ID injection fix      | +5 to +8%      | ~52-58%   |
| More epochs (100→200)        | +2 to +5%      | ~55-63%   |
| Stack 2 PropagationAttn layers| +3 to +5%     | ~58-68%   |
| Boundary refinement head     | +4 to +7%      | ~62-75%   |

**Target**: ≥ 75% J&F (within striking distance of AOTT) with full fixes.

