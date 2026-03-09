"""Local debug script for diagnosing a flat training loss curve.

Run from the project root:
    python scripts/debug_local.py

Checks:
  1. Batch dispatch — confirms DAVIS 4-tuple hits the right path
  2. Loss sanity — expected range for cross-entropy on 11 classes
  3. Gradient norms per component — are memory bank / attention / decoder / SSM actually learning?
  4. Loss over 5 mini-steps — is there any downward signal at all?
  5. KAN gate statistics — has the gate escaped its 0.5-neutral initialization?
  6. PropagationAttention entropy — is attention concentrated or uniform over memory?
  7. SSM contribution — does the SSM change the representation (ablate it away)?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
B, T, H, W = 2, 4, 224, 224   # small enough to run on CPU in <60 s
NUM_SEG = 11                    # must match model default
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4

print(f"\n{'='*64}")
print(f"  VOS Debug  |  device={DEVICE}  B={B} T={T} H={H} W={W}")
print(f"{'='*64}\n")

# ── Build model ───────────────────────────────────────────────────────────────
from models.video_mamba import VideoMambaSystem

model = VideoMambaSystem(
    num_seg_classes=NUM_SEG,
    prop_d_key=256,
    prop_d_value=256,
    prop_n_heads=8,
    max_mem_frames=5,
    memory_update_freq=1,
    fusion_mode="kan_spatial",
    modulator_type="kan",
    learning_rate=LR,
).to(DEVICE)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

# ── Fake DAVIS batch ──────────────────────────────────────────────────────────
def make_batch():
    ref_img   = torch.randn(B, 3, H, W, device=DEVICE)
    ref_mask  = torch.randint(0, 3, (B, H, W), device=DEVICE)       # [B, H, W] int
    q_imgs    = torch.randn(B, T, 3, H, W, device=DEVICE)
    q_masks   = torch.randint(0, 3, (B, T, H, W), device=DEVICE)    # int labels
    return ref_img, ref_mask, q_imgs, q_masks

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — batch dispatch
# ─────────────────────────────────────────────────────────────────────────────
print("── CHECK 1: Batch dispatch ──────────────────────────────────")
batch = make_batch()
is_vos = model._is_vos_batch(batch)
is_davis = (
    len(batch) == 4
    and batch[0].dim() == 4
    and batch[1].dim() == 3
    and batch[2].dim() == 5
)
print(f"  _is_vos_batch  : {is_vos}   (expected False for DAVIS 4-tuple)")
print(f"  DAVIS 4-tuple  : {is_davis}  (expected True)")
if not is_davis or is_vos:
    print("  ❌ WRONG PATH — batch is not routing to the DAVIS cross-entropy branch!")
else:
    print("  ✅ Routing to DAVIS cross-entropy path")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — loss sanity
# ─────────────────────────────────────────────────────────────────────────────
print("\n── CHECK 2: Loss sanity ─────────────────────────────────────")
import math
random_ce = math.log(NUM_SEG)
print(f"  Expected CE for random predictions ({NUM_SEG} classes): {random_ce:.4f}")
print(f"  If observed train_loss >> {random_ce:.1f}: magnitude issue")
print(f"  If observed train_loss == {random_ce:.1f}: model predicts uniform (not learning)")
print(f"  If observed train_loss < {random_ce:.1f}: model is learning class priors at minimum")
# From Comet: train_loss ~0.55 — WAY below random. The model learned background dominance.
# But val_loss 0.33 < train_loss 0.55 — train uses label_smoothing=0.1, val does not.
upper = random_ce + 0.2
# Note: train uses label_smoothing=0.1 which shifts CE baseline upward slightly
print(f"  Observed train_loss ~0.55, val_loss ~0.33 ← label_smoothing=0.1 on train")
print(f"  This means the model learned background class dominance but not object detail.\n")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — gradient norms per component
# ─────────────────────────────────────────────────────────────────────────────
print("── CHECK 3: Gradient norms per component ────────────────────")
ref_img, ref_mask, q_imgs, q_masks = make_batch()
optimizer.zero_grad()
_, _, _, logits_seg = model(q_imgs, ref_frame=ref_img, ref_mask=ref_mask, query_masks=q_masks)

BT = B * T
C = logits_seg.shape[2]
loss = F.cross_entropy(
    logits_seg.view(BT, C, H, W),
    q_masks.view(BT, H, W),
    ignore_index=255,
    label_smoothing=0.1,
)
loss.backward()

components = {
    "memory_bank.proj_key":     model.memory_bank.proj_key,
    "memory_bank.proj_value":   model.memory_bank.proj_value,
    "memory_bank.id_embeddings": model.memory_bank.id_embeddings,
    "prop_attention.W_q":       model.propagation_attention.W_q,
    "prop_attention.proj_out":  model.propagation_attention.proj_out,
    "temporal_model":           model.temporal_model,
    "seg_decoder.up3":          model.seg_decoder.up3,
    "seg_decoder.final_head":   model.seg_decoder.final_head,
}
# Add KAN gate from first decoder up-block (up1)
if hasattr(model.seg_decoder, "up1"):
    up1 = model.seg_decoder.up1
    if hasattr(up1, "gate_kan"):
        components["seg_decoder.up1.gate_kan.rbf_weight"] = up1.gate_kan
        components["seg_decoder.up1.proj"] = up1.proj

dead_components = []
for name, module in components.items():
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        total_norm = 0.0
    else:
        total_norm = sum(g.norm().item() ** 2 for g in grads) ** 0.5
    status = "✅" if total_norm > 1e-6 else "❌ DEAD"
    if total_norm <= 1e-6:
        dead_components.append(name)
    print(f"  {status}  {name:45s}  grad_norm={total_norm:.6f}")

if dead_components:
    print(f"\n  ❌ Dead gradients in: {dead_components}")
else:
    print("\n  ✅ All components receive gradients")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — does loss decrease over 5 steps?
# ─────────────────────────────────────────────────────────────────────────────
print("\n── CHECK 4: Loss over 5 optimizer steps ─────────────────────")
losses = []
for step in range(5):
    ref_img, ref_mask, q_imgs, q_masks = make_batch()
    optimizer.zero_grad()
    _, _, _, logits_seg = model(q_imgs, ref_frame=ref_img, ref_mask=ref_mask, query_masks=q_masks)
    loss = F.cross_entropy(
        logits_seg.view(B * T, C, H, W),
        q_masks.view(B * T, H, W),
        ignore_index=255,
        label_smoothing=0.1,
    )
    loss.backward()

    # Check gradient norm BEFORE clipping
    total_grad = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    losses.append(loss.item())
    print(f"  step {step+1}  loss={loss.item():.4f}  pre-clip grad_norm={total_grad:.2f}")

if losses[-1] < losses[0]:
    print(f"  ✅ Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
else:
    print(f"  ❌ Loss did NOT decrease: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"     The model might be stuck in a local minimum or LR is wrong.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — KAN gate statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n── CHECK 5: KAN gate activation statistics ──────────────────")
model.eval()
with torch.no_grad():
    ref_img, ref_mask, q_imgs, _ = make_batch()
    # Hook into the gate of up1 to capture output
    gate_outputs = []
    def _hook(module, inp, out):
        # out: [BT, skip_ch, H, W] from out_conv, but we want the gate
        pass

    up1 = model.seg_decoder.up1
    up1.return_gate = True
    _ = model(q_imgs, ref_frame=ref_img, ref_mask=ref_mask)
    gate = up1._last_gate   # [BT, skip_ch, H, W] or None
    up1.return_gate = False

if gate is not None:
    g_mean = gate.mean().item()
    g_std  = gate.std().item()
    g_min  = gate.min().item()
    g_max  = gate.max().item()
    print(f"  gate mean={g_mean:.4f}  std={g_std:.4f}  min={g_min:.4f}  max={g_max:.4f}")
    if g_std < 0.02:
        print("  ❌ Gate is collapsed (std<0.02) — uniform ≈0.5 everywhere, not learning")
    elif 0.45 < g_mean < 0.55 and g_std < 0.05:
        print("  ⚠️  Gate mean ≈0.5 with low std — partially neutral, still early in training")
    else:
        print("  ✅ Gate has meaningful variation — spatially selective")
else:
    print("  ⚠️  Could not capture gate (return_gate not implemented on this decoder block)")

model.train()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — PropagationAttention entropy (after short warm-up)
# We simulate learning on a fixed batch to test if Q-K alignment emerges.
# On a freshly initialized model with random inputs entropy will always be ~1.0
# regardless of LR — alignment requires gradient signal from real structure.
# We therefore do N warm-up steps on a FIXED batch first, then measure.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── CHECK 6: PropagationAttention entropy (after 50-step warm-up) ──")
# Re-init a clean model with the same config but the differential-LR optimizer
model6 = VideoMambaSystem(
    num_seg_classes=NUM_SEG,
    prop_d_key=256,
    prop_d_value=256,
    prop_n_heads=8,
    max_mem_frames=5,
    memory_update_freq=1,
    fusion_mode="kan_spatial",
    modulator_type="kan",
    learning_rate=LR,
).to(DEVICE)

# Replicate differential-LR optimizer (the fix under test)
prop_params6  = list(model6.memory_bank.parameters()) + list(model6.propagation_attention.parameters())
prop_ids6     = {id(p) for p in prop_params6}
base_params6  = [p for p in model6.parameters() if id(p) not in prop_ids6]
optimizer6 = torch.optim.AdamW(
    [{"params": prop_params6, "lr": LR * 10}, {"params": base_params6, "lr": LR}],
    weight_decay=1e-2,
)

# Fixed batch — same content repeated so Q-K can learn to match it
fixed_ref_img, fixed_ref_mask, fixed_q_imgs, fixed_q_masks = make_batch()

model6.train()
for warm_step in range(50):
    optimizer6.zero_grad()
    model6.memory_bank.reset()
    _, _, _, logits6 = model6(fixed_q_imgs, ref_frame=fixed_ref_img,
                              ref_mask=fixed_ref_mask, query_masks=fixed_q_masks)
    loss6 = F.cross_entropy(
        logits6.view(B * T, NUM_SEG, H, W),
        fixed_q_masks.view(B * T, H, W),
        ignore_index=255,
    )
    loss6.backward()
    torch.nn.utils.clip_grad_norm_(model6.parameters(), 1.0)
    optimizer6.step()
print(f"  Warm-up loss after 50 steps: {loss6.item():.4f}")

# Now measure entropy on the warmed-up model
model6.eval()
attn_weights_collector = []
_orig_forward6 = model6.propagation_attention.forward

def _patched_forward6(query_feat, memory_K, memory_V):
    pa = model6.propagation_attention
    B_, P_, _ = query_feat.shape
    MP_ = memory_K.shape[1]
    Q = pa.W_q(pa.norm_q(query_feat))
    Q = Q.view(B_, P_, pa.n_heads, pa.head_dim_k).transpose(1, 2)
    K = memory_K.view(B_, MP_, pa.n_heads, pa.head_dim_k).transpose(1, 2)
    V = memory_V.view(B_, MP_, pa.n_heads, pa.head_dim_v).transpose(1, 2)
    attn = torch.matmul(Q, K.transpose(-2, -1)) / pa.scale
    attn_w = F.softmax(attn, dim=-1)
    attn_weights_collector.append(attn_w.detach().cpu())
    out = torch.matmul(attn_w, V)
    out = out.transpose(1, 2).reshape(B_, P_, -1)
    out = pa.proj_out(out)
    out = pa.norm_out(out + query_feat)
    return out

model6.propagation_attention.forward = _patched_forward6
with torch.no_grad():
    model6.memory_bank.reset()
    _ = model6(fixed_q_imgs, ref_frame=fixed_ref_img, ref_mask=fixed_ref_mask)
model6.propagation_attention.forward = _orig_forward6

if attn_weights_collector:
    # Each element: [B, n_heads, P, MP] — MP grows as memory fills, so compute per frame
    eps = 1e-9
    max_entropies = []
    norm_entropies = []
    for aw in attn_weights_collector:
        entropy = -(aw * (aw + eps).log()).sum(dim=-1)  # [B, n_heads, P]
        max_entropy = torch.log(torch.tensor(aw.shape[-1], dtype=torch.float))
        if max_entropy > 0:
            norm_entropies.append((entropy / max_entropy).mean().item())
    norm_entropy = sum(norm_entropies) / len(norm_entropies) if norm_entropies else 0.0

    print(f"  Normalised attention entropy: {norm_entropy:.4f}  (0=focused, 1=uniform)")
    if norm_entropy > 0.96:
        print("  ❌ No Q-K alignment at all — differential LR fix not wired correctly")
    elif norm_entropy > 0.85:
        print("  ⚠️  Alignment starting (expected on random data with short warm-up)")
        print(f"     Dropped from ~0.978 baseline → {norm_entropy:.3f}. ✅ Fix is active.")
        print(f"     On real DAVIS data with full training, expect entropy < 0.75.")
    else:
        print("  ✅ Attention is reasonably focused")
else:
    print("  ⚠️  No attention weights captured")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — SSM contribution (ablation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── CHECK 7: SSM contribution (identity ablation) ────────────")
model.eval()
with torch.no_grad():
    ref_img, ref_mask, q_imgs, _ = make_batch()

    # Normal forward
    _, _, _, logits_normal = model(q_imgs, ref_frame=ref_img, ref_mask=ref_mask)

    # Monkey-patch SSM to identity: output = input
    orig_ssm_forward = model.temporal_model.forward
    def _identity_ssm(x, prev_states=None, return_last_state=False):
        if return_last_state:
            return x, [None]
        return x
    model.temporal_model.forward = _identity_ssm
    model.memory_bank.reset()

    _, _, _, logits_no_ssm = model(q_imgs, ref_frame=ref_img, ref_mask=ref_mask)
    model.temporal_model.forward = orig_ssm_forward

# Measure how much the SSM changes predictions
diff = (logits_normal - logits_no_ssm).abs().mean().item()
prob_diff = (
    F.softmax(logits_normal, dim=2) - F.softmax(logits_no_ssm, dim=2)
).abs().mean().item()
print(f"  Logit diff with vs without SSM:       {diff:.6f}")
print(f"  Probability diff with vs without SSM: {prob_diff:.6f}")
if prob_diff < 1e-4:
    print("  ❌ SSM has no effect — temporal model is essentially a skip connection")
elif prob_diff < 0.01:
    print("  ⚠️  SSM has minimal effect — contributes little to predictions")
else:
    print("  ✅ SSM meaningfully changes predictions")

model.train()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  SUMMARY")
print(f"{'='*64}")
print(f"  device        : {DEVICE}")
print(f"  B×T           : {B}×{T}")
print(f"  Loss (step 1) : {losses[0]:.4f}")
print(f"  Loss (step 5) : {losses[-1]:.4f}")
print(f"  Gate std      : {g_std:.4f}" if gate is not None else "  Gate          : not captured")
print(f"  Attn entropy  : {norm_entropy:.4f}  (normalised, 0=focused 1=uniform)" if attn_weights_collector else "  Attn entropy  : not captured")
print(f"  SSM prob_diff : {prob_diff:.6f}")
print(f"\nHint — flat loss usually means one of:")
print(f"  A) Model converged to predicting background only (check class-wise IoU)")
print(f"  B) LR too high → loss oscillates around saddle (try 1e-5)")
print(f"  C) BatchNorm running stats are stale (model.train() not called)")
print(f"  D) PropagationAttention is uniform → memory keys not discriminative")
print(f"  E) SSM not contributing → temporal context bypassed\n")
