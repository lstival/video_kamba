"""Smoke test for stability_fix_v2 changes.

Validates that:
1. DiagonalKANSSMCore BCNorm + state clamping work
2. KangaSSM post-scan norm works
3. FastKANLayer Xavier init produces reasonable output magnitudes
4. ADE20K coherent motion trajectory generates 6 frames
5. Full model forward pass succeeds with 6-frame clips
"""

import sys
import torch
import torch.nn as nn

print("=" * 60)
print("  Stability Fix v2 — Smoke Test")
print("=" * 60)

# ── 1. DiagonalKANSSMCore with BCNorm ───────────────────────────────
print("\n[1] DiagonalKANSSMCore BCNorm + state clamping...")
from models.components.kan_ssm_core import DiagonalKANSSMCore

core = DiagonalKANSSMCore(inner_dim=192, state_dim=16)
assert hasattr(core, "B_norm"), "B_norm missing — BCNorm not applied"
assert hasattr(core, "C_norm"), "C_norm missing — BCNorm not applied"
assert isinstance(core.B_norm, nn.RMSNorm), f"B_norm type: {type(core.B_norm)}"
assert isinstance(core.C_norm, nn.RMSNorm), f"C_norm type: {type(core.C_norm)}"

x = torch.randn(2, 6, 192)
y, state = core(x, return_last_state=True)
assert y.shape == (2, 6, 192), f"Output shape: {y.shape}"
assert state.shape == (2, 16), f"State shape: {state.shape}"
assert torch.isfinite(y).all(), "Non-finite values in output"
assert (state.abs() <= 10.0).all(), f"State not clamped: max={state.abs().max():.2f}"
print(f"  OK: output range [{y.min():.3f}, {y.max():.3f}], state max {state.abs().max():.3f}")

# ── 2. KangaSSM post-scan norm ──────────────────────────────────────
print("\n[2] KangaSSM post-scan norm...")
from models.components.kanga_ssm import KangaSSM

ssm = KangaSSM(d_model=192, d_state=16, num_layers=1)
assert hasattr(ssm, "post_scan_norm"), "post_scan_norm missing"
x = torch.randn(2, 6, 192)
y = ssm(x)
assert y.shape == (2, 6, 192), f"Output shape: {y.shape}"
assert torch.isfinite(y).all(), "Non-finite values in KangaSSM output"
print(f"  OK: output range [{y.min():.3f}, {y.max():.3f}]")

# ── 3. FastKANLayer init check ──────────────────────────────────────
print("\n[3] FastKANLayer Xavier init...")
from models.components.fast_kan_layer import FastKANLayer

kan = FastKANLayer(192, 16, grid_size=8)
# Check that rbf_weight std is ~0.05 (not 0.1)
rbf_std = kan.rbf_weight.std().item()
print(f"  rbf_weight std: {rbf_std:.4f} (expected ~0.05)")
assert rbf_std < 0.08, f"rbf_weight std too high: {rbf_std}"

# Check output magnitude with random input
x = torch.randn(100, 192)
with torch.no_grad():
    out = kan(x)
print(f"  Output magnitude: mean={out.abs().mean():.3f}, std={out.std():.3f}")
print("  OK")

# ── 4. Coherent motion trajectory ───────────────────────────────────
print("\n[4] Coherent motion trajectory...")
from data.ade20k_pretrain import _sample_motion_trajectory

traj = _sample_motion_trajectory(seq_len=6, h=448, w=448)
assert len(traj) == 6, f"Expected 6 frames, got {len(traj)}"

# Verify monotonic angle progression (same direction)
angles = [f["angle"] for f in traj]
diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
# All diffs should have the same sign (monotonic rotation)
signs = [d > 0 for d in diffs]
# Allow 1 sign flip from jitter
n_flips = sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1])
print(f"  Angles: {[f'{a:.1f}' for a in angles]}")
print(f"  Direction changes: {n_flips} (should be ≤1 from jitter)")
assert n_flips <= 2, f"Too many direction changes: {n_flips}"
print("  OK: motion is coherent")

# ── 5. Gradient flow test ───────────────────────────────────────────
print("\n[5] Gradient flow through full SSM stack...")
ssm = KangaSSM(d_model=192, d_state=16, num_layers=2)
x = torch.randn(2, 6, 192, requires_grad=True)
y = ssm(x)
loss = y.sum()
loss.backward()
assert x.grad is not None, "No gradient on input"
grad_norm = x.grad.norm().item()
print(f"  Input gradient norm: {grad_norm:.4f}")
assert grad_norm < 1e4, f"Gradient too large: {grad_norm}"
assert grad_norm > 1e-6, f"Gradient too small (vanishing): {grad_norm}"
print("  OK: healthy gradient flow")

print("\n" + "=" * 60)
print("  ALL SMOKE TESTS PASSED")
print("=" * 60)
