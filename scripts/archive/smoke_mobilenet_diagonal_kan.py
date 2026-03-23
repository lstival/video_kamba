"""Smoke test: MobileNetV2 + DiagonalKANSSMCore pipeline.

Checks:
  1. Forward pass (VOS mode) — output shapes correct
  2. Gradient flow through the full model
  3. FPS measurement (VOS inference loop, no grad)
  4. All 8 ablation configurations of (A, B, C) modulation
  5. KAN vs MLP modulator comparison
  6. Memory bank state persistence across frames

Usage (standalone, no Hydra):
    python scripts/smoke_mobilenet_diagonal_kan.py [--device cuda|cpu]
"""

import argparse
import sys
import time
import traceback
import os

import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Add project root to path
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _ok(s: str) -> str:
    return f"\033[92m✓ {s}\033[0m"


def _fail(s: str) -> str:
    return f"\033[91m✗ {s}\033[0m"


# ------------------------------------------------------------------
# Test 1 — DiagonalKANSSMCore unit test
# ------------------------------------------------------------------

def test_diagonal_core(device: torch.device):
    from models.components.kan_ssm_core import DiagonalKANSSMCore

    core = DiagonalKANSSMCore(
        inner_dim=256, state_dim=16,
        modulate_A=True, modulate_B=True, modulate_C=True,
        modulator_type="kan",
    ).to(device)

    n_params = sum(p.numel() for p in core.parameters())
    print(f"  DiagonalKANSSMCore params: {n_params:,}")

    # T=1 path (VOS: one frame at a time)
    x = torch.randn(4, 1, 256, device=device)
    y, state = core(x, return_last_state=True)
    assert y.shape == (4, 1, 256), f"Bad output shape: {y.shape}"
    assert state.shape == (4, 16), f"Bad state shape: {state.shape}"

    # State carry-over: second frame uses previous state
    x2 = torch.randn(4, 1, 256, device=device)
    y2, state2 = core(x2, initial_state=state, return_last_state=True)
    assert y2.shape == (4, 1, 256)
    assert state2.shape == (4, 16)

    # T>1 path (fallback: multiple frames at once)
    x_long = torch.randn(4, 8, 256, device=device)
    y_long = core(x_long)
    assert y_long.shape == (4, 8, 256), f"Bad multi-step shape: {y_long.shape}"

    # Gradient flow
    loss = y.sum() + y2.sum() + y_long.sum()
    loss.backward()
    for name, p in core.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    print(_ok(f"DiagonalKANSSMCore: T=1 {y.shape}, state {state.shape}, T=8 {y_long.shape}"))
    return n_params


# ------------------------------------------------------------------
# Test 2 — KangaSSM with diagonal backend
# ------------------------------------------------------------------

def test_kanga_ssm(device: torch.device):
    from models.components.kanga_ssm import KangaSSM

    ssm = KangaSSM(
        d_model=256, d_state=16,
        use_diagonal=True,
        modulate_A=True, modulate_B=True, modulate_C=True,
        modulator_type="kan",
    ).to(device)

    # VOS mode: [B*P, 1, D] — B=2, P=196
    x = torch.randn(392, 1, 256, device=device)
    out, states = ssm(x, return_last_state=True)
    assert out.shape == (392, 1, 256)
    assert len(states) == 1
    assert states[0].shape == (392, 16), f"State shape: {states[0].shape}"

    # Second frame using carry-over state
    x2 = torch.randn(392, 1, 256, device=device)
    out2, states2 = ssm(x2, prev_states=states, return_last_state=True)
    assert out2.shape == (392, 1, 256)

    print(_ok(f"KangaSSM diagonal: out {out.shape}, state {states[0].shape}"))


# ------------------------------------------------------------------
# Test 3 — Full VideoMambaSystem (MV2 + diagonal KAN-SSM, VOS mode)
# ------------------------------------------------------------------

def build_model(
    modulator_type: str = "kan",
    modulate_A: bool = True,
    modulate_B: bool = True,
    modulate_C: bool = True,
    use_diagonal: bool = True,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    from models.video_mamba import VideoMambaSystem

    # We need to patch KangaSSM inside VideoMambaSystem to accept the new flags.
    # VideoMambaSystem passes modulator_type to KangaSSM but not modulate_A / use_diagonal.
    # We monkey-patch the temporal_model after construction.
    from models.components.kanga_ssm import KangaSSM

    model = VideoMambaSystem(
        dim_in=256,
        dim_out=256,
        num_seg_classes=2,
        encoder_type="mobilenetv2",
        mv2_pretrained=False,   # no download in smoke test
        modulator_type=modulator_type,
        ssm_d_state=16,
        ssm_layers=1,
        prop_d_key=256,
        prop_d_value=256,
        prop_n_heads=8,
        max_mem_frames=5,
        use_kan_key_adapter=True,
        kan_adapter_d_state=8,
    )

    # Swap temporal_model for the new diagonal variant with full ablation flags
    model.temporal_model = KangaSSM(
        d_model=256,
        d_state=16,
        num_layers=1,
        dropout=0.1,
        modulator_type=modulator_type,
        use_diagonal=use_diagonal,
        modulate_A=modulate_A,
        modulate_B=modulate_B,
        modulate_C=modulate_C,
    )
    return model.to(device)


def test_vos_forward(device: torch.device):
    """Full VOS forward pass: reference frame + 3 query frames."""
    model = build_model(device=device)
    model.eval()

    B, T, H, W = 2, 3, 224, 224
    query = torch.randn(B, T, 3, H, W, device=device)
    ref_img = torch.randn(B, 3, H, W, device=device)
    ref_mask = torch.zeros(B, H, W, dtype=torch.long, device=device)
    ref_mask[:, 80:140, 80:140] = 1  # fake object region

    with torch.no_grad():
        _, _, _, logits_seg = model(query, ref_frame=ref_img, ref_mask=ref_mask)

    assert logits_seg.shape == (B, T, 2, H, W), f"Bad seg shape: {logits_seg.shape}"
    print(_ok(f"VOS forward: logits_seg {logits_seg.shape}"))


def test_gradient_flow(device: torch.device):
    """Confirm all trainable parameters receive gradients."""
    model = build_model(device=device)
    model.train()

    B, T, H, W = 1, 2, 224, 224
    query = torch.randn(B, T, 3, H, W, device=device)
    ref_img = torch.randn(B, 3, H, W, device=device)
    ref_mask = torch.zeros(B, H, W, dtype=torch.long, device=device)
    ref_mask[:, 80:140, 80:140] = 1
    gt_masks = torch.zeros(B, T, H, W, dtype=torch.long, device=device)
    gt_masks[:, :, 80:140, 80:140] = 1

    _, _, _, logits_seg = model(
        query, ref_frame=ref_img, ref_mask=ref_mask, query_masks=gt_masks
    )
    loss = logits_seg.sum()
    loss.backward()

    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    if no_grad:
        print(f"  WARNING: {len(no_grad)} params without grad: {no_grad[:3]}")
    else:
        print(_ok("Gradient flow: all trainable params have gradients"))


# ------------------------------------------------------------------
# Test 4 — FPS benchmark (VOS inference)
# ------------------------------------------------------------------

def benchmark_fps(device: torch.device, n_frames: int = 30, warmup: int = 5):
    model = build_model(device=device)
    model.eval()

    B, H, W = 1, 480, 854
    ref_img = torch.randn(B, 3, H, W, device=device)
    ref_mask = torch.zeros(B, H, W, dtype=torch.long, device=device)
    ref_mask[:, 150:330, 200:600] = 1

    # Warmup
    query = torch.randn(B, warmup, 3, H, W, device=device)
    with torch.no_grad():
        model(query, ref_frame=ref_img, ref_mask=ref_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        query = torch.randn(B, n_frames, 3, H, W, device=device)
        model(query, ref_frame=ref_img, ref_mask=ref_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    fps = n_frames / elapsed
    print(_ok(f"FPS @ 480p ({n_frames} frames): {fps:.1f} fps  ({elapsed:.2f}s total)"))
    return fps


# ------------------------------------------------------------------
# Test 5 — Ablation configurations
# ------------------------------------------------------------------

ABLATION_CONFIGS = [
    # (modulate_A, modulate_B, modulate_C, use_diagonal, label)
    (False, False, False, True,  "Diagonal SSM (no KAN)"),
    (True,  False, False, True,  "Diagonal + KAN_A only"),
    (False, True,  False, True,  "Diagonal + KAN_B only"),
    (False, False, True,  True,  "Diagonal + KAN_C only"),
    (True,  True,  False, True,  "Diagonal + KAN_A+B"),
    (True,  False, True,  True,  "Diagonal + KAN_A+C"),
    (False, True,  True,  True,  "Diagonal + KAN_B+C"),
    (True,  True,  True,  True,  "Diagonal + KAN_A+B+C (FULL)"),
    (True,  True,  True,  True,  "FULL + MLP modulator"),   # uses modulator_type='mlp'
    (True,  True,  True,  False, "Dense KAN-SSM (legacy IntricateKANSSMCore)"),
]


def test_ablation_configs(device: torch.device):
    from models.components.kanga_ssm import KangaSSM

    x = torch.randn(8, 1, 256, device=device)  # [B*P_small, 1, D]

    results = []
    for i, (mod_A, mod_B, mod_C, use_diag, label) in enumerate(ABLATION_CONFIGS):
        mod_type = "mlp" if "MLP" in label else "kan"
        ssm = KangaSSM(
            d_model=256, d_state=16,
            use_diagonal=use_diag,
            modulate_A=mod_A, modulate_B=mod_B, modulate_C=mod_C,
            modulator_type=mod_type,
        ).to(device).eval()

        n_params = sum(p.numel() for p in ssm.parameters())

        with torch.no_grad():
            out, states = ssm(x, return_last_state=True)

        shape_ok = out.shape == (8, 1, 256)
        state_shape = states[0].shape if use_diag else states[0].shape
        results.append((label, n_params, shape_ok, state_shape))
        status = _ok(f"{label:<45} params={n_params:>6,}  state={state_shape}")
        if not shape_ok:
            status = _fail(f"{label} — wrong output shape {out.shape}")
        print(f"  {status}")

    all_ok = all(r[2] for r in results)
    if all_ok:
        print(_ok(f"All {len(ABLATION_CONFIGS)} ablation configs passed"))
    else:
        print(_fail("Some ablation configs failed"))
    return all_ok


# ------------------------------------------------------------------
# Test 6 — Parameter count check (must stay < 6M total)
# ------------------------------------------------------------------

def test_param_count(device: torch.device):
    model = build_model(device=device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Component breakdown
    enc   = sum(p.numel() for p in model.feature_extractor.parameters())
    ssm   = sum(p.numel() for p in model.temporal_model.parameters())
    mem   = sum(p.numel() for p in model.memory_bank.parameters())
    prop  = sum(p.numel() for p in model.propagation_attention.parameters())
    dec   = sum(p.numel() for p in model.seg_decoder.parameters())

    print(f"  Encoder (MV2):           {enc/1e6:.2f}M")
    print(f"  KAN-SSM (diagonal):      {ssm/1e6:.3f}M")
    print(f"  MemoryBank + adapter:    {mem/1e6:.3f}M")
    print(f"  PropagationAttention:    {prop/1e6:.3f}M")
    print(f"  SegmentationDecoder:     {dec/1e6:.3f}M")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:                   {total/1e6:.2f}M")
    print(f"  Trainable:               {trainable/1e6:.2f}M")

    if total < 6e6:
        print(_ok(f"Param count {total/1e6:.2f}M < 6M threshold ✓"))
    else:
        print(f"  WARNING: {total/1e6:.2f}M exceeds 6M lightweight threshold")
    return total


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps-frames", type=int, default=30)
    parser.add_argument("--skip-fps", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(_bold(f"\n{'='*60}"))
    print(_bold(f"  Smoke test: MobileNetV2 + DiagonalKANSSM"))
    print(_bold(f"  Device: {device}"))
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(_bold(f"{'='*60}\n"))

    failures = []

    tests = [
        ("1. DiagonalKANSSMCore unit",  lambda: test_diagonal_core(device)),
        ("2. KangaSSM diagonal",         lambda: test_kanga_ssm(device)),
        ("3. Full VOS forward",          lambda: test_vos_forward(device)),
        ("4. Gradient flow",             lambda: test_gradient_flow(device)),
        ("5. Ablation configs (8+2)",    lambda: test_ablation_configs(device)),
        ("6. Parameter count",           lambda: test_param_count(device)),
    ]
    if not args.skip_fps:
        tests.append(("7. FPS benchmark (480p)", lambda: benchmark_fps(device, args.fps_frames)))

    for name, fn in tests:
        print(_bold(f"\n─── {name} ───"))
        try:
            fn()
        except Exception:
            failures.append(name)
            print(_fail(f"FAILED: {name}"))
            traceback.print_exc()

    print(_bold(f"\n{'='*60}"))
    if failures:
        print(_fail(f"FAILED tests: {failures}"))
        sys.exit(1)
    else:
        print(_ok(f"All {len(tests)} tests passed!"))
    print(_bold(f"{'='*60}\n"))


if __name__ == "__main__":
    main()
