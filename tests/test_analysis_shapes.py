"""Pytest test suite for ablation analysis utilities.

Tests tensor shapes, value ranges, and data-integrity invariants for the
core computation functions in the three analysis scripts.

All tests are CPU-only and use small synthetic tensors — no checkpoint or
dataset is required.

Run with::

    pytest tests/test_analysis_shapes.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn


# ============================================================================
# Experiment 1 helpers
# ============================================================================

from scripts.exp1_activation_entropy import compute_shannon_entropy


class TestComputeShannonEntropy:
    """Shape, range, and edge-case tests for compute_shannon_entropy."""

    def test_output_shape(self) -> None:
        """Entropy output should have one value per sample."""
        N, D = 64, 768
        activations = torch.rand(N, D) + 0.1
        entropy = compute_shannon_entropy(activations)
        assert entropy.shape == (N,), f"Expected ({N},), got {entropy.shape}"

    def test_non_negative(self) -> None:
        """Shannon entropy is always ≥ 0."""
        activations = torch.rand(32, 128) + 0.1
        entropy = compute_shannon_entropy(activations)
        assert (entropy >= 0).all(), "Entropy must be non-negative"

    def test_upper_bound(self) -> None:
        """Entropy cannot exceed log(D) (maximum for uniform distribution)."""
        N, D = 16, 256
        activations = torch.rand(N, D) + 0.1
        entropy = compute_shannon_entropy(activations)
        max_entropy = math.log(D)
        assert (entropy <= max_entropy + 1e-5).all(), (
            f"Entropy exceeded log(D)={max_entropy:.4f}"
        )

    def test_uniform_distribution_is_max_entropy(self) -> None:
        """A perfectly uniform activation vector should achieve near-maximum entropy."""
        D = 64
        uniform = torch.ones(1, D)
        entropy = compute_shannon_entropy(uniform)
        expected = math.log(D)
        assert abs(entropy.item() - expected) < 1e-4, (
            f"Expected entropy ≈ {expected:.4f}, got {entropy.item():.4f}"
        )

    def test_one_hot_is_zero_entropy(self) -> None:
        """A one-hot activation vector should have entropy ≈ 0."""
        D = 64
        one_hot = torch.zeros(1, D)
        one_hot[0, 0] = 1.0
        entropy = compute_shannon_entropy(one_hot)
        assert entropy.item() < 1e-4, (
            f"One-hot entropy should be ≈ 0, got {entropy.item():.6f}"
        )

    def test_negative_values_handled(self) -> None:
        """Negative activations should be handled via absolute value."""
        activations = torch.randn(8, 32)  # may contain negatives
        entropy = compute_shannon_entropy(activations)
        assert entropy.shape == (8,)
        assert (entropy >= 0).all()

    def test_zero_activation_row_no_nan(self) -> None:
        """An all-zero row should not produce NaN (clamp in denominator)."""
        activations = torch.zeros(1, 64)
        entropy = compute_shannon_entropy(activations)
        assert not torch.isnan(entropy).any(), "NaN in entropy for zero activation"


# ============================================================================
# Experiment 2 helpers
# ============================================================================

from scripts.exp2_spatial_gate_iou import compute_iou_gate, gate_to_spatial_map


class TestComputeIoUGate:
    """Shape, range, and correctness tests for compute_iou_gate."""

    def test_output_shape(self) -> None:
        """IoU_gate should return one scalar per sample."""
        N, H, W = 8, 14, 14
        gate_map = torch.rand(N, 1, H, W)
        gt_mask = (torch.rand(N, H, W) > 0.5).long()
        iou = compute_iou_gate(gate_map, gt_mask, tau=0.5)
        assert iou.shape == (N,), f"Expected ({N},), got {iou.shape}"

    def test_range_zero_to_one(self) -> None:
        """IoU values must lie in [0, 1]."""
        N, H, W = 4, 28, 28
        gate_map = torch.rand(N, 1, H, W)
        gt_mask = (torch.rand(N, H, W) > 0.3).long()
        iou = compute_iou_gate(gate_map, gt_mask, tau=0.5)
        assert (iou >= 0).all() and (iou <= 1).all(), "IoU must be in [0, 1]"

    def test_perfect_match_returns_one(self) -> None:
        """When the thresholded gate exactly matches the GT mask, IoU = 1."""
        N, H, W = 2, 14, 14
        mask = (torch.rand(N, H, W) > 0.5).long()
        # Gate above threshold where mask is 1, below where mask is 0
        gate_map = mask.float().unsqueeze(1) * 0.9 + (1 - mask.float()).unsqueeze(1) * 0.1
        iou = compute_iou_gate(gate_map, mask, tau=0.5)
        assert (iou > 0.99).all(), f"Expected IoU ≈ 1, got {iou}"

    def test_no_match_returns_zero(self) -> None:
        """When gate and mask are completely disjoint, IoU = 0."""
        N, H, W = 2, 14, 14
        mask = torch.zeros(N, H, W).long()
        mask[:, :H // 2, :] = 1
        # Gate fires only in bottom half
        gate_map = torch.zeros(N, 1, H, W)
        gate_map[:, :, H // 2:, :] = 1.0
        iou = compute_iou_gate(gate_map, mask, tau=0.5)
        assert (iou == 0).all(), f"Expected IoU = 0, got {iou}"

    def test_empty_gt_mask_no_crash(self) -> None:
        """All-zero GT mask (no object) should not crash."""
        N, H, W = 2, 14, 14
        gate_map = torch.rand(N, 1, H, W)
        gt_mask = torch.zeros(N, H, W).long()
        iou = compute_iou_gate(gate_map, gt_mask, tau=0.5)
        assert iou.shape == (N,)
        assert not torch.isnan(iou).any()


class TestGateToSpatialMap:
    """Shape tests for gate_to_spatial_map."""

    def test_kan_gate_output_shape(self) -> None:
        """KAN gate [BT, C, H, W] → [BT, 1, H_t, W_t]."""
        BT, C, H, W = 4, 768, 14, 14
        gate = torch.rand(BT, C, H, W)
        out = gate_to_spatial_map(gate, (28, 28), model_type="kan", bt=BT)
        assert out.shape == (BT, 1, 28, 28), f"Shape mismatch: {out.shape}"

    def test_xattn_gate_output_shape(self) -> None:
        """Cross-attention map [B, N_q, N_kv] → [BT, 1, H_t, W_t]."""
        B, N = 2, 196  # 14*14
        T = 2
        attn = torch.rand(B, N, N)
        out = gate_to_spatial_map(attn, (14, 14), model_type="xattn", bt=B * T)
        assert out.shape == (B * T, 1, 14, 14), f"Shape mismatch: {out.shape}"

    def test_output_in_zero_one(self) -> None:
        """Normalised spatial map should be in [0, 1]."""
        gate = torch.rand(4, 128, 14, 14) * 5 + 2  # large range
        out = gate_to_spatial_map(gate, (14, 14), model_type="kan", bt=4)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6


# ============================================================================
# Experiment 3 helpers
# ============================================================================

from models.components.fast_kan_layer import FastKANLayer
from scripts.exp3_rbf_profile import reconstruct_1d_function, relu_reference


class TestReconstructRBF:
    """Shape and consistency tests for reconstruct_1d_function."""

    @pytest.fixture
    def small_kan(self) -> FastKANLayer:
        return FastKANLayer(in_features=16, out_features=16, grid_size=4)

    def test_output_length_matches_x_points(self, small_kan: FastKANLayer) -> None:
        """Both returned arrays must have length == n_points."""
        x_grid, y_vals = reconstruct_1d_function(
            small_kan, channel_in=0, channel_out=0,
            x_range=(-3.0, 3.0), n_points=200,
        )
        assert len(x_grid) == 200
        assert len(y_vals) == 200

    def test_x_grid_sorted(self, small_kan: FastKANLayer) -> None:
        """x_grid must be monotonically increasing."""
        x_grid, _ = reconstruct_1d_function(
            small_kan, channel_in=0, channel_out=0,
            x_range=(-3.0, 3.0), n_points=100,
        )
        assert (np.diff(x_grid) > 0).all(), "x_grid not monotonically increasing"

    def test_no_nan_in_output(self, small_kan: FastKANLayer) -> None:
        """Reconstructed function should not contain NaN."""
        x_grid, y_vals = reconstruct_1d_function(
            small_kan, channel_in=0, channel_out=0,
            x_range=(-3.0, 3.0), n_points=100,
        )
        assert not np.isnan(y_vals).any(), "NaN found in reconstructed RBF function"

    def test_x_range_respected(self, small_kan: FastKANLayer) -> None:
        """x_grid must lie within the specified x_range."""
        x_range = (-2.0, 2.0)
        x_grid, _ = reconstruct_1d_function(
            small_kan, channel_in=0, channel_out=0,
            x_range=x_range, n_points=100,
        )
        assert x_grid.min() >= x_range[0] - 1e-6
        assert x_grid.max() <= x_range[1] + 1e-6


class TestReluReference:
    """Tests for relu_reference utility."""

    def test_non_negative_input(self) -> None:
        """ReLU of non-negative input is the input itself."""
        x = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(relu_reference(x), x)

    def test_negative_input_clamped(self) -> None:
        """ReLU of negative input is zero."""
        x = np.array([-1.0, -0.5, -0.1])
        np.testing.assert_array_almost_equal(relu_reference(x), np.zeros_like(x))


# ============================================================================
# Model extension tests
# ============================================================================

class TestMLPModulator:
    """Shape and interface tests for the new MLPModulator class."""

    def test_output_shape(self) -> None:
        """MLPModulator output must match (B, output_dim)."""
        from models.components.kan_ssm_core import MLPModulator
        mod = MLPModulator(input_dim=64, output_dim=64)
        x = torch.randn(8, 64)
        out = mod(x)
        assert out.shape == (8, 64), f"Unexpected shape: {out.shape}"

    def test_output_positive_softplus(self) -> None:
        """With softplus activation, all outputs must be positive."""
        from models.components.kan_ssm_core import MLPModulator
        mod = MLPModulator(input_dim=32, output_dim=32, activation="softplus")
        x = torch.randn(4, 32)
        out = mod(x)
        assert (out > 0).all(), "softplus output must be strictly positive"

    def test_regularization_loss_scalar(self) -> None:
        """regularization_loss must return a scalar tensor."""
        from models.components.kan_ssm_core import MLPModulator
        mod = MLPModulator(input_dim=32, output_dim=32)
        loss = mod.regularization_loss()
        assert loss.ndim == 0, "regularization_loss must be scalar"

    def test_grid_size_kwarg_accepted(self) -> None:
        """grid_size kwarg must be accepted without error (API parity)."""
        from models.components.kan_ssm_core import MLPModulator
        mod = MLPModulator(input_dim=16, output_dim=16, grid_size=8)
        out = mod(torch.randn(2, 16))
        assert out.shape == (2, 16)


class TestReturnGateFlag:
    """Tests that _last_gate is populated when return_gate=True."""

    def test_kan_block_stores_gate(self) -> None:
        """KANSpatialGatingUpBlock._last_gate populated after forward()."""
        from models.components.segmentation_decoder import KANSpatialGatingUpBlock
        block = KANSpatialGatingUpBlock(
            in_channels=64, skip_channels=64, out_channels=32, return_gate=True
        )
        x = torch.randn(2, 64, 7, 7)
        skip = torch.randn(2, 64, 14, 14)
        _ = block(x, skip)
        assert block._last_gate is not None, "_last_gate should be set"
        assert block._last_gate.shape == (2, 64, 14, 14), (
            f"Unexpected gate shape: {block._last_gate.shape}"
        )

    def test_kan_block_no_gate_when_disabled(self) -> None:
        """_last_gate must stay None when return_gate=False (default)."""
        from models.components.segmentation_decoder import KANSpatialGatingUpBlock
        block = KANSpatialGatingUpBlock(
            in_channels=32, skip_channels=32, out_channels=16
        )
        x = torch.randn(1, 32, 7, 7)
        skip = torch.randn(1, 32, 14, 14)
        _ = block(x, skip)
        assert block._last_gate is None, "_last_gate should remain None by default"

    def test_gate_detached_from_graph(self) -> None:
        """Cached gate must not require gradients (detached)."""
        from models.components.segmentation_decoder import KANSpatialGatingUpBlock
        block = KANSpatialGatingUpBlock(
            in_channels=32, skip_channels=32, out_channels=16, return_gate=True
        )
        x = torch.randn(1, 32, 7, 7, requires_grad=True)
        skip = torch.randn(1, 32, 14, 14)
        _ = block(x, skip)
        assert not block._last_gate.requires_grad, "Cached gate must not require grad"
