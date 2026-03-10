"""pytest tests for KANKeyAdapter and the MemoryBank + adapter integration.

Run with:
    pytest tests/test_kan_key_adapter.py -v

All tests use synthetic CPU tensors — no dataset download required.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_KEY   = 256  # match MemoryBank default d_key
D_STATE = 4
B       = 2
P       = 16   # number of patches
D_MODEL = 256  # d_model fed to MemoryBank (matches d_key for the isolated test)


@pytest.fixture(scope="module")
def memory_bank_with_adapter():
    from models.components.kan_key_adapter import KANKeyAdapter
    from models.components.memory_bank import MemoryBank
    adapter = KANKeyAdapter(d_key=D_KEY, d_state=D_STATE, num_layers=1)
    bank = MemoryBank(
        d_model=D_MODEL,
        d_model_fine=D_MODEL,
        d_key=D_KEY,
        d_value=D_KEY,
        n_objects=4,
        max_mem_frames=5,
        use_dual_scale=False,  # single-scale for simplicity
        key_adapter=adapter,
    )
    return bank


# Standalone adapter fixture — derived from the bank so weights are shared
@pytest.fixture(scope="module")
def adapter(memory_bank_with_adapter):
    return memory_bank_with_adapter.key_adapter


# ---------------------------------------------------------------------------
# KANKeyAdapter shape tests
# ---------------------------------------------------------------------------

class TestKANKeyAdapterShapes:

    def test_output_shape(self, adapter):
        """adapt() must return a tensor of the same shape as input key."""
        adapter.reset()
        key = torch.randn(B, P, D_KEY)
        correction = adapter.adapt(key)
        assert correction.shape == (B, P, D_KEY), (
            f"Expected ({B}, {P}, {D_KEY}), got {correction.shape}"
        )

    def test_output_dtype(self, adapter):
        """Output must be float32 (same as input)."""
        adapter.reset()
        key = torch.randn(B, P, D_KEY)
        correction = adapter.adapt(key)
        assert correction.dtype == torch.float32

    def test_residual_not_all_zero(self, adapter):
        """The correction should contain non-zero values (adapter contributes signal)."""
        adapter.reset()
        key = torch.randn(B, P, D_KEY)
        correction = adapter.adapt(key)
        # alpha starts near 0.01 so the correction is small but non-zero
        assert correction.abs().max().item() >= 0.0  # should be non-nan at minimum
        # Not all identical to zero input
        assert not (correction == 0).all(), "Correction is all zeros — adapter is silent"

    def test_state_persists_across_calls(self, adapter):
        """Two consecutive adapt() calls with same input must differ due to state."""
        adapter.reset()
        key = torch.randn(B, P, D_KEY)
        c1 = adapter.adapt(key.clone())
        c2 = adapter.adapt(key.clone())
        # State changes between frames so corrections should be different
        assert not torch.allclose(c1, c2), (
            "Corrections are identical across calls — hidden state does not update"
        )

    def test_reset_clears_state(self, adapter):
        """After reset(), the first correction is reproducible."""
        key = torch.randn(B, P, D_KEY)

        adapter.reset()
        c_before = adapter.adapt(key.clone())

        adapter.reset()
        c_after = adapter.adapt(key.clone())

        assert torch.allclose(c_before, c_after, atol=1e-6), (
            "reset() did not reproduce the same correction from the same input"
        )


# ---------------------------------------------------------------------------
# MemoryBank + adapter integration tests
# ---------------------------------------------------------------------------

class TestMemoryBankWithAdapter:

    def _make_batch(self):
        """Minimal (B, P, D) features and (B, H, W) integer mask."""
        features = torch.randn(B, P, D_KEY)
        mask = torch.zeros(B, 4, 4, dtype=torch.long)  # all background
        mask[0, 0, 0] = 1  # one foreground patch per sample
        return features, mask

    def test_adapter_modifies_frame_keys(
        self,
        memory_bank_with_adapter,
    ):
        """Keys stored after add_frame with adapter must differ from keys without adapter.

        Strategy: use the SAME bank but compare the key from add_frame with and
        without monkey-patching the adapter off, so projection weights are identical.
        """
        features, mask = self._make_batch()
        bank = memory_bank_with_adapter

        # ── Run WITH adapter ──────────────────────────────────────────────
        bank.reset()
        bank.encode_reference(features.clone(), mask.clone())
        bank.add_frame(features.clone(), mask.clone())
        K_with, _ = bank.get_memory()
        K_query_with = K_with[:, P:, :].clone()

        # ── Run WITHOUT adapter (temporarily disabled) ────────────────────
        orig_adapter = bank.key_adapter
        bank.key_adapter = None
        bank.reset()
        bank.encode_reference(features.clone(), mask.clone())
        bank.add_frame(features.clone(), mask.clone())
        K_without, _ = bank.get_memory()
        K_query_without = K_without[:, P:, :].clone()
        bank.key_adapter = orig_adapter  # restore

        assert K_with.shape == K_without.shape, "Shape mismatch with/without adapter"

        # Reference slot must be identical (same projection weights, adapter never touches it)
        assert torch.allclose(K_with[:, :P, :], K_without[:, :P, :], atol=1e-5), (
            "Reference frame keys were modified by adapter (should not be)"
        )

        # Query slot must differ (adapter added a residual)
        assert not torch.allclose(K_query_with, K_query_without, atol=1e-5), (
            "Query frame keys are identical — adapter had no effect"
        )

    def test_reference_key_unchanged(self, memory_bank_with_adapter):
        """The reference frame's key must not be modified by the adapter."""
        features, mask = self._make_batch()
        bank = memory_bank_with_adapter

        # Get keys without any add_frame call
        bank.reset()
        bank.encode_reference(features.clone(), mask.clone())
        K_ref_only, _ = bank.get_memory()

        # Add a frame, then compare the reference slot
        bank.reset()
        bank.encode_reference(features.clone(), mask.clone())
        bank.add_frame(features.clone(), mask.clone())
        K_with_frame, _ = bank.get_memory()

        # The reference slot (first P tokens) must be identical
        assert torch.allclose(
            K_ref_only[:, :P, :], K_with_frame[:, :P, :], atol=1e-5
        ), "Reference frame key changed after add_frame() — adapter corrupted the anchor"

    def test_reset_also_resets_adapter(self, memory_bank_with_adapter):
        """bank.reset() must reset the adapter state so corrections are reproducible."""
        features, mask = self._make_batch()
        bank = memory_bank_with_adapter

        bank.reset()
        bank.encode_reference(features.clone(), mask.clone())
        bank.add_frame(features.clone(), mask.clone())
        K1, _ = bank.get_memory()
        K1_query = K1[:, P:, :].clone()

        bank.reset()
        bank.encode_reference(features.clone(), mask.clone())
        bank.add_frame(features.clone(), mask.clone())
        K2, _ = bank.get_memory()
        K2_query = K2[:, P:, :].clone()

        assert torch.allclose(K1_query, K2_query, atol=1e-5), (
            "Keys differ after bank.reset() — adapter state was not cleared"
        )


# ---------------------------------------------------------------------------
# VideoMambaSystem integration test
# ---------------------------------------------------------------------------

class TestVideoMambaSystemWithAdapter:
    """Integration-level tests for the VOS training step with KAN key adapter.

    Uses real default dimensions (dim_in=768 for DINOv2) so the Linear
    projection weights match the actual encoder output size.
    """

    @pytest.fixture(scope="class")
    def model_with_adapter(self):
        from models.video_mamba import VideoMambaSystem
        return VideoMambaSystem(
            num_clf_classes=5,
            num_seg_classes=5,
            target_size=16,
            use_kan_key_adapter=True,
            kan_adapter_d_state=D_STATE,
        )

    @pytest.fixture(scope="class")
    def model_no_adapter(self):
        from models.video_mamba import VideoMambaSystem
        return VideoMambaSystem(
            num_clf_classes=5,
            num_seg_classes=5,
            target_size=16,
            use_kan_key_adapter=False,
        )

    def _make_vos_batch(self):
        B, T, H, W = 1, 3, 16, 16
        n_id = 4
        n_cls = 5
        return (
            torch.randn(B, 3, H, W),                    # ref_img
            torch.randint(0, n_cls, (B, H, W)),         # ref_mask
            torch.randn(B, T, 3, H, W),                 # query_images
            torch.randint(0, n_cls, (B, T, H, W)),      # query_masks
            torch.ones(B, T, n_id, dtype=torch.bool),   # obj_present
            [{"video_id": "test", "seen_obj_ids": [], "unseen_obj_ids": []}],
        )

    def test_adapter_enabled_finite_loss(self, model_with_adapter):
        model_with_adapter.eval()
        batch = self._make_vos_batch()
        with torch.no_grad():
            loss = model_with_adapter._shared_step(batch, 0, prefix="val")
        assert torch.isfinite(loss), f"Loss not finite with adapter: {loss}"

    def test_adapter_disabled_finite_loss(self, model_no_adapter):
        model_no_adapter.eval()
        batch = self._make_vos_batch()
        with torch.no_grad():
            loss = model_no_adapter._shared_step(batch, 0, prefix="val")
        assert torch.isfinite(loss), f"Loss not finite without adapter: {loss}"
