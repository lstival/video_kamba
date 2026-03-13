"""KAN-SSM Memory Key Adapter.

Wraps a lightweight KangaSSM to apply a learned, recurrent residual correction
to memory-bank keys **after** they have been projected but **before** they are
stored.  The adapter maintains a hidden state across frames so the correction
accumulates knowledge of how the object's appearance has changed since the
reference frame.

Design notes
------------
* Operates in ``d_key`` space (default 256-dim), not the full feature space
  (768-dim for DINOv2), so the cost is low (~0.1 % extra parameters).
* The reference-frame key is **never** adapted — it is the fixed anchor.
  Only keys added via ``MemoryBank.add_frame()`` receive the correction.
* The correction is an **additive residual** gated by a learned scalar
  ``alpha`` that starts near zero, allowing gradual ramp-up during training.
* Reset between videos: call ``reset()`` before each new sequence, consistent
  with ``MemoryBank.reset()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from .kanga_ssm import KangaSSM


class KANKeyAdapter(nn.Module):
    """Recurrent KAN-SSM adapter that refines memory-bank keys over time.

    The adapter is stateful: it accumulates a hidden SSM state across calls
    to ``adapt()``, so each frame's key correction depends on all previous
    frames.  Call ``reset()`` between videos.

    Args:
        d_key:      Dimensionality of the projected keys (must match
                    ``MemoryBank.d_key``).
        d_state:    SSM hidden-state dimension (smaller = cheaper).
        num_layers: Number of stacked ``IntricateKANSSMCore`` layers.
        modulator_type: ``'kan'`` (default) or ``'mlp'`` for ablation.
        alpha_init: Initial scale applied to the residual correction.
                    Starts small so the adapter does not disrupt early training.
    """

    def __init__(
        self,
        d_key: int = 256,
        d_state: int = 8,
        num_layers: int = 1,
        modulator_type: str = "kan",
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_key = d_key

        # Lightweight SSM operating in d_key space
        self._ssm = KangaSSM(
            d_model=d_key,
            d_state=d_state,
            num_layers=num_layers,
            dropout=0.0,          # no dropout — we only process 1 token per step
            modulator_type=modulator_type,
        )

        # Learnable gating scalar: starts near 0, grows during training
        self.alpha = nn.Parameter(torch.full((1,), alpha_init))

        # Persistent SSM hidden state (not a nn.Parameter — managed manually)
        self._state: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the recurrent state — must be called between videos."""
        self._state = None

    def adapt(
        self,
        key: Float[torch.Tensor, "B P d_key"],
    ) -> Float[torch.Tensor, "B P d_key"]:
        """Compute an additive residual correction for ``key``.

        The SSM processes the key as a single-step sequence ``[B*P, 1, d_key]``
        and returns a correction of the same shape.  The hidden state is
        preserved across calls so the correction is conditioned on the full
        key history seen so far.

        Args:
            key: Projected coarse key tensor ``[B, P, d_key]``.

        Returns:
            Residual correction ``[B, P, d_key]`` (add to the original key).
        """
        B, P, D = key.shape
        assert D == self.d_key, (
            f"KANKeyAdapter: expected d_key={self.d_key}, got {D}"
        )

        # Reshape to [B*P, 1, d_key] — process all patches as independent
        # 1-element sequences, sharing the SSM across patches for efficiency.
        x = key.reshape(B * P, 1, D)                  # [B*P, 1, d_key]

        correction_flat, next_state = self._ssm(
            x,
            prev_states=self._state,
            return_last_state=True,
        )                                               # [B*P, 1, d_key]

        # Persist state for the next call
        self._state = next_state

        correction = correction_flat.squeeze(1).reshape(B, P, D)  # [B, P, d_key]

        # Scale correction — alpha grows from near-zero during training
        return self.alpha * correction
