"""Per-sample gradient norm tracker for Lightning training loops.

Registers temporary autograd hooks on leaf parameters before each backward pass,
captures per-parameter gradient norms aligned with sample indices, then removes
all hooks after backward. Saves per-epoch .npz archives and logs to Comet.
"""

from __future__ import annotations

import logging
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torch.utils.hooks import RemovableHandle

log = logging.getLogger(__name__)


class PerSampleGradientTracker(Callback):
    """Tracks per-sample gradient norms via temporary autograd hooks.

    Registers one hook per leaf parameter in ``on_before_backward`` and removes
    all hooks in ``on_after_backward``. After each step, aligns captured norms
    with the batch sample identifiers. Saves a ``.npz`` archive per epoch and
    logs scalar summaries to the Comet experiment logger.

    Args:
        output_dir: Directory for ``.npz`` archives. Created if absent.
        log_every_n_steps: Frequency (in optimizer steps) at which to record
            gradient norms. Defaults to 1 (every step).
        sample_index_key: Key in the batch dict containing sample indices.
            If the batch is a tuple (no dict), falls back to sequential integers.
        track_layer_names: Optional list of parameter name substrings to track.
            If ``None``, all leaf parameters with gradients are tracked.
        comet_prefix: Prefix for Comet metric keys.
    """

    def __init__(
        self,
        output_dir: str = "artifacts/gradient_tracking",
        log_every_n_steps: int = 1,
        sample_index_key: str = "sample_idx",
        track_layer_names: list[str] | None = None,
        comet_prefix: str = "grad_tracking",
    ) -> None:
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.log_every_n_steps = log_every_n_steps
        self.sample_index_key = sample_index_key
        self.track_layer_names = track_layer_names
        self.comet_prefix = comet_prefix

        # Per-step state — reset each backward
        self._hooks: list[RemovableHandle] = []
        self._grad_norms_this_step: dict[str, float] = {}

        # Epoch accumulators
        self._steps: list[int] = []
        self._total_norms: list[float] = []
        self._sample_indices: list[Any] = []
        self._layer_norms_epoch: list[dict[str, float]] = []
        self._layer_names_seen: list[str] = []

        self._current_batch: Any = None
        self._global_step: int = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset epoch accumulators and clear any stale hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._steps = []
        self._total_norms = []
        self._sample_indices = []
        self._layer_norms_epoch = []

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._current_batch = batch
        self._global_step = trainer.global_step

    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: Float[Tensor, ""]
    ) -> None:
        """Register fresh autograd hooks on all tracked leaf parameters."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        self._grad_norms_this_step = {}

        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            if self.track_layer_names and not any(s in name for s in self.track_layer_names):
                continue

            def _make_hook(layer_name: str):
                def _hook(grad: Float[Tensor, "..."]) -> None:
                    self._grad_norms_this_step[layer_name] = grad.detach().norm().item()

                return _hook

            handle = param.register_hook(_make_hook(name))
            self._hooks.append(handle)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Remove hooks and record gradient norms for this step."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not self._grad_norms_this_step:
            return

        # Resolve sample indices
        sample_idx = self._resolve_sample_indices(self._current_batch)

        norms = list(self._grad_norms_this_step.values())
        total_norm = float(np.linalg.norm(norms)) if norms else 0.0

        self._steps.append(self._global_step)
        self._total_norms.append(total_norm)
        self._sample_indices.append(sample_idx)
        self._layer_norms_epoch.append(dict(self._grad_norms_this_step))

        # Update seen layer names (union across steps)
        for key in self._grad_norms_this_step:
            if key not in self._layer_names_seen:
                self._layer_names_seen.append(key)

        # Log to Comet
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None:
            experiment.log_metric(
                f"{self.comet_prefix}/total_norm",
                total_norm,
                step=self._global_step,
            )
            for layer_name, norm in self._grad_norms_this_step.items():
                safe_name = layer_name.replace(".", "/")
                experiment.log_metric(
                    f"{self.comet_prefix}/layer/{safe_name}",
                    norm,
                    step=self._global_step,
                )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save epoch archive to disk."""
        if not self._steps:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch
        path = self.output_dir / f"gradient_tracking_epoch_{epoch:04d}.npz"

        np.savez(
            path,
            steps=np.array(self._steps, dtype=np.int64),
            total_grad_norms=np.array(self._total_norms, dtype=np.float32),
            sample_indices=np.array(self._sample_indices, dtype=object),
            layer_norms=np.array(self._layer_norms_epoch, dtype=object),
            layer_names=np.array(self._layer_names_seen),
        )
        log.info("PerSampleGradientTracker: saved %s", path)

    def _resolve_sample_indices(self, batch: Any) -> Any:
        """Extract sample indices from the batch, falling back to a sentinel.

        Args:
            batch: The raw batch from the dataloader (dict or tuple).

        Returns:
            Sample index scalar, list, or a sequential-integer sentinel string.
        """
        if isinstance(batch, dict) and self.sample_index_key in batch:
            return batch[self.sample_index_key]
        # VOS tuple: (ref_img, ref_mask, query_imgs, query_masks, seq_names)
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            last = batch[-1]
            if isinstance(last, (list, str)):
                return last  # seq_names
        log.warning(
            "PerSampleGradientTracker: '%s' not found in batch; using step index.",
            self.sample_index_key,
        )
        return self._global_step
