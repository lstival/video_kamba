"""Per-sample loss trajectory tracker with forgetting event detection.

Reads ``module.last_per_sample_losses`` (a detached tensor exposed by the
LightningModule's training step) to track per-sample loss across epochs.

Forgetting events (Toneva et al., ICLR 2019) are detected when a sample's loss
increases after having decreased. Hard examples are identified as samples whose
mean epoch loss exceeds the ``high_loss_percentile`` threshold.

Integration requirement — add to LightningModule._vos_step (or training_step):

    per_sample_loss = vos_loss_fn(logits_seg, query_masks, obj_present,
                                  reduction="none")
    self.last_per_sample_losses = per_sample_loss.detach()
    return per_sample_loss.mean()
"""

from __future__ import annotations

import logging
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class PerSampleLossTrajectoryTracker(Callback):
    """Tracks per-sample loss trajectories and detects forgetting events.

    Reads ``pl_module.last_per_sample_losses`` after each training step.
    Forgetting events (Toneva et al., ICLR 2019): a forgetting event occurs
    when a sample's loss increases after having previously decreased.

    Args:
        output_dir: Directory for ``.npz`` archives.
        log_every_n_steps: Recording frequency in optimizer steps.
        high_loss_percentile: Percentile above which a sample is considered
            hard. Defaults to 90 (p90).
        sample_index_key: Key in the batch dict for sample indices.
        comet_prefix: Prefix for Comet metric keys.
    """

    def __init__(
        self,
        output_dir: str = "artifacts/loss_trajectories",
        log_every_n_steps: int = 1,
        high_loss_percentile: float = 90.0,
        sample_index_key: str = "sample_idx",
        comet_prefix: str = "loss_trajectory",
    ) -> None:
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.log_every_n_steps = log_every_n_steps
        self.high_loss_percentile = high_loss_percentile
        self.sample_index_key = sample_index_key
        self.comet_prefix = comet_prefix

        # State for forgetting event detection (persists across steps)
        self._previous_losses: dict[Any, float] = {}
        self._forgetting_counts: dict[Any, int] = defaultdict(int)

        # Epoch accumulators
        self._steps: list[int] = []
        self._sample_indices_epoch: list[Any] = []
        self._per_sample_losses_epoch: list[list[float]] = []
        self._forgetting_events_this_epoch: list[Any] = []

        self._current_batch: Any = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._steps = []
        self._sample_indices_epoch = []
        self._per_sample_losses_epoch = []
        self._forgetting_events_this_epoch = []

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._current_batch = batch

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Read last_per_sample_losses and detect forgetting events."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        per_sample_losses: torch.Tensor | None = getattr(
            pl_module, "last_per_sample_losses", None
        )
        if per_sample_losses is None:
            return

        losses = per_sample_losses.float().cpu().tolist()
        if not isinstance(losses, list):
            losses = [losses]

        sample_indices = self._resolve_sample_indices(batch, len(losses))

        # Detect forgetting events
        for idx, loss_val in zip(sample_indices, losses):
            if idx in self._previous_losses:
                prev = self._previous_losses[idx]
                if loss_val > prev:
                    self._forgetting_counts[idx] += 1
                    self._forgetting_events_this_epoch.append(idx)
            self._previous_losses[idx] = loss_val

        self._steps.append(trainer.global_step)
        self._sample_indices_epoch.append(list(sample_indices))
        self._per_sample_losses_epoch.append(losses)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Identify hard examples, save archive, and log to Comet."""
        if not self._steps:
            return

        all_losses = [l for batch in self._per_sample_losses_epoch for l in batch]
        mean_epoch_loss = float(np.mean(all_losses)) if all_losses else 0.0
        threshold = float(np.percentile(all_losses, self.high_loss_percentile)) if all_losses else 0.0
        hard_example_count = int(np.sum(np.array(all_losses) > threshold))
        forgetting_events_total = len(self._forgetting_events_this_epoch)

        epoch = trainer.current_epoch
        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None:
            experiment.log_metric(f"{self.comet_prefix}/mean_epoch_loss", mean_epoch_loss, step=epoch)
            experiment.log_metric(f"{self.comet_prefix}/hard_example_count", hard_example_count, step=epoch)
            experiment.log_metric(
                f"{self.comet_prefix}/forgetting_events_total", forgetting_events_total, step=epoch
            )
            experiment.log_metric(
                f"{self.comet_prefix}/high_loss_threshold_p{int(self.high_loss_percentile)}",
                threshold,
                step=epoch,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"loss_trajectory_epoch_{epoch:04d}.npz"
        np.savez(
            path,
            steps=np.array(self._steps, dtype=np.int64),
            sample_indices=np.array(self._sample_indices_epoch, dtype=object),
            per_sample_losses=np.array(self._per_sample_losses_epoch, dtype=object),
            forgetting_indices=np.array(self._forgetting_events_this_epoch, dtype=object),
            forgetting_counts=np.array(
                list(self._forgetting_counts.values()), dtype=np.int64
            ),
        )
        log.info(
            "PerSampleLossTrajectoryTracker: epoch %d — mean_loss=%.4f, "
            "hard_examples=%d, forgetting_events=%d. Saved %s",
            epoch,
            mean_epoch_loss,
            hard_example_count,
            forgetting_events_total,
            path,
        )

    def _resolve_sample_indices(self, batch: Any, batch_size: int) -> list[Any]:
        """Extract sample identifiers from the batch.

        Args:
            batch: Raw dataloader batch (dict or tuple).
            batch_size: Number of samples in the batch (fallback size).

        Returns:
            List of per-sample identifiers.
        """
        if isinstance(batch, dict) and self.sample_index_key in batch:
            indices = batch[self.sample_index_key]
            return indices.tolist() if hasattr(indices, "tolist") else list(indices)
        # VOS tuple: (ref_img, ref_mask, query_imgs, query_masks, seq_names)
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            last = batch[-1]
            if isinstance(last, list) and last and isinstance(last[0], (str, int)):
                return last
            # last element is meta dicts — extract video_id as hashable key
            if isinstance(last, list) and last and isinstance(last[0], dict):
                return [m.get("video_id", i) for i, m in enumerate(last)]
        log.warning(
            "PerSampleLossTrajectoryTracker: sample indices not found; "
            "using sequential step-based fallback."
        )
        base = getattr(self, "_step_counter", 0)
        self._step_counter = base + batch_size  # type: ignore[attr-defined]
        return list(range(base, base + batch_size))
