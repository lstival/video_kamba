"""Layer-wise gradient norm tracker with pathology detection.

Reads ``.grad`` attributes after each backward pass — no autograd hooks.
Detects vanishing gradients (persistent streaks), exploding gradients, and
dead layers (zero norm for an entire epoch). Saves per-epoch ``.npz`` archives
and logs per-layer statistics to Comet.
"""

from __future__ import annotations

import logging
import pathlib
from collections import defaultdict

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)

# Pathology thresholds
_VANISHING_THRESHOLD: float = 1e-7
_EXPLODING_THRESHOLD: float = 1e3


class LayerGradientNormTracker(Callback):
    """Monitors per-layer gradient norms and flags training pathologies.

    Reads ``.grad`` directly after backward (no hooks), making it safe to
    combine with ``PerSampleGradientTracker`` without doubling hook overhead.

    Pathologies detected:
    - **Vanishing**: layer norm < ``vanishing_threshold`` for ``patience``
      consecutive steps → ``WARNING``.
    - **Exploding**: layer norm > ``exploding_threshold`` → ``WARNING``.
    - **Dead layer**: norm == 0 for the entire epoch → ``ERROR``.

    Args:
        output_dir: Directory for ``.npz`` archives.
        log_every_n_steps: Recording frequency in optimizer steps.
        vanishing_threshold: Norm below which a gradient is considered vanishing.
        exploding_threshold: Norm above which a gradient is considered exploding.
        patience: Consecutive vanishing steps before emitting a warning.
        track_layer_names: Optional list of parameter name substrings to restrict
            tracking. ``None`` tracks all parameters with gradients.
        comet_prefix: Prefix for Comet metric keys.
    """

    def __init__(
        self,
        output_dir: str = "artifacts/gradient_norms",
        log_every_n_steps: int = 1,
        vanishing_threshold: float = _VANISHING_THRESHOLD,
        exploding_threshold: float = _EXPLODING_THRESHOLD,
        patience: int = 10,
        track_layer_names: list[str] | None = None,
        comet_prefix: str = "layer_grad_norm",
    ) -> None:
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.log_every_n_steps = log_every_n_steps
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.patience = patience
        self.track_layer_names = track_layer_names
        self.comet_prefix = comet_prefix

        # Per-layer streak counters — reset each epoch
        self._vanishing_streak: dict[str, int] = defaultdict(int)

        # Epoch accumulators: layer_name → list of norms
        self._epoch_norms: dict[str, list[float]] = defaultdict(list)
        self._vanishing_counts: dict[str, int] = defaultdict(int)
        self._exploding_counts: dict[str, int] = defaultdict(int)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._vanishing_streak = defaultdict(int)
        self._epoch_norms = defaultdict(list)
        self._vanishing_counts = defaultdict(int)
        self._exploding_counts = defaultdict(int)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Read .grad attributes and check for pathologies."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        experiment = getattr(trainer.logger, "experiment", None)

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            if self.track_layer_names and not any(s in name for s in self.track_layer_names):
                continue

            norm = param.grad.detach().norm().item()
            self._epoch_norms[name].append(norm)

            # Vanishing detection with streak tracking
            if norm < self.vanishing_threshold:
                self._vanishing_streak[name] += 1
                self._vanishing_counts[name] += 1
                if self._vanishing_streak[name] >= self.patience:
                    log.warning(
                        "Vanishing gradient detected in '%s': norm=%.2e for %d "
                        "consecutive steps (threshold=%.1e).",
                        name,
                        norm,
                        self._vanishing_streak[name],
                        self.vanishing_threshold,
                    )
            else:
                self._vanishing_streak[name] = 0

            # Exploding detection
            if norm > self.exploding_threshold:
                self._exploding_counts[name] += 1
                log.warning(
                    "Exploding gradient detected in '%s': norm=%.2e (threshold=%.1e).",
                    name,
                    norm,
                    self.exploding_threshold,
                )

            # Log per-step to Comet
            if experiment is not None:
                safe_name = name.replace(".", "/")
                experiment.log_metric(
                    f"{self.comet_prefix}/{safe_name}",
                    norm,
                    step=trainer.global_step,
                )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute epoch stats, flag dead layers, save archive, log to Comet."""
        if not self._epoch_norms:
            return

        layer_names = list(self._epoch_norms.keys())
        mean_norms, std_norms, p95_norms = [], [], []
        vanishing_counts, exploding_counts = [], []

        experiment = getattr(trainer.logger, "experiment", None)
        epoch = trainer.current_epoch

        for name in layer_names:
            arr = np.array(self._epoch_norms[name], dtype=np.float32)
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            p95_val = float(np.percentile(arr, 95))

            mean_norms.append(mean_val)
            std_norms.append(std_val)
            p95_norms.append(p95_val)
            vanishing_counts.append(self._vanishing_counts[name])
            exploding_counts.append(self._exploding_counts[name])

            # Dead layer: every recorded norm was exactly zero
            if arr.max() == 0.0:
                log.error(
                    "Dead layer detected: '%s' had zero gradient norm for the entire epoch %d.",
                    name,
                    epoch,
                )

            # Log epoch-level stats to Comet
            if experiment is not None:
                safe_name = name.replace(".", "/")
                prefix = f"{self.comet_prefix}_epoch/{safe_name}"
                experiment.log_metric(f"{prefix}/mean", mean_val, step=epoch)
                experiment.log_metric(f"{prefix}/std", std_val, step=epoch)
                experiment.log_metric(f"{prefix}/p95", p95_val, step=epoch)
                experiment.log_metric(
                    f"{prefix}/vanishing_count", self._vanishing_counts[name], step=epoch
                )
                experiment.log_metric(
                    f"{prefix}/exploding_count", self._exploding_counts[name], step=epoch
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"layer_grad_norms_epoch_{epoch:04d}.npz"
        np.savez(
            path,
            layer_names=np.array(layer_names),
            mean_norms=np.array(mean_norms, dtype=np.float32),
            std_norms=np.array(std_norms, dtype=np.float32),
            p95_norms=np.array(p95_norms, dtype=np.float32),
            vanishing_counts=np.array(vanishing_counts, dtype=np.int64),
            exploding_counts=np.array(exploding_counts, dtype=np.int64),
        )
        log.info("LayerGradientNormTracker: saved %s", path)
