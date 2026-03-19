"""TRAK influence function approximation via random gradient projections.

Implements the core TRAK formula (Park et al., ICML 2023,
https://arxiv.org/abs/2303.14186):

    TRAK(z, z') ≈ Φ(z)ᵀ Φ(z')
    where Φ(x) = J(x) @ P
          J(x) = flattened gradient vector for sample x
          P    = random projection matrix ∈ ℝ^{D × proj_dim}

The projection matrix is initialised once (lazy, seed-fixed, normalised) and
reused across all steps, reducing memory by projecting from parameter space D
down to ``projection_dim`` (default 4096).

Modes:
- ``online``: projects every step (default — more accurate, higher cost).
- ``checkpoint``: projects only at steps listed in ``checkpoint_steps``.

Requires ``pl_module.last_per_sample_losses`` (same as
``PerSampleLossTrajectoryTracker``).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

log = logging.getLogger(__name__)


class TRAKInfluenceCallback(Callback):
    """Approximates training-sample influence on validation via TRAK.

    See: Park et al., "TRAK: Attributing Model Behaviour at Scale",
    ICML 2023, https://arxiv.org/abs/2303.14186 (Eq. 3).

    Args:
        output_dir: Directory for ``.npz`` archives.
        projection_dim: Dimensionality of the random projection.
            Use 2048 for fast runs, 4096 for standard, 8192 for
            publication-quality influence scores.
        mode: ``"online"`` projects every step; ``"checkpoint"`` projects
            only at steps in ``checkpoint_steps``.
        checkpoint_steps: Steps at which to project (only used when
            ``mode="checkpoint"``).
        num_val_samples: Maximum number of validation batches to collect
            gradients from. Set to 0 to disable validation influence.
        seed: Random seed for the projection matrix.
        comet_prefix: Prefix for Comet metric keys.
    """

    def __init__(
        self,
        output_dir: str = "artifacts/trak",
        projection_dim: int = 4096,
        mode: str = "online",
        checkpoint_steps: list[int] | None = None,
        num_val_samples: int = 50,
        seed: int = 42,
        comet_prefix: str = "trak",
    ) -> None:
        super().__init__()
        assert mode in {"online", "checkpoint"}, f"mode must be 'online' or 'checkpoint', got '{mode}'"
        self.output_dir = pathlib.Path(output_dir)
        self.projection_dim = projection_dim
        self.mode = mode
        self.checkpoint_steps = set(checkpoint_steps or [])
        self.num_val_samples = num_val_samples
        self.seed = seed
        self.comet_prefix = comet_prefix

        # Projection matrix — lazily initialised on first use
        self._projection_matrix: Float[Tensor, "param_dim proj_dim"] | None = None
        self._param_dim: int | None = None

        # Epoch accumulators
        self._train_projections: list[Float[Tensor, "proj_dim"]] = []
        self._train_indices: list[Any] = []
        self._val_projections: list[Float[Tensor, "proj_dim"]] = []

        self._current_batch: Any = None
        self._val_batches_collected: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._train_projections = []
        self._train_indices = []
        self._val_projections = []
        self._val_batches_collected = 0

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._current_batch = batch

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and store gradient projection for this training step."""
        if self.mode == "checkpoint" and trainer.global_step not in self.checkpoint_steps:
            return
        if self.num_val_samples == 0 and not self._train_projections:
            # Lazy: skip if val influence is disabled and no val projections to compare
            pass

        projection = self._project_gradients(pl_module)
        if projection is None:
            return

        self._train_projections.append(projection.cpu())
        sample_idx = self._resolve_sample_indices(self._current_batch)
        self._train_indices.append(sample_idx)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect validation gradient projections."""
        if self.num_val_samples == 0:
            return
        if self._val_batches_collected >= self.num_val_samples:
            return

        per_sample_losses: torch.Tensor | None = getattr(
            pl_module, "last_per_sample_losses", None
        )
        if per_sample_losses is None:
            return

        # Compute gradient of mean val loss w.r.t. parameters
        pl_module.zero_grad()
        val_loss = per_sample_losses.float().mean()
        val_loss.backward(retain_graph=False)

        projection = self._project_gradients(pl_module)
        pl_module.zero_grad()

        if projection is not None:
            self._val_projections.append(projection.cpu())
            self._val_batches_collected += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute influence scores, save archive, and log to Comet."""
        if not self._train_projections:
            return

        epoch = trainer.current_epoch
        train_proj = torch.stack(self._train_projections)  # [N_train, proj_dim]

        influence_scores: Float[Tensor, "n_train n_val"] | None = None
        if self._val_projections:
            val_proj = torch.stack(self._val_projections)          # [N_val, proj_dim]
            influence_scores = train_proj @ val_proj.T              # [N_train, N_val]

        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None and influence_scores is not None:
            mean_abs = float(influence_scores.abs().mean().item())
            experiment.log_metric(
                f"{self.comet_prefix}/mean_abs_influence_score", mean_abs, step=epoch
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"trak_epoch_{epoch:04d}.npz"
        save_dict: dict[str, Any] = {
            "train_projections": train_proj.numpy().astype(np.float32),
            "train_indices": np.array(self._train_indices, dtype=object),
        }
        if influence_scores is not None:
            save_dict["influence_scores"] = influence_scores.numpy().astype(np.float32)
        np.savez(path, **save_dict)
        log.info("TRAKInfluenceCallback: saved %s (train=%d, val=%d)", path, len(self._train_projections), len(self._val_projections))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_projection_matrix(
        self, param_dim: int, device: torch.device
    ) -> Float[Tensor, "param_dim proj_dim"]:
        """Return (lazily initialised) seed-fixed normalised projection matrix.

        Args:
            param_dim: Total number of trainable parameters (flattened).
            device: Target device for the matrix.

        Returns:
            Projection matrix of shape ``[param_dim, projection_dim]``.
        """
        if self._projection_matrix is None or self._param_dim != param_dim:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            matrix = torch.randn(
                param_dim, self.projection_dim, generator=generator
            ) / (self.projection_dim ** 0.5)
            self._projection_matrix = matrix
            self._param_dim = param_dim
            log.info(
                "TRAKInfluenceCallback: initialised projection matrix [%d × %d].",
                param_dim,
                self.projection_dim,
            )
        return self._projection_matrix.to(device)

    def _project_gradients(
        self, pl_module: LightningModule
    ) -> Float[Tensor, "proj_dim"] | None:
        """Flatten all parameter gradients and project to ``projection_dim``.

        Args:
            pl_module: The LightningModule whose ``.grad`` attributes are read.

        Returns:
            Projected gradient vector of shape ``[projection_dim]``, or ``None``
            if no gradients are available.
        """
        grads = []
        for param in pl_module.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().reshape(-1))

        if not grads:
            return None

        grad_vector: Float[Tensor, "param_dim"] = torch.cat(grads)  # [D]
        proj_matrix = self._get_projection_matrix(grad_vector.shape[0], grad_vector.device)
        projected: Float[Tensor, "proj_dim"] = grad_vector @ proj_matrix  # [proj_dim]
        return projected

    def _resolve_sample_indices(self, batch: Any) -> Any:
        # VOS tuple: (ref_img, ref_mask, query_imgs, query_masks, seq_names)
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            last = batch[-1]
            if isinstance(last, list):
                return last
        if isinstance(batch, dict) and "sample_idx" in batch:
            return batch["sample_idx"]
        return None
