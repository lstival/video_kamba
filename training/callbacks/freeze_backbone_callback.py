"""Callback to freeze backbone (feature_extractor) parameters at train start.

Prevents gradients flowing through the frozen encoder during Phase 4 fine-tuning,
where only the propagation components (MemoryBank, PropagationAttention, KAN-SSM)
and segmentation decoder should be updated.
"""

from __future__ import annotations

import logging

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class FreezeBackboneCallback(Callback):
    """Freezes named submodules at the start of training.

    Args:
        module_names: List of top-level attribute names on the LightningModule
            to freeze. Defaults to ``["feature_extractor"]``.
        unfreeze_epoch: If set, unfreezes the modules at this epoch (zero-indexed).
            Useful for gradual unfreezing schedules.
    """

    def __init__(
        self,
        module_names: list[str] | None = None,
        unfreeze_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.module_names = module_names or ["feature_extractor"]
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._freeze(pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.unfreeze_epoch is not None and trainer.current_epoch == self.unfreeze_epoch:
            self._unfreeze(pl_module)

    def _freeze(self, pl_module: LightningModule) -> None:
        for name in self.module_names:
            module = getattr(pl_module, name, None)
            if module is None:
                log.warning("FreezeBackboneCallback: module '%s' not found on model.", name)
                continue
            module.requires_grad_(False)
            n_params = sum(p.numel() for p in module.parameters())
            log.info(
                "FreezeBackboneCallback: froze '%s' (%d parameters).", name, n_params
            )

    def _unfreeze(self, pl_module: LightningModule) -> None:
        for name in self.module_names:
            module = getattr(pl_module, name, None)
            if module is None:
                continue
            module.requires_grad_(True)
            log.info("FreezeBackboneCallback: unfroze '%s' at epoch %d.", name, self.unfreeze_epoch)
