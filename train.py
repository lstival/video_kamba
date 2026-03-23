from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Node

# PyTorch 2.6 changed weights_only default to True, breaking checkpoints saved with older
# versions that contain builtins (dict, typing.Any, etc.) via Hydra/OmegaConf pickling.
# Our checkpoints are internally produced and trusted, so we revert the default to False.
_torch_load_orig = torch.load
torch.load = lambda *args, **kwargs: _torch_load_orig(  # type: ignore[assignment]
    *args, **{**kwargs, "weights_only": False}
)

LOGGER = logging.getLogger(__name__)


def _safe_command(args: list[str]) -> str:
    """Return command output or 'unknown' when command execution fails."""
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def _log_run_metadata(cfg: DictConfig) -> None:
    """Log reproducibility metadata at run start."""
    git_commit = _safe_command(["git", "rev-parse", "HEAD"])
    git_branch = _safe_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    try:
        git_dirty = bool(_safe_command(["git", "status", "--porcelain"]))
    except Exception:
        git_dirty = False

    LOGGER.info(
        "Run metadata | git_commit=%s git_branch=%s git_dirty=%s",
        git_commit,
        git_branch,
        git_dirty,
    )
    LOGGER.info(
        "Environment | python=%s torch=%s lightning=%s cuda_available=%s cuda_version=%s",
        platform.python_version(),
        torch.__version__,
        L.__version__,
        torch.cuda.is_available(),
        torch.version.cuda,
    )
    LOGGER.info(
        "System | platform=%s hostname=%s",
        platform.platform(),
        platform.node(),
    )
    LOGGER.info(
        "Config | seed=%s trainer.deterministic=%s",
        cfg.get("seed", None),
        cfg.trainer.get("deterministic", None),
    )


def _instantiate_callbacks(cfg: DictConfig | None) -> list[Any]:
    """Instantiate callbacks from Hydra config.

    Supports a single callback config (`_target_` at root) or a mapping/list
    of callback configs.
    """
    if cfg is None:
        return []

    callbacks = []

    if isinstance(cfg, DictConfig) and "_target_" in cfg:
        callbacks.append(hydra.utils.instantiate(cfg))
        return callbacks

    if isinstance(cfg, DictConfig):
        for cb_cfg in cfg.values():
            if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))
        return callbacks

    if isinstance(cfg, (list, tuple)):
        for cb_cfg in cfg:
            if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))

    return callbacks

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set matmul precision for A100 Tensor Core optimization
    torch.set_float32_matmul_precision("medium")

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    _log_run_metadata(cfg)

    # Initialize logger using plain Python containers to avoid ListConfig
    # leaking into third-party APIs (e.g. Comet add_tags expects list).
    logger_cfg: Any = cfg.get("logger") if "logger" in cfg else None
    if isinstance(logger_cfg, DictConfig):
        logger_cfg = OmegaConf.to_container(logger_cfg, resolve=True)

    logger = hydra.utils.instantiate(logger_cfg) if logger_cfg is not None else None
    if logger is not None and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Initialize callbacks from config (checkpointing, LR monitor, etc.)
    callbacks = _instantiate_callbacks(cfg.get("callbacks"))
    
    # Initialize trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )
    
    # Initialize datamodule (Placeholder for now)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # Initialize model
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Partial weight transfer from a pretrained checkpoint (shape-safe).
    # Use `+pretrained_weights=path` when the architecture has changed since
    # the checkpoint was saved. Only parameters whose names AND shapes match
    # are transferred; everything else is left at its random initialisation.
    # This is distinct from `+checkpoint` (full resume with optimizer state).
    pretrained_weights_path = cfg.get("pretrained_weights")
    if pretrained_weights_path:
        pretrained_weights_path = os.path.expanduser(str(pretrained_weights_path))
        if not os.path.exists(pretrained_weights_path):
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights_path}")
        ckpt = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
        ckpt_state = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        transferable = {
            k: v for k, v in ckpt_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = [k for k in ckpt_state if k not in transferable]
        model.load_state_dict({**model_state, **transferable}, strict=True)
        LOGGER.info(
            "Partial weight transfer: %d/%d params loaded from %s — skipped %d "
            "(missing or shape mismatch): %s",
            len(transferable), len(ckpt_state),
            pretrained_weights_path, len(skipped),
            skipped[:10],
        )

    # Resume full training state from checkpoint when provided.
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path:
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        LOGGER.info(
            "Resuming training from checkpoint with optimizer state: %s",
            checkpoint_path,
        )
    
    # Optional: Automatically find maximum batch size
    if cfg.get("auto_batch_size", False):
        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(trainer)
        # This automatically modifies model.hparams.batch_size or datamodule.batch_size
        tuner.scale_batch_size(model, datamodule=datamodule, mode="binsearch")
        LOGGER.info("Auto batch size tuning completed.")
        
    # Train the model
    # If checkpoint is provided, we can either use it to resume training 
    # (including optimizer state) or just as weight initialization.
    # For fine-tuning on a different dataset, we usually want weight init only.
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    
    LOGGER.info("Training finished successfully.")

if __name__ == "__main__":
    main()
