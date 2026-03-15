from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

LOGGER = logging.getLogger(__name__)


def _safe_command(args: list[str]) -> str:
    """Return command output or 'unknown' when command execution fails."""
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def _collect_metadata() -> dict[str, Any]:
    """Collect run metadata used for reproducibility auditing."""
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "lightning_version": L.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": _safe_command(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_safe_command(["git", "status", "--porcelain"])),
    }


class SaveHydraConfigCallback(Callback):
    """Save Hydra config and run metadata into the checkpoint directory."""

    def __init__(
        self,
        config_filename: str = "config.yaml",
        metadata_filename: str = "run_metadata.json",
        save_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.config_filename = config_filename
        self.metadata_filename = metadata_filename
        self.save_dir = save_dir

    def _resolve_save_dir(self, trainer: L.Trainer) -> Path:
        if self.save_dir is not None:
            return Path(self.save_dir)

        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.dirpath:
                return Path(callback.dirpath)

        return Path(trainer.default_root_dir) / "checkpoints"

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        target_dir = self._resolve_save_dir(trainer)
        target_dir.mkdir(parents=True, exist_ok=True)

        source_config_path: Path | None = None
        try:
            hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
            candidate = hydra_output_dir / ".hydra" / "config.yaml"
            if candidate.exists():
                source_config_path = candidate
        except Exception:
            source_config_path = None

        target_config_path = target_dir / self.config_filename
        if source_config_path is not None:
            shutil.copy2(source_config_path, target_config_path)
            LOGGER.info("Saved Hydra config to %s", target_config_path)
        else:
            LOGGER.warning("Could not find Hydra config.yaml; skipped config copy.")

        metadata = _collect_metadata()
        metadata_path = target_dir / self.metadata_filename
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("Saved run metadata to %s", metadata_path)
