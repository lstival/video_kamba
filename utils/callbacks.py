from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
from collections import deque
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


class SpikeDiagnosticsCallback(Callback):
    """Log which sequences are in the batch whenever training loss spikes.

    Maintains a rolling window of recent step losses and flags any batch whose
    loss exceeds ``mean + spike_z_threshold * std``.  Flagged batches are
    appended as JSON lines to ``log_path`` so they can be inspected offline.

    This helps identify problematic sequences (e.g. crowded scenes, rare object
    categories, extreme aspect ratios) that repeatedly cause gradient spikes.

    Args:
        spike_z_threshold: Z-score threshold above which a step is a spike.
        rolling_window:     Number of recent steps used to compute mean/std.
        warmup_steps:       Steps to skip before spike detection activates
                            (avoids false positives during early loss descent).
        log_path:           JSONL file to append spike records.
    """

    def __init__(
        self,
        spike_z_threshold: float = 2.5,
        rolling_window: int = 500,
        warmup_steps: int = 50,
        log_path: str = "logs/spike_diagnostics.jsonl",
    ) -> None:
        super().__init__()
        self._z_threshold = spike_z_threshold
        self._warmup_steps = warmup_steps
        self._log_path_str = log_path
        self._log_path: Path | None = None
        self._losses: deque[float] = deque(maxlen=rolling_window)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_path = Path(self._log_path_str)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("SpikeDiagnosticsCallback writing to %s", self._log_path)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss_val = self._extract_loss(outputs)
        if loss_val is None:
            return

        self._losses.append(loss_val)

        if len(self._losses) < self._warmup_steps:
            return

        mu = statistics.mean(self._losses)
        sigma = statistics.stdev(self._losses)
        if sigma < 1e-8:
            return

        z = (loss_val - mu) / sigma
        if z < self._z_threshold:
            return

        seqs = self._extract_seqs(batch)
        record = {
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "batch_idx": batch_idx,
            "loss": round(loss_val, 6),
            "rolling_mean": round(mu, 6),
            "rolling_std": round(sigma, 6),
            "z_score": round(z, 3),
            "sequences": seqs,
        }
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

        LOGGER.warning(
            "Loss spike z=%.2f step=%d loss=%.4f seqs=%s",
            z,
            trainer.global_step,
            loss_val,
            seqs,
        )

    @staticmethod
    def _extract_loss(outputs: Any) -> float | None:
        """Return scalar loss from training_step output."""
        if isinstance(outputs, torch.Tensor):
            return outputs.item()
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"].item()
        try:
            return float(outputs)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_seqs(batch: Any) -> list[str]:
        """Extract sequence identifiers from any supported batch format.

        Supported formats:
        - 5-tuple (DAVIS / YTB via _vos_collate): last element is list[str]
        - 6-tuple (COCO / multi-obj VOS): last element is list[dict] with
          "video_id" or "seq" key
        - DAVISDataset 5-tuple: last element is a single str (batched to list
          by default collate)
        """
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return ["unknown"]

        last = batch[-1]

        # _vos_collate / default collate on DAVISDataset: list of str seq names
        if isinstance(last, (list, tuple)) and last and isinstance(last[0], str):
            return list(last)

        # _coco_collate_fn / _vos_step meta: list of dicts
        if isinstance(last, (list, tuple)) and last and isinstance(last[0], dict):
            return [d.get("video_id", d.get("seq", "unknown")) for d in last]

        return ["unknown"]
