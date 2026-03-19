"""Analyse a SLURM training run from log files and saved gradient/loss artifacts.

Produces a structured report covering:
  - Job metadata and run status (via sacct)
  - Run provenance (git commit, environment)
  - Training quality: loss per epoch, forgetting events (from .npz artifacts)
  - Gradient health: per-layer norms, explosion/vanishing counts (from .npz artifacts)
  - Gradient explosion diagnosis: worst offenders, module-level grouping
  - Plain-text report printed to stdout and optionally saved as Markdown

Usage (Hydra):
    python scripts/analyse_training_run.py job_id=65737509
    python scripts/analyse_training_run.py job_id=65737509 top_n_layers=15 save_report=true
    python scripts/analyse_training_run.py log_file=logs/slurm/vim_stability_fix_v2_65737509.out
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLURM_LOG_DIR = "logs/slurm"
ARTIFACTS_GRAD_NORMS = "artifacts/gradient_norms"
ARTIFACTS_LOSS_TRAJ = "artifacts/loss_trajectories"
EXPLODING_THRESHOLD = 1_000.0
VANISHING_THRESHOLD = 1e-7


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class JobInfo:
    """Metadata returned by sacct for a single SLURM job."""

    job_id: str
    job_name: str
    state: str
    exit_code: str
    elapsed: str
    start: str
    end: str


@dataclass
class RunProvenance:
    """Git and environment provenance parsed from the .out log."""

    git_commit: str = "unknown"
    git_branch: str = "unknown"
    git_dirty: bool = False
    python_version: str = "unknown"
    torch_version: str = "unknown"
    lightning_version: str = "unknown"
    hostname: str = "unknown"
    experiment_name: str = "unknown"
    job_id_from_log: str = "unknown"


@dataclass
class EpochLossRecord:
    """Per-epoch loss summary from the PerSampleLossTrajectoryTracker callback."""

    epoch: int
    mean_loss: float
    hard_examples: int
    forgetting_events: int


@dataclass
class EpochGradRecord:
    """Per-epoch gradient norm summary loaded from the layer_grad_norms .npz file."""

    epoch: int
    layer_names: list[str]
    mean_norms: np.ndarray
    std_norms: np.ndarray
    p95_norms: np.ndarray
    vanishing_counts: np.ndarray
    exploding_counts: np.ndarray

    def worst_layers(self, n: int = 10) -> list[tuple[str, float, float]]:
        """Return the n layers with highest mean gradient norm.

        Returns:
            List of (layer_name, mean_norm, p95_norm) sorted descending by mean.
        """
        indices = np.argsort(self.mean_norms)[::-1][:n]
        return [
            (self.layer_names[i], float(self.mean_norms[i]), float(self.p95_norms[i]))
            for i in indices
        ]

    @property
    def total_exploding(self) -> int:
        return int(self.exploding_counts.sum())

    @property
    def total_vanishing(self) -> int:
        return int(self.vanishing_counts.sum())


@dataclass
class GradExplosionEvent:
    """A single exploding-gradient warning line from the .out log."""

    timestamp: datetime
    layer_name: str
    norm: float


@dataclass
class TrainingReport:
    """Aggregated analysis of a single training run."""

    job_info: JobInfo | None
    provenance: RunProvenance
    epoch_losses: list[EpochLossRecord]
    epoch_grads: list[EpochGradRecord]
    explosion_events: list[GradExplosionEvent]
    final_val_loss: float | None
    final_train_loss: float | None
    max_epoch_seen: int


# ---------------------------------------------------------------------------
# SLURM job query
# ---------------------------------------------------------------------------


def query_sacct(job_id: str) -> JobInfo | None:
    """Query sacct for job accounting information.

    Args:
        job_id: SLURM job ID string.

    Returns:
        JobInfo if the job exists, None otherwise.
    """
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End",
        "-X",
        "--noheader",
        "--parsable2",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = line.split("|")
        if len(parts) < 7:
            return None
        return JobInfo(
            job_id=parts[0].strip(),
            job_name=parts[1].strip(),
            state=parts[2].strip(),
            exit_code=parts[3].strip(),
            elapsed=parts[4].strip(),
            start=parts[5].strip(),
            end=parts[6].strip(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Log file parser
# ---------------------------------------------------------------------------

_RE_TIMESTAMP_MODULE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\]\[([^\]]+)\]\[([^\]]+)\] - (.+)$"
)
_RE_METADATA = re.compile(
    r"Run metadata \| git_commit=(\S+) git_branch=(\S+) git_dirty=(\S+)"
)
_RE_ENVIRONMENT = re.compile(
    r"Environment \| python=(\S+) torch=(\S+) lightning=(\S+)"
)
_RE_SYSTEM = re.compile(r"System \|.*hostname=(\S+)")
_RE_EXPLODING = re.compile(
    r"Exploding gradient detected in '([^']+)': norm=([0-9.e+]+)"
)
_RE_LOSS_TRAJ = re.compile(
    r"PerSampleLossTrajectoryTracker: epoch (\d+) — mean_loss=([0-9.]+), "
    r"hard_examples=(\d+), forgetting_events=(\d+)"
)
_RE_JOB_ID_HEADER = re.compile(r"Job ID\s+:\s+(\S+)")
_RE_EXPERIMENT = re.compile(r"Experiment\s*:\s*(\S+)")
_RE_EPOCH_PROGRESS = re.compile(r"^Epoch (\d+)/(\d+)")
_RE_TQDM_METRIC = re.compile(r"(train_loss_epoch|val_loss_vos|val_loss|train_loss_step):\s*([0-9.]+)")


@dataclass
class _LogParseAccumulator:
    provenance: RunProvenance = field(default_factory=RunProvenance)
    epoch_losses: list[EpochLossRecord] = field(default_factory=list)
    explosion_events: list[GradExplosionEvent] = field(default_factory=list)
    final_val_loss: float | None = None
    final_train_loss: float | None = None
    max_epoch_seen: int = 0
    # Internal: buffer for multi-line tqdm metric block
    _tqdm_buffer: list[str] = field(default_factory=list)
    _in_tqdm_block: bool = False


def _flush_tqdm_buffer(acc: _LogParseAccumulator, buffer: list[str]) -> None:
    """Extract metric values from a collected tqdm epoch-progress block."""
    joined = " ".join(buffer)
    for m in _RE_TQDM_METRIC.finditer(joined):
        key, val = m.group(1), float(m.group(2))
        if key == "val_loss_vos" or key == "val_loss":
            acc.final_val_loss = val
        elif key == "train_loss_epoch":
            acc.final_train_loss = val


def parse_slurm_log(log_path: Path) -> _LogParseAccumulator:
    """Parse a SLURM .out log file into structured records.

    Args:
        log_path: Path to the SLURM stdout log file.

    Returns:
        Populated _LogParseAccumulator with all extracted information.

    Raises:
        FileNotFoundError: If log_path does not exist.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    acc = _LogParseAccumulator()
    tqdm_buffer: list[str] = []
    in_tqdm_block = False

    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()

            # ── Job header block ──────────────────────────────────────────
            if m := _RE_JOB_ID_HEADER.search(line):
                acc.provenance.job_id_from_log = m.group(1)
            if m := _RE_EXPERIMENT.search(line):
                acc.provenance.experiment_name = m.group(1)

            # ── Structured log lines ──────────────────────────────────────
            if m := _RE_TIMESTAMP_MODULE.match(line):
                ts_str, _module, _level, message = (
                    m.group(1),
                    m.group(2),
                    m.group(3),
                    m.group(4),
                )
                ts = _parse_timestamp(ts_str)

                if meta := _RE_METADATA.search(message):
                    acc.provenance.git_commit = meta.group(1)
                    acc.provenance.git_branch = meta.group(2)
                    acc.provenance.git_dirty = meta.group(3).lower() == "true"

                elif env := _RE_ENVIRONMENT.search(message):
                    acc.provenance.python_version = env.group(1)
                    acc.provenance.torch_version = env.group(2)
                    acc.provenance.lightning_version = env.group(3)

                elif sys := _RE_SYSTEM.search(message):
                    acc.provenance.hostname = sys.group(1)

                elif exp := _RE_EXPLODING.search(message):
                    acc.explosion_events.append(
                        GradExplosionEvent(
                            timestamp=ts,
                            layer_name=exp.group(1),
                            norm=float(exp.group(2)),
                        )
                    )

                elif lt := _RE_LOSS_TRAJ.search(message):
                    acc.epoch_losses.append(
                        EpochLossRecord(
                            epoch=int(lt.group(1)),
                            mean_loss=float(lt.group(2)),
                            hard_examples=int(lt.group(3)),
                            forgetting_events=int(lt.group(4)),
                        )
                    )

                in_tqdm_block = False
                if tqdm_buffer:
                    _flush_tqdm_buffer(acc, tqdm_buffer)
                    tqdm_buffer = []

            # ── tqdm epoch progress block (multi-line, non-structured) ────
            elif _RE_EPOCH_PROGRESS.match(line):
                if tqdm_buffer:
                    _flush_tqdm_buffer(acc, tqdm_buffer)
                tqdm_buffer = [line]
                in_tqdm_block = True
                em = _RE_EPOCH_PROGRESS.match(line)
                if em:
                    acc.max_epoch_seen = max(acc.max_epoch_seen, int(em.group(1)))

            elif in_tqdm_block and line.strip():
                tqdm_buffer.append(line)

            else:
                if tqdm_buffer and not line.strip():
                    in_tqdm_block = False

    if tqdm_buffer:
        _flush_tqdm_buffer(acc, tqdm_buffer)

    return acc


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse a log timestamp of the form '2026-03-18 18:16:13,256'."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return datetime.min


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------


def load_epoch_grad_norms(artifacts_dir: Path) -> list[EpochGradRecord]:
    """Load all layer_grad_norms_epoch_XXXX.npz files from artifacts_dir.

    Args:
        artifacts_dir: Path to artifacts/gradient_norms/.

    Returns:
        Sorted list of EpochGradRecord, one per epoch file found.
    """
    records: list[EpochGradRecord] = []
    pattern = "layer_grad_norms_epoch_*.npz"
    for npz_path in sorted(artifacts_dir.glob(pattern)):
        epoch = _epoch_from_filename(npz_path.stem)
        try:
            data = np.load(npz_path, allow_pickle=True)
            records.append(
                EpochGradRecord(
                    epoch=epoch,
                    layer_names=data["layer_names"].tolist(),
                    mean_norms=data["mean_norms"].astype(np.float32),
                    std_norms=data["std_norms"].astype(np.float32),
                    p95_norms=data["p95_norms"].astype(np.float32),
                    vanishing_counts=data["vanishing_counts"].astype(np.int64),
                    exploding_counts=data["exploding_counts"].astype(np.int64),
                )
            )
        except Exception as exc:
            log.warning("Could not load %s: %s", npz_path, exc)
    return records


def _epoch_from_filename(stem: str) -> int:
    """Extract epoch index from a filename like 'layer_grad_norms_epoch_0003'."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    job_id: str | None,
    log_path: Path,
    grad_norms_dir: Path,
) -> TrainingReport:
    """Build a complete TrainingReport from all available data sources.

    Args:
        job_id: SLURM job ID to query via sacct (optional).
        log_path: Path to the SLURM .out log file.
        grad_norms_dir: Directory containing layer_grad_norms_epoch_*.npz files.

    Returns:
        Fully populated TrainingReport.
    """
    job_info = query_sacct(job_id) if job_id else None

    log.info("Parsing log file: %s", log_path)
    acc = parse_slurm_log(log_path)

    log.info("Loading gradient norm artifacts from: %s", grad_norms_dir)
    epoch_grads = load_epoch_grad_norms(grad_norms_dir) if grad_norms_dir.exists() else []

    return TrainingReport(
        job_info=job_info,
        provenance=acc.provenance,
        epoch_losses=sorted(acc.epoch_losses, key=lambda r: r.epoch),
        epoch_grads=epoch_grads,
        explosion_events=acc.explosion_events,
        final_val_loss=acc.final_val_loss,
        final_train_loss=acc.final_train_loss,
        max_epoch_seen=acc.max_epoch_seen,
    )


# ---------------------------------------------------------------------------
# Gradient explosion analysis
# ---------------------------------------------------------------------------


def _explosion_counts_by_module(events: list[GradExplosionEvent]) -> dict[str, int]:
    """Count explosion events grouped by top-level module.

    Args:
        events: All GradExplosionEvent records from the log.

    Returns:
        Dict mapping module name → explosion count, sorted descending.
    """
    counts: dict[str, int] = {}
    for ev in events:
        module = ev.layer_name.split(".")[0]
        counts[module] = counts.get(module, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _norm_trend(epoch_grads: list[EpochGradRecord]) -> str:
    """Summarise whether overall gradient norms are growing, stable, or shrinking."""
    if len(epoch_grads) < 2:
        return "insufficient data"
    global_means = [float(r.mean_norms.mean()) for r in epoch_grads]
    first, last = global_means[0], global_means[-1]
    ratio = last / first if first > 0 else float("inf")
    if ratio > 2.0:
        return f"GROWING ({ratio:.1f}x from epoch {epoch_grads[0].epoch} to {epoch_grads[-1].epoch})"
    elif ratio < 0.5:
        return f"shrinking ({ratio:.2f}x)"
    return f"stable ({ratio:.2f}x)"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_SEP = "─" * 72
_THICK = "═" * 72


def _select_epoch_sample(
    epoch_grads: list[EpochGradRecord], max_cols: int = 12
) -> list[EpochGradRecord]:
    """Return a representative subset of at most max_cols epochs.

    Keeps the first 3, last 3, and evenly samples from the middle so that
    the progression table fits in a terminal without wrapping.

    Args:
        epoch_grads: Full list of EpochGradRecord sorted by epoch.
        max_cols: Maximum number of columns to include.

    Returns:
        Subset list preserving order.
    """
    n = len(epoch_grads)
    if n <= max_cols:
        return epoch_grads
    head = epoch_grads[:3]
    tail = epoch_grads[-3:]
    remaining_slots = max_cols - 6
    middle = epoch_grads[3:-3]
    if remaining_slots <= 0 or not middle:
        return head + tail
    step = max(1, len(middle) // remaining_slots)
    sampled_middle = middle[::step][:remaining_slots]
    seen: set[int] = set()
    result: list[EpochGradRecord] = []
    for rec in head + sampled_middle + tail:
        if rec.epoch not in seen:
            seen.add(rec.epoch)
            result.append(rec)
    return result


def render_report(report: TrainingReport, top_n: int = 10) -> str:
    """Render the full analysis report as a plain-text/Markdown string.

    Args:
        report: The aggregated TrainingReport.
        top_n: Number of worst gradient layers to show.

    Returns:
        Multi-line string ready for printing or saving.
    """
    lines: list[str] = []
    _h = lines.append

    _h(_THICK)
    _h("  TRAINING RUN ANALYSIS")
    _h(_THICK)

    # ── Job Summary ───────────────────────────────────────────────────────
    _h("\n## Job Summary\n")
    ji = report.job_info
    prov = report.provenance

    job_id_display = ji.job_id if ji else prov.job_id_from_log
    _h(f"  Job ID       : {job_id_display}")
    _h(f"  Job name     : {ji.job_name if ji else 'n/a'}")
    _h(f"  State        : {ji.state if ji else 'n/a'}")
    _h(f"  Exit code    : {ji.exit_code if ji else 'n/a'}")
    _h(f"  Duration     : {ji.elapsed if ji else 'n/a'}")
    _h(f"  Start        : {ji.start if ji else 'n/a'}")
    _h(f"  End          : {ji.end if ji else 'n/a'}")
    _h(f"  Epochs seen  : {report.max_epoch_seen} / (check config for max_epochs)")

    # ── Provenance ────────────────────────────────────────────────────────
    _h("\n## Run Provenance\n")
    _h(f"  Git commit   : {prov.git_commit}")
    _h(f"  Git branch   : {prov.git_branch}")
    _h(f"  Git dirty    : {prov.git_dirty}")
    _h(f"  Experiment   : {prov.experiment_name}")
    _h(f"  Python       : {prov.python_version}")
    _h(f"  PyTorch      : {prov.torch_version}")
    _h(f"  Lightning    : {prov.lightning_version}")
    _h(f"  Hostname     : {prov.hostname}")

    # ── Training Quality ──────────────────────────────────────────────────
    _h("\n## Training Quality\n")
    if report.epoch_losses:
        _h(f"  {'Epoch':>6}  {'Mean Loss':>10}  {'Hard Ex.':>10}  {'Forgetting':>12}")
        _h(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}")
        for rec in report.epoch_losses:
            _h(
                f"  {rec.epoch:>6}  {rec.mean_loss:>10.4f}  "
                f"{rec.hard_examples:>10}  {rec.forgetting_events:>12}"
            )
        # Loss trend
        if len(report.epoch_losses) >= 2:
            first_loss = report.epoch_losses[0].mean_loss
            last_loss = report.epoch_losses[-1].mean_loss
            delta = last_loss - first_loss
            trend_sym = "↓" if delta < 0 else "↑"
            _h(f"\n  Loss trend   : {trend_sym} {abs(delta):.4f} over {len(report.epoch_losses)} epochs")
    else:
        _h("  No per-epoch loss records found in log.")

    _h("")
    if report.final_val_loss is not None:
        _h(f"  Final val_loss_vos   : {report.final_val_loss:.4f}")
    if report.final_train_loss is not None:
        _h(f"  Final train_loss_epoch: {report.final_train_loss:.4f}")

    # ── Gradient Health ───────────────────────────────────────────────────
    _h("\n## Gradient Health\n")
    if report.epoch_grads:
        _h(f"  Epochs with npz artifacts : {len(report.epoch_grads)}")
        _h(f"  Global norm trend         : {_norm_trend(report.epoch_grads)}")

        last_grads = report.epoch_grads[-1]
        _h(f"\n  Last epoch ({last_grads.epoch}) — total exploding layers : {last_grads.total_exploding}")
        _h(f"  Last epoch ({last_grads.epoch}) — total vanishing layers : {last_grads.total_vanishing}")

        _h(f"\n  ### Top {top_n} layers by mean gradient norm (epoch {last_grads.epoch})\n")
        _h(f"  {'Layer':<60}  {'Mean Norm':>12}  {'P95 Norm':>12}")
        _h(f"  {'─'*60}  {'─'*12}  {'─'*12}")
        for layer_name, mean_norm, p95_norm in last_grads.worst_layers(top_n):
            flag = " ⚠" if mean_norm > EXPLODING_THRESHOLD else ""
            _h(f"  {layer_name:<60}  {mean_norm:>12.2e}  {p95_norm:>12.2e}{flag}")

        # Norm progression for top 5 worst layers — show at most 12 epochs
        if len(report.epoch_grads) > 1:
            top5 = [name for name, _, _ in last_grads.worst_layers(5)]
            shown_recs = _select_epoch_sample(report.epoch_grads, max_cols=12)
            _h(f"\n  ### Norm progression for worst 5 layers (sampled epochs)\n")
            header_epochs = "  ".join(f"ep{r.epoch:02d}" for r in shown_recs)
            _h(f"  {'Layer':<55}  {header_epochs}")
            _h(f"  {'─'*55}  {'─'*max(len(header_epochs), 4)}")
            for layer in top5:
                row_vals: list[str] = []
                for rec in shown_recs:
                    if layer in rec.layer_names:
                        idx = rec.layer_names.index(layer)
                        row_vals.append(f"{rec.mean_norms[idx]:>6.1e}")
                    else:
                        row_vals.append(f"{'n/a':>6}")
                _h(f"  {layer:<55}  {'  '.join(row_vals)}")
    else:
        _h("  No gradient norm artifacts found.")

    # ── Explosion Events (from log warnings) ─────────────────────────────
    _h("\n## Gradient Explosion Events (from log warnings)\n")
    total_events = len(report.explosion_events)
    _h(f"  Total explosion warnings : {total_events:,}")

    if report.explosion_events:
        first_ev = report.explosion_events[0]
        last_ev = report.explosion_events[-1]
        _h(f"  First explosion at       : {first_ev.timestamp.strftime('%H:%M:%S')}  layer={first_ev.layer_name}  norm={first_ev.norm:.2e}")
        _h(f"  Last explosion at        : {last_ev.timestamp.strftime('%H:%M:%S')}  layer={last_ev.layer_name}  norm={last_ev.norm:.2e}")

        by_module = _explosion_counts_by_module(report.explosion_events)
        _h("\n  Explosions by top-level module:\n")
        _h(f"  {'Module':<40}  {'Count':>8}  {'% of total':>10}")
        _h(f"  {'─'*40}  {'─'*8}  {'─'*10}")
        for module, count in by_module.items():
            pct = 100.0 * count / total_events
            _h(f"  {module:<40}  {count:>8,}  {pct:>9.1f}%")

        # Worst individual layers (by max norm seen)
        layer_max_norms: dict[str, float] = {}
        for ev in report.explosion_events:
            layer_max_norms[ev.layer_name] = max(
                layer_max_norms.get(ev.layer_name, 0.0), ev.norm
            )
        top_layers = sorted(layer_max_norms.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        _h(f"\n  Top {top_n} layers by peak norm during training:\n")
        _h(f"  {'Layer':<60}  {'Peak Norm':>12}")
        _h(f"  {'─'*60}  {'─'*12}")
        for layer_name, peak_norm in top_layers:
            _h(f"  {layer_name:<60}  {peak_norm:>12.2e}")

    # ── Diagnosis ─────────────────────────────────────────────────────────
    _h("\n## Diagnosis\n")
    diagnostics = _diagnose(report)
    for severity, message in diagnostics:
        symbol = {"CRITICAL": "✗", "WARNING": "⚠", "OK": "✓"}.get(severity, "·")
        _h(f"  [{severity:<8}] {symbol} {message}")

    _h("\n" + _THICK)
    return "\n".join(lines)


def _diagnose(report: TrainingReport) -> list[tuple[str, str]]:
    """Generate diagnostic messages from the report data.

    Args:
        report: The aggregated TrainingReport.

    Returns:
        List of (severity, message) tuples. Severity: CRITICAL | WARNING | OK.
    """
    results: list[tuple[str, str]] = []

    # Job state
    if report.job_info:
        state = report.job_info.state
        if state == "COMPLETED":
            results.append(("OK", f"Job completed normally (state={state})"))
        elif state == "FAILED":
            results.append(("CRITICAL", f"Job FAILED (ExitCode={report.job_info.exit_code})"))
        elif state == "TIMEOUT":
            results.append(("WARNING", f"Job hit wall-time limit (state={state})"))
        else:
            results.append(("WARNING", f"Unexpected job state: {state}"))

    # Gradient explosions
    total_exp = len(report.explosion_events)
    if total_exp == 0:
        results.append(("OK", "No gradient explosions detected"))
    elif total_exp < 100:
        results.append(("WARNING", f"{total_exp} explosion warnings — occasional instability"))
    else:
        results.append(
            ("CRITICAL", f"{total_exp:,} explosion warnings — persistent gradient instability")
        )

    # First explosion timing
    if report.explosion_events and report.provenance.git_commit != "unknown":
        first = report.explosion_events[0]
        results.append((
            "CRITICAL",
            f"Explosions started at {first.timestamp.strftime('%H:%M:%S')} "
            f"— likely from step 1, not late-training divergence",
        ))

    # Norm trend
    if len(report.epoch_grads) >= 2:
        trend = _norm_trend(report.epoch_grads)
        if "GROWING" in trend:
            results.append(("CRITICAL", f"Global gradient norm trend: {trend}"))
        elif "shrinking" in trend:
            results.append(("OK", f"Global gradient norm trend: {trend}"))
        else:
            results.append(("OK", f"Global gradient norm trend: {trend}"))

    # Loss trend
    if len(report.epoch_losses) >= 2:
        first_l = report.epoch_losses[0].mean_loss
        last_l = report.epoch_losses[-1].mean_loss
        if last_l < first_l * 0.9:
            results.append(("OK", f"Loss decreasing: {first_l:.4f} → {last_l:.4f}"))
        elif last_l > first_l * 1.1:
            results.append(("CRITICAL", f"Loss increasing: {first_l:.4f} → {last_l:.4f}"))
        else:
            results.append(("WARNING", f"Loss stagnant: {first_l:.4f} → {last_l:.4f}"))

    # Dominant explosion modules
    if report.explosion_events:
        by_module = _explosion_counts_by_module(report.explosion_events)
        top_modules = list(by_module.items())[:3]
        for module, count in top_modules:
            pct = 100.0 * count / len(report.explosion_events)
            if pct > 20:
                results.append((
                    "WARNING",
                    f"Module '{module}' accounts for {pct:.0f}% of explosions ({count:,}) "
                    f"— primary instability site",
                ))

    # out_proj specifically (known Mamba instability point)
    out_proj_events = [
        ev for ev in report.explosion_events if "out_proj" in ev.layer_name
    ]
    if out_proj_events:
        peak = max(ev.norm for ev in out_proj_events)
        results.append((
            "CRITICAL",
            f"out_proj layers exploding (peak norm={peak:.2e}) — "
            f"missing residual scale or output projection not gated",
        ))

    # KAN modulator explosions
    kan_events = [ev for ev in report.explosion_events if "kan" in ev.layer_name.lower()]
    if kan_events:
        peak = max(ev.norm for ev in kan_events)
        results.append((
            "WARNING",
            f"KAN modulator layers exploding (peak norm={peak:.2e}) — "
            f"rbf/base init scale or LN placement may need revisiting",
        ))

    return results


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path=None)
def main(cfg: DictConfig) -> None:
    """Main entry point — parses config, builds report, prints and saves it.

    Args:
        cfg: Hydra config with fields: job_id, log_file, artifacts_dir,
             log_dir, top_n_layers, save_report, output_dir.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    job_id: str | None = str(cfg.get("job_id")) if cfg.get("job_id") is not None else None
    log_file_override: str | None = cfg.get("log_file", None)
    artifacts_dir_str: str = cfg.get("artifacts_dir", ARTIFACTS_GRAD_NORMS)
    log_dir: str = cfg.get("log_dir", SLURM_LOG_DIR)
    top_n: int = int(cfg.get("top_n_layers", 10))
    do_save: bool = bool(cfg.get("save_report", False))
    output_dir: str = cfg.get("output_dir", "analysis")

    # Resolve log file path
    if log_file_override:
        log_path = Path(log_file_override)
    elif job_id:
        # Glob for any file matching *_{job_id}.out in log_dir
        candidates = list(Path(log_dir).glob(f"*_{job_id}.out"))
        if not candidates:
            log.error(
                "No .out log found for job %s in %s. "
                "Provide log_file= explicitly.",
                job_id,
                log_dir,
            )
            raise SystemExit(1)
        log_path = candidates[0]
        log.info("Found log file: %s", log_path)
    else:
        log.error("Provide job_id=<ID> or log_file=<path>")
        raise SystemExit(1)

    artifacts_dir = Path(artifacts_dir_str)
    report = build_report(job_id=job_id, log_path=log_path, grad_norms_dir=artifacts_dir)
    rendered = render_report(report, top_n=top_n)

    print(rendered)

    if do_save:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = log_path.stem
        out_path = out_dir / f"{stem}_analysis.md"
        out_path.write_text(rendered, encoding="utf-8")
        log.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
