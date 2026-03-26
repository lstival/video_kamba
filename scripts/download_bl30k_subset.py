"""Download and extract the full BL30K dataset (all 6 segments, ~700 GB).

All segments are downloaded sequentially: each tar is fetched, extracted into
``extract_dir``, then deleted before the next segment starts — keeping peak
disk usage to one segment at a time (~120 GB).

Official dataset page: https://henghuiding.github.io/MIVOS/ (BL30K section)
Illinois Data Bank:    https://databank.illinois.edu/datasets/IDB-1702934

Usage (SLURM / non-interactive):
    python scripts/download_bl30k_subset.py \\
        --download-dir  /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/tars \\
        --extract-dir   /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/BL30K \\
        --cleanup

Usage (single segment, backward-compatible):
    python scripts/download_bl30k_subset.py \\
        --segments a \\
        --download-dir  /tmp/stiva001 \\
        --extract-dir   /home/WUR/stiva001/WUR/video_kamba/data/BL30K \\
        --cleanup
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Illinois Data Bank download links ────────────────────────────────────────
# All links are from: https://databank.illinois.edu/datasets/IDB-1702934
# Verify current links on the dataset page if any redirect fails.
BL30K_URLS: dict[str, str] = {
    "a": "https://databank.illinois.edu/datafiles/rvqd3/download",
    "b": "https://databank.illinois.edu/datafiles/mgxzg/download",
    "c": "https://databank.illinois.edu/datafiles/gmu94/download",
    "d": "https://databank.illinois.edu/datafiles/osrh3/download",
    "e": "https://databank.illinois.edu/datafiles/wqjjv/download",
    "f": "https://databank.illinois.edu/datafiles/t86no/download",
}

ALL_SEGMENTS: list[str] = ["a", "b", "c", "d", "e", "f"]

DEFAULT_DOWNLOAD_DIR = Path("/lustre/scratch/WUR/AIN/stiva001/video_kamba_data/tars")
DEFAULT_EXTRACT_DIR  = Path("/lustre/scratch/WUR/AIN/stiva001/video_kamba_data/BL30K")


# ── Core helpers ─────────────────────────────────────────────────────────────

def download_segment(segment: str, download_dir: Path, url: str | None = None) -> Path | None:
    """Download one BL30K segment tar; skip if already present.

    Args:
        segment:      Segment letter, e.g. ``"a"``.
        download_dir: Directory to save the tar file into.
        url:          Override download URL; falls back to ``BL30K_URLS[segment]``.

    Returns:
        Path to the downloaded tar, or ``None`` on failure.
    """
    resolved_url = url or BL30K_URLS.get(segment)
    if not resolved_url:
        LOGGER.error("No URL configured for segment '%s'. Add it to BL30K_URLS.", segment)
        return None

    tar_path = download_dir / f"BL30K_{segment}.tar"
    if tar_path.exists():
        LOGGER.info("[%s] Archive already exists at %s — skipping download.", segment, tar_path)
        return tar_path

    download_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[%s] Downloading: %s → %s", segment, resolved_url, tar_path)

    try:
        with requests.get(resolved_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB — larger chunks for Lustre throughput

            with open(tar_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        LOGGER.info(
                            "[%s]  %.1f%%  (%d / %d MB)",
                            segment,
                            100 * downloaded / total_bytes,
                            downloaded // (1024 ** 2),
                            total_bytes // (1024 ** 2),
                        )

        LOGGER.info("[%s] Download complete: %s", segment, tar_path)
        return tar_path

    except Exception as exc:
        LOGGER.error("[%s] Download failed: %s", segment, exc)
        if tar_path.exists():
            tar_path.unlink()
        return None


def extract_segment(tar_path: Path, extract_dir: Path) -> bool:
    """Extract one BL30K segment tar into ``extract_dir``.

    Args:
        tar_path:    Path to the .tar archive.
        extract_dir: Destination directory.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s → %s …", tar_path.name, extract_dir)
    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            LOGGER.info("  Members in archive: %d", len(members))
            tar.extractall(path=extract_dir)
        LOGGER.info("Extraction complete: %s", tar_path.name)
        return True
    except Exception as exc:
        LOGGER.error("Extraction failed for %s: %s", tar_path.name, exc)
        return False


def process_segment(
    segment: str,
    download_dir: Path,
    extract_dir: Path,
    cleanup: bool,
    url: str | None = None,
) -> bool:
    """Download, extract, and optionally delete one segment.

    Args:
        segment:      Segment letter.
        download_dir: Where to save the tar.
        extract_dir:  Where to extract.
        cleanup:      Delete tar after successful extraction.
        url:          Optional URL override.

    Returns:
        ``True`` if the segment was fully processed, ``False`` otherwise.
    """
    tar_path = download_dir / f"BL30K_{segment}.tar"
    already_extracted = _segment_already_extracted(segment, extract_dir)

    if already_extracted:
        LOGGER.info("[%s] Already extracted — skipping.", segment)
        return True

    # Download if tar not already on disk
    if not tar_path.exists():
        tar_path = download_segment(segment, download_dir, url=url)
        if tar_path is None:
            return False

    ok = extract_segment(tar_path, extract_dir)
    if not ok:
        return False

    if cleanup:
        LOGGER.info("[%s] Removing tar %s …", segment, tar_path)
        tar_path.unlink()

    return True


def _segment_already_extracted(segment: str, extract_dir: Path) -> bool:
    """Heuristic: segment is extracted if its top-level subdir exists and is non-empty."""
    candidate = extract_dir / f"BL30K_{segment}"
    if candidate.is_dir() and any(candidate.iterdir()):
        return True
    # Also handle flat layout where all sequences land directly in extract_dir
    # (for segment 'a' only, as older tarballs may not have a subdirectory).
    return False


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + extract the full BL30K dataset (all 6 segments)."
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=ALL_SEGMENTS,
        choices=ALL_SEGMENTS,
        metavar="SEGMENT",
        help="Which segments to download (default: all — a b c d e f).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Directory to save .tar files (default: {DEFAULT_DOWNLOAD_DIR}).",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help=f"Where to extract the dataset (default: {DEFAULT_EXTRACT_DIR}).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete each .tar after successful extraction (saves ~120 GB per segment).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; go straight to extraction (tars must already exist).",
    )
    # Backward-compat alias: --download-path maps to --download-dir
    parser.add_argument(
        "--download-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # legacy; use --download-dir
    )
    args = parser.parse_args()

    # Legacy compatibility: if --download-path passed, derive download-dir from it
    if args.download_path is not None:
        args.download_dir = args.download_path.parent

    LOGGER.info("=== BL30K Full Dataset Downloader ===")
    LOGGER.info("  Segments  : %s", args.segments)
    LOGGER.info("  Download  : %s", args.download_dir)
    LOGGER.info("  Extract   : %s", args.extract_dir)
    LOGGER.info("  Cleanup   : %s", args.cleanup)
    LOGGER.info("  Total     : ~%d GB", 120 * len(args.segments))

    failed: list[str] = []

    for seg in args.segments:
        LOGGER.info("")
        LOGGER.info("─── Segment %s (%d / %d) ───", seg, args.segments.index(seg) + 1, len(args.segments))

        if args.skip_download:
            tar_path = args.download_dir / f"BL30K_{seg}.tar"
            ok = extract_segment(tar_path, args.extract_dir)
            if not ok:
                failed.append(seg)
            elif args.cleanup and tar_path.exists():
                tar_path.unlink()
        else:
            ok = process_segment(
                segment=seg,
                download_dir=args.download_dir,
                extract_dir=args.extract_dir,
                cleanup=args.cleanup,
            )
            if not ok:
                failed.append(seg)

    LOGGER.info("")
    if failed:
        LOGGER.error("=== FAILED segments: %s ===", failed)
        raise SystemExit(1)

    LOGGER.info("=== Done — all segments processed successfully ===")


if __name__ == "__main__":
    main()
