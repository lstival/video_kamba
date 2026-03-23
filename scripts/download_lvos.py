"""Download and extract the LVOS V2 dataset to a target directory.

LVOS V2 (Long-term Video Object Segmentation Benchmark, ICCV 2023) is hosted
on Google Drive.  This script uses ``gdown`` to download each file by its
Drive file ID, then unzips it in-place and deletes the zip to keep peak disk
usage below one file at a time (~25 GB).

Official page  : https://lingyihongfd.github.io/lvos.github.io/dataset.html
Evaluation kit : https://github.com/LingyiHongfd/lvos-evaluation

FILE IDs
--------
The Google Drive file IDs below are taken from the LVOS V2 official release.
If any download fails with a permission or quota error, visit the official page
above to obtain the current IDs and update ``LVOS_FILES`` accordingly.

Usage (from SLURM job scripts/slurm_setup_lvos_lustre.sh):

    python scripts/download_lvos.py \\
        --splits train valid \\
        --download-dir  /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/lvos_zips \\
        --extract-dir   /lustre/scratch/WUR/AIN/stiva001/video_kamba_data/LVOS \\
        --cleanup

Usage (single split, quick test):

    python scripts/download_lvos.py --splits valid --cleanup
"""

from __future__ import annotations

import argparse
import logging
import os
import zipfile
from pathlib import Path

import gdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── LVOS V2 Google Drive file IDs ─────────────────────────────────────────────
# Source: https://lingyihongfd.github.io/lvos.github.io/dataset.html
# Verify current IDs at the official page if downloads fail.
#
# Each entry: (google_drive_file_id, local_filename, split_key)
LVOS_FILES: dict[str, list[tuple[str, str]]] = {
    "train": [
        # Training images + annotations bundled together
        # Source: https://lingyihongfd.github.io/lvos.github.io/dataset.html
        ("1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6", "train.zip"),
    ],
    "valid": [
        # Validation images + annotations bundled together
        ("17Hwc__6i2rpF5e2s5OPqoywNxG5bzlcO", "valid.zip"),
    ],
}

DEFAULT_DOWNLOAD_DIR = Path("/lustre/scratch/WUR/AIN/stiva001/video_kamba_data/lvos_zips")
DEFAULT_EXTRACT_DIR  = Path("/lustre/scratch/WUR/AIN/stiva001/video_kamba_data/LVOS")

ALL_SPLITS: list[str] = ["train", "valid"]


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_gdrive(file_id: str, dest_path: Path) -> None:
    """Download a single Google Drive file with gdown."""
    if dest_path.exists():
        LOGGER.info("  SKIP (already exists): %s", dest_path)
        return
    LOGGER.info("  Downloading %s → %s", file_id, dest_path)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest_path), quiet=False, fuzzy=True)
    if not dest_path.exists():
        raise RuntimeError(f"gdown did not produce expected file: {dest_path}")
    LOGGER.info("  Downloaded: %s  (%.1f GB)", dest_path.name, dest_path.stat().st_size / 1e9)


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip archive and log progress."""
    LOGGER.info("  Extracting %s → %s", zip_path.name, extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.namelist())
        LOGGER.info("    Members: %d", total)
        zf.extractall(str(extract_dir))
    LOGGER.info("  Extraction complete: %s", zip_path.name)


def _download_split(
    split: str,
    download_dir: Path,
    extract_dir: Path,
    cleanup: bool,
) -> None:
    """Download and extract all files for one LVOS split."""
    files = LVOS_FILES.get(split)
    if not files:
        LOGGER.warning("Unknown split '%s' — skipping.", split)
        return

    LOGGER.info("=" * 60)
    LOGGER.info("  LVOS %s  (%d files)", split.upper(), len(files))
    LOGGER.info("=" * 60)

    for file_id, filename in files:
        zip_path = download_dir / filename
        LOGGER.info("── %s ──", filename)

        # 1. Download
        _download_gdrive(file_id, zip_path)

        # 2. Extract
        _extract_zip(zip_path, extract_dir)

        # 3. Optionally delete zip to free space
        if cleanup:
            zip_path.unlink()
            LOGGER.info("  Deleted zip: %s", filename)

    LOGGER.info("  %s done.", split.upper())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LVOS V2 dataset from Google Drive via gdown.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        choices=ALL_SPLITS,
        metavar="SPLIT",
        help="Which splits to download. Default: train valid",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory for temporary zip files (one at a time).",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help="Destination directory for extracted LVOS data.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Delete each zip after extraction to conserve disk space.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("LVOS V2 Download")
    LOGGER.info("  Splits       : %s", " ".join(args.splits))
    LOGGER.info("  Download dir : %s", args.download_dir)
    LOGGER.info("  Extract dir  : %s", args.extract_dir)
    LOGGER.info("  Cleanup zips : %s", args.cleanup)
    LOGGER.info("")

    for split in args.splits:
        _download_split(split, args.download_dir, args.extract_dir, args.cleanup)

    LOGGER.info("")
    LOGGER.info("All requested splits downloaded and extracted.")
    LOGGER.info("Dataset location: %s", args.extract_dir)

    # Verify expected structure
    for split in args.splits:
        img_dir = args.extract_dir / split / "JPEGImages"
        ann_dir = args.extract_dir / split / "Annotations"
        if img_dir.is_dir() and ann_dir.is_dir():
            n_seqs = len(list(img_dir.iterdir()))
            LOGGER.info("  %s: %d sequences found in JPEGImages/", split, n_seqs)
        else:
            LOGGER.warning(
                "  %s: expected structure not found at %s",
                split, args.extract_dir / split,
            )


if __name__ == "__main__":
    main()
