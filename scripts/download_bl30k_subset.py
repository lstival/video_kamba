"""Script to download and extract a 100GB subset of BL30K Segment 1.

Since the full Segment 1 is ~110GB, this script streams the download and
stops extracting after a specified number of sequences to fit within disk 
space constraints (e.g., 100GB).
"""

import os
import sys
import requests
import tarfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Official Illinois Data Bank link for BL30K Segment 1 (BL30K_a.tar)
BL30K_A_URL = "https://databank.illinois.edu/datafiles/rvqd3/download"

"""Script to download BL30K Segment 1 from Disk E and extract to project folder.

Features:
- Downloads Segment 1 (BL30K_a.tar) to Disk E (~108GB required).
- Extracts files to the project's data/BL30K directory (~115GB required).
- Cleans up the .tar file after successful extraction to free space on Disk E.
"""

import os
import sys
import requests
import tarfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Official Illinois Data Bank link for BL30K Segment 1 (BL30K_a.tar)
BL30K_A_URL = "https://databank.illinois.edu/datafiles/rvqd3/download"

def download_and_extract_bl30k(download_path, extract_dir):
    """
    Downloads the tar file to download_path and extracts it to extract_dir.
    """
    download_path = Path(download_path)
    extract_dir = Path(extract_dir)
    
    os.makedirs(download_path.parent, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    # 1. Download
    if not download_path.exists():
        LOGGER.info(f"Downloading BL30K Segment 1 to {download_path}...")
        try:
            with requests.get(BL30K_A_URL, stream=True) as r:
                r.raise_for_status()
                with open(download_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                        f.write(chunk)
            LOGGER.info("Download complete.")
        except Exception as e:
            LOGGER.error(f"Download failed: {e}")
            return
    else:
        LOGGER.info(f"Download found at {download_path}. Skipping download.")

    # 2. Extract
    LOGGER.info(f"Extracting to {extract_dir}...")
    try:
        with tarfile.open(download_path, 'r') as tar:
            tar.extractall(path=extract_dir)
        LOGGER.info("Extraction complete.")
    except Exception as e:
        LOGGER.error(f"Extraction failed: {e}")
        return

    # 3. Cleanup (Optional)
    choice = input(f"Delete temporary tar file {download_path} to free space on Disk E? (y/n): ")
    if choice.lower() == 'y':
        os.remove(download_path)
        LOGGER.info("Temporary file deleted.")

if __name__ == "__main__":
    # Settings based on user request (Disk E for download, project dir for extraction)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = Path(base_dir).parent
    
    # Target file on disk E
    download_dir_e = Path("E:/")
    download_file_e = download_dir_e / "BL30K_a.tar"
    
    # Target extraction dir in project
    target_extract_dir = project_root / "data" / "BL30K"
    
    print(f"--- BL30K Downloader (Disk E -> Disk C) ---")
    print(f"TEMP DOWNLOAD: {download_file_e} (Needs ~108GB on Disk E)")
    print(f"EXTRACTION: {target_extract_dir} (Needs ~115GB on Disk C)")
    
    choice = input(f"Proceed? (y/n): ")
    if choice.lower() == 'y':
        download_and_extract_bl30k(download_file_e, target_extract_dir)
    else:
        print("Aborted.")
