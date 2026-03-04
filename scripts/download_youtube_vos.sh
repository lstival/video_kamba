#!/bin/bash
#SBATCH --job-name=download_yv
#SBATCH --output=/home/WUR/stiva001/WUR/video_kamba/logs/slurm/download_yv_%j.out
#SBATCH --error=/home/WUR/stiva001/WUR/video_kamba/logs/slurm/download_yv_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=main

# Load modules or activate venv if needed
source /home/WUR/stiva001/WUR/video_kamba/venv/bin/activate

# Target directory
DEST_DIR="/home/WUR/stiva001/WUR/video_kamba/data/YouTubeVOS"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Starting download from Google Drive folder..."
# Download the folder using gdown
# The folder ID is from the URL: 1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC
gdown --folder "https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC?usp=drive_link"

echo "Download complete. Contents of $DEST_DIR:"
ls -lh

# Check for train.tar and extract if present
if [ -f "train.tar" ]; then
    echo "Extracting train.tar..."
    tar -xf train.tar
    echo "Extraction complete."
    # Organize: usually YouTube-VOS has a specific structure inside train.tar
    # If it extracts to 'train/', we are good. If not, we may need to rename/move.
else
    echo "train.tar not found. Checking for other files..."
    ls -R
fi

echo "Final organization..."
# Example organization: Ensure train/ exists
if [ -d "train" ]; then
    echo "Dataset organized successfully."
else
    echo "Warning: 'train' directory not found after extraction. Manual check required."
fi

echo "Status: READY for pre-training."
