#!/bin/bash
#SBATCH --job-name=download_yv_ids
#SBATCH --output=/home/WUR/stiva001/WUR/video_kamba/logs/slurm/download_yv_ids_%j.out
#SBATCH --error=/home/WUR/stiva001/WUR/video_kamba/logs/slurm/download_yv_ids_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=main

# Load modules or activate venv if needed
source /home/WUR/stiva001/WUR/video_kamba/venv/bin/activate

# Target directory
DEST_DIR="/home/WUR/stiva001/WUR/video_kamba/data/YouTubeVOS"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Attempting to download train.tar and valid.tar individually..."

# Download train.tar
echo "Downloading train.tar (ID: 1lU9jCX-H0ntwh87tt2cA0xEPeWOJzD6S)..."
gdown --id 1lU9jCX-H0ntwh87tt2cA0xEPeWOJzD6S --output train.tar

# Download valid.tar
echo "Downloading valid.tar (ID: 1bw8KcpzfrT08HYbuROZmY0bp4TkYl4_g)..."
gdown --id 1bw8KcpzfrT08HYbuROZmY0bp4TkYl4_g --output valid.tar

echo "Download complete. Contents of $DEST_DIR:"
ls -lh

# Extraction
if [ -f "train.tar" ]; then
    echo "Extracting train.tar..."
    tar -xf train.tar
    echo "train.tar extraction complete."
fi

if [ -f "valid.tar" ]; then
    echo "Extracting valid.tar..."
    tar -xf valid.tar
    echo "valid.tar extraction complete."
fi

echo "Final organization..."
# Check if extraction created 'train' and 'valid' folders
if [ -d "train" ] && [ -d "valid" ]; then
    echo "Dataset organized successfully."
else
    # Sometimes it extracts into 'YouTubeVOS/train' or similar. 
    # Let's check for any directories and move them if needed.
    ls -d */
fi

echo "Status: READY for pre-training."
