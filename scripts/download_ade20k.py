import os
import argparse
from datasets import load_dataset

def download_ade20k(cache_dir):
    print(f"Downloading ADE20K (1aurent/ADE20K) to {cache_dir}...")
    
    # Use 1aurent/ADE20K which is stored in Parquet format (modern, script-free)
    
    # Download Training set
    print("Fetching training split...")
    load_dataset(
        "1aurent/ADE20K",
        split="train",
        cache_dir=cache_dir,
        streaming=False
    )
    
    # Download Validation set
    print("Fetching validation split...")
    load_dataset(
        "1aurent/ADE20K",
        split="validation",
        cache_dir=cache_dir,
        streaming=False
    )
    
    print("Download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./data/ade20k_cache")
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    download_ade20k(args.cache_dir)
