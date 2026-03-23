import os
import sys

# Ensure data directory is in path if not run as module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.hmdb51 import HMDB51DataModule
from data.davis import DAVISDataModule

def test_hmdb51():
    print("Testing HMDB51DataModule...")
    hmdb_dir = os.path.join("data", "HMDB51", "extracted", "hmdb51")
    if not os.path.exists(hmdb_dir):
        print(f"Skipping HMDB51, directory not found: {hmdb_dir}")
        return
        
    dm = HMDB51DataModule(hmdb_dir, batch_size=2)
    dm.setup("fit")
    loader = dm.train_dataloader()
    
    print(f"HMDB51 Train Dataset Size: {len(dm.train_dataset)}")
    
    for batch_idx, (frames, labels) in enumerate(loader):
        print(f"Batch {batch_idx+1}:")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Frames min/max: {frames.min().item():.3f} / {frames.max().item():.3f}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break
    print("HMDB51 testing complete.\n")


def test_davis():
    print("Testing DAVISDataModule...")
    davis_dir = os.path.join("data", "DAVIS", "DAVIS")
    if not os.path.exists(davis_dir):
        print(f"Skipping DAVIS, directory not found: {davis_dir}")
        return
        
    dm = DAVISDataModule(davis_dir, batch_size=2)
    dm.setup("fit")
    loader = dm.train_dataloader()
    
    print(f"DAVIS Train Dataset Size: {len(dm.train_dataset)}")
    
    for batch_idx, (frames, masks) in enumerate(loader):
        print(f"Batch {batch_idx+1}:")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Frames min/max: {frames.min().item():.3f} / {frames.max().item():.3f}")
        print(f"  Masks shape: {masks.shape}")
        unique_mask_vals = masks.unique().tolist()
        print(f"  Unique mask values: {unique_mask_vals}")
        break
    print("DAVIS testing complete.\n")


if __name__ == "__main__":
    print("Running DataModule Tests...\n")
    test_hmdb51()
    test_davis()
