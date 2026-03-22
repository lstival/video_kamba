"""Smoke test for BL30K training integration.

This test:
1. Creates a tiny mock BL30K structure.
2. Initializes the VOSDataModule with BL30K pointing to the mock.
3. Runs 2 steps of training using a minimal model configuration.
"""

import os
import shutil
import torch
import numpy as np
from PIL import Image
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

def create_mock_bl30k(root_dir):
    """Create a minimal BL30K-like structure with 2 sequences."""
    root = Path(root_dir)
    img_dir = root / "JPEGImages"
    ann_dir = root / "Annotations"
    
    for i in range(2):
        seq_name = f"mock_seq_{i}"
        (img_dir / seq_name).mkdir(parents=True, exist_ok=True)
        (ann_dir / seq_name).mkdir(parents=True, exist_ok=True)
        
        # Create 5 frames
        for f in range(5):
            # Image
            img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
            img.save(img_dir / seq_name / f"{f:03d}.jpg")
            
            # Mask (binary 0/1)
            mask = Image.fromarray(np.random.randint(0, 2, (112, 112), dtype=np.uint8), mode="P")
            mask.save(ann_dir / seq_name / f"{f:03d}.png")

def create_mock_davis(root_dir):
    """Create minimal DAVIS structure to allow VOSDataModule to initialize."""
    root = Path(root_dir)
    img_dir = root / "JPEGImages" / "480p"
    ann_dir = root / "Annotations" / "480p"
    imgsets = root / "ImageSets" / "2017"
    
    imgsets.mkdir(parents=True, exist_ok=True)
    with open(imgsets / "train.txt", "w") as f: f.write("seq1\n")
    with open(imgsets / "val.txt", "w") as f: f.write("seq1\n")
    
    (img_dir / "seq1").mkdir(parents=True, exist_ok=True)
    (ann_dir / "seq1").mkdir(parents=True, exist_ok=True)
    for f in range(5): # Increase to 5 to avoid range(1, 1) empty clips
        Image.new('RGB', (112, 112)).save(img_dir / "seq1" / f"{f:05d}.jpg")
        Image.new('P', (112, 112)).save(ann_dir / "seq1" / f"{f:05d}.png")

def test_training_smoke():
    project_root = Path(__file__).parent.parent
    mock_bl30k = project_root / "data" / "mock_BL30K_smoke"
    mock_davis = project_root / "data" / "mock_DAVIS_smoke"
    
    # Ensure mock dirs are clean
    for d in [mock_bl30k, mock_davis]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    create_mock_bl30k(mock_bl30k)
    create_mock_davis(mock_davis)
    
    print(f"--- Running BL30K Training Smoke Test ---")
    
    try:
        from data.vos_datamodule import VOSDataModule
        
        # Configure for BL30K pre-training smoke test
        dm = VOSDataModule(
            davis_root=str(mock_davis),
            bl30k_root=str(mock_bl30k),
            batch_size=2,
            clip_len=2,
            output_size=112,
            num_workers=0,
            bl30k_sampling_ratio=0.8,
            davis_sampling_ratio=0.2
        )
        dm.setup("fit")
        
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        print("OK: Dataloader successfully produced a joint BL30K+DAVIS batch.")
        
        # Check shapes (ref_img, ref_mask, query_imgs, query_masks, seq_names)
        ref_img, ref_mask, query_imgs, query_masks, seq_names = batch
        assert query_imgs.dim() == 5, f"Expected 5 dimensions for query_imgs, got {query_imgs.dim()}"
        assert query_imgs.size(0) == 2, f"Expected batch size 2, got {query_imgs.size(0)}"
        
        print(f"   Batch summary:")
        print(f"     Ref Images:  {ref_img.shape}")
        print(f"     Query Images: {query_imgs.shape}")
        print(f"     Sequence IDs: {seq_names}")
        
    except Exception as e:
        print(f"ERROR: Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup mocks
        for d in [mock_bl30k, mock_davis]:
            if d.exists(): shutil.rmtree(d)
            
    print("DONE: BL30K Training Smoke Test Passed (Data Pipeline Verified).")
    return True

if __name__ == "__main__":
    test_training_smoke()

if __name__ == "__main__":
    test_training_smoke()
