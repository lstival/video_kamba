import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.video_mamba import VideoMambaSystem
from data.davis import DAVISDataModule

def test_davis_pipeline():
    print("Initializing DAVIS datamodule...")
    # Using a dummy data_dir because we just want to see if the dataset initializes properly 
    # and if we can mock a batch. If data isn't downloaded, we will mock the tensors.
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "DAVIS")
    
    # Check if data exists
    if not os.path.exists(os.path.join(data_dir, "ImageSets", "2017", "train.txt")):
        print("DAVIS not found locally. Proceeding with dummy tensor shapes for the model test.")
        
        # Simulate DAVIS batch: ref_img, ref_mask, query_images, query_masks
        B = 2
        T = 4
        C = 3
        H = 224
        W = 224
        num_classes = 10
        
        ref_img = torch.randn(B, C, H, W)
        ref_mask = torch.randint(0, num_classes, (B, H, W))
        query_images = torch.randn(B, T, C, H, W)
        query_masks = torch.randint(0, num_classes, (B, T, H, W))
        
        batch = (ref_img, ref_mask, query_images, query_masks)
    
    else:
        print("DAVIS data found. Loading from valid dataset module...")
        dm = DAVISDataModule(data_dir=data_dir, batch_size=2, seq_len=4, img_size=224)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        ref_img, ref_mask, query_images, query_masks = batch
        print(f"Batch shapes: Ref Img: {ref_img.shape}, Ref Mask: {ref_mask.shape}")
        print(f"Query Img: {query_images.shape}, Query Mask: {query_masks.shape}")

    print("Initializing VideoMambaSystem...")
    model = VideoMambaSystem(dim_in=768, num_classes=51) # 51 to cover potential hmdb51 max classes
    model.eval()
    
    print("Testing forward pass...")
    with torch.no_grad():
        logits_clf, pred_boxes, pred_box_logits, logits_seg = model(query_images, ref_frame=ref_img, ref_mask=ref_mask)
        
    print(f"Logits Seg Shape: {logits_seg.shape} (Expected: B, T, num_classes, H, W)")
    
    print("Testing _shared_step for loss computation and metrics update...")
    with torch.no_grad():
        # Validate val step logic
        loss = model._shared_step(batch, 0, prefix="val")
        print(f"Loss computed: {loss.item()}")
        
    print("Testing metric compute...")
    metrics = model.davis_metric.compute()
    print(f"J metrics: {metrics['J'].item():.4f}, F metric: {metrics['F'].item():.4f}, J&F: {metrics['J&F'].item():.4f}")

if __name__ == "__main__":
    test_davis_pipeline()
    print("All tests passed successfully.")
