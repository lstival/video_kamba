import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from models.video_mamba import VideoMambaSystem
from data.davis import DAVISDataModule
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

def jaccard(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0
    return intersection / union

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = cfg.checkpoint
    if not checkpoint_path:
        print("Error: Please provide a checkpoint path via checkpoint=...")
        return

    print(f"Loading model from {checkpoint_path}...")
    model = VideoMambaSystem.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model.cuda()

    dm = DAVISDataModule(
        data_dir=cfg.datamodule.data_dir,
        batch_size=1,
        num_workers=2,
        seq_len=cfg.datamodule.seq_len,
        img_size=cfg.datamodule.img_size
    )
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    results = {
        "identity": [],
        "shuffle": [],
        "static": [],
        "drift": [],
        "normal": []
    }

    print("Running Bottleneck Tests...")
    for i, batch in enumerate(tqdm(loader)):
        if i >= 10: break # Small sample for speed
        
        ref_img, ref_mask, query_images, query_masks = [x.cuda() for x in batch]
        
        # 1. Normal Inference
        with torch.no_grad():
            _, _, _, logits = model(query_images, ref_frame=ref_img, ref_mask=ref_mask)
            preds = torch.argmax(logits, dim=2)
            ji = jaccard(preds.cpu().numpy() > 0, query_masks.cpu().numpy() > 0)
            results["normal"].append(ji)

        # 2. Identity Test (Is the model consistent?)
        identity_query = query_images.clone()
        identity_query[:, 0] = ref_img
        with torch.no_grad():
            _, _, _, logits = model(identity_query, ref_frame=ref_img, ref_mask=ref_mask)
            preds = torch.argmax(logits, dim=2)[:, 0] # Check first frame
            ji = jaccard(preds.cpu().numpy() > 0, ref_mask.cpu().numpy() > 0)
            results["identity"].append(ji)

        # 3. Shuffle Test (Does temporal order matter?)
        idx = torch.randperm(query_images.shape[1])
        shuffled_query = query_images[:, idx]
        shuffled_masks = query_masks[:, idx]
        with torch.no_grad():
            _, _, _, logits = model(shuffled_query, ref_frame=ref_img, ref_mask=ref_mask)
            preds = torch.argmax(logits, dim=2)
            ji = jaccard(preds.cpu().numpy() > 0, shuffled_masks.cpu().numpy() > 0)
            results["shuffle"].append(ji)

        # 4. Static Test (Does it over-depend on appearance?)
        # Repeat the reference frame 16 times. 
        static_query = ref_img.unsqueeze(1).repeat(1, query_images.shape[1], 1, 1, 1)
        with torch.no_grad():
            _, _, _, logits = model(static_query, ref_frame=ref_img, ref_mask=ref_mask)
            preds = torch.argmax(logits, dim=2)
            # Match against ref_mask repeated
            ref_mask_rep = ref_mask.unsqueeze(1).repeat(1, query_images.shape[1], 1, 1)
            ji = jaccard(preds.cpu().numpy() > 0, ref_mask_rep.cpu().numpy() > 0)
            results["static"].append(ji)

        # 5. Drift Test (Accuracy over time)
        # Check only the LAST frame of the sequence.
        last_pred = preds[:, -1]
        last_gt = query_masks[:, -1]
        ji_last = jaccard(last_pred.cpu().numpy() > 0, last_gt.cpu().numpy() > 0)
        results["drift"].append(ji_last)

    print("\n--- Diagnostic Results (Jaccard Index) ---")
    print(f"Normal Val      : {np.mean(results['normal']):.4f}")
    print(f"Identity Test   : {np.mean(results['identity']):.4f} (Consistency)")
    print(f"Shuffle Test    : {np.mean(results['shuffle']):.4f} (Temporal Sensitivity)")
    print(f"Static Test     : {np.mean(results['static']):.4f} (Appearance Dependency)")
    print(f"Final Frame J   : {np.mean(results['drift']):.4f} (Memory Stability)")
    
    if np.mean(results['identity']) < 0.6:
        print("\n[ALERT] Identity Fail: Spatial context infusion is weak.")
    if abs(np.mean(results['shuffle']) - np.mean(results['normal'])) < 0.05:
        print("[ALERT] Temporal Fail: SSM is ignoring frame order.")
    if np.mean(results['static']) > 0.8:
        print("[ALERT] Static Bias: Model relies too much on start-frame appearance.")

if __name__ == "__main__":
    main()
