import torch
import torch.nn.functional as F
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig
import os
import json
import glob
from datetime import datetime
from PIL import Image
from models.video_mamba import VideoMambaSystem

def save_gif_with_cv2(frames, save_path, fps=8):
    """Save a list of numpy frames as a video/GIF using OpenCV fallback for now or PIL."""
    # Since writing GIFs with OpenCV is tricky, we'll use PIL's native capability
    pil_frames = [Image.fromarray(f) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),
            loop=0
        )

def export_mask(mask, save_path):
    """Save a segmentation mask as a grayscale PNG."""
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(save_path)

def visualize_sequence(model, dataset, clip_idx, save_dir):
    """Generate visuals (masks, GIFs) for a specific sequence/clip."""
    batch = dataset[clip_idx]
    # batch: ref_img, ref_mask, query_images, query_masks
    ref_img, ref_mask, query_images, query_masks = [b.unsqueeze(0).to(model.device) for b in batch]
    
    with torch.no_grad():
        _, _, _, logits_seg = model(query_images, ref_frame=ref_img, ref_mask=ref_mask)
    
    preds = torch.argmax(logits_seg, dim=2).squeeze(0).cpu().numpy() # [T, H, W]
    gt_masks = query_masks.squeeze(0).cpu().numpy() # [T, H, W]
    query_imgs = query_images.squeeze(0).cpu() # [T, C, H, W]
    
    seq_name = dataset.clips[clip_idx]['seq']
    seq_dir = os.path.join(save_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "masks"), exist_ok=True)
    
    gif_frames = []
    
    # Pre-define colors for up to 10 objects
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]
    
    for t in range(preds.shape[0]):
        # Save raw predicted mask
        export_mask(preds[t], os.path.join(seq_dir, "masks", f"frame_{t:04d}.png"))
        
        # Prepare GIF frame (Overlay) with OpenCV
        img = query_imgs[t].permute(1, 2, 0).numpy()
        # De-normalize ImageNet
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Overlay prediction
        mask = preds[t]
        overlay = img_bgr.copy()
        for obj_id in range(1, int(mask.max()) + 1):
            overlay[mask == obj_id] = colors[(obj_id - 1) % len(colors)]
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Overlay GT contour
        if gt_masks[t].max() > 0:
            for obj_id in range(1, int(gt_masks[t].max()) + 1):
                contours, _ = cv2.findContours(
                    (gt_masks[t] == obj_id).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img_bgr, contours, -1, (255, 255, 255), 1)
        
        # Convert back to RGB for PIL GIF
        gif_frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
    save_gif_with_cv2(gif_frames, os.path.join(seq_dir, f"{seq_name}_overlay.gif"))
    print(f"Exported visuals for {seq_name} to {seq_dir}")

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint={checkpoint_path} invalid.")
        return

    # Determine look-up directory
    output_subdir = cfg.get("output_subdir", "")
    base_dir = os.path.join("eval_results", output_subdir)
    
    # Find latest manifest
    manifest_files = sorted(glob.glob(os.path.join(base_dir, "top_5_manifest_*.json")))
    if not manifest_files:
        print(f"Error: No top_5_manifest found in {base_dir}. Please run eval_metrics.py first.")
        return
    
    with open(manifest_files[-1], 'r') as f:
        top_5 = json.load(f)

    print(f"Loading model from {checkpoint_path}...")
    num_seg = cfg.get("num_seg_classes") or cfg.model.get("num_seg_classes", 11)
    num_clf = cfg.get("num_clf_classes") or cfg.model.get("num_clf_classes", 51)
    
    model = VideoMambaSystem.load_from_checkpoint(
        checkpoint_path, 
        strict=False,
        num_seg_classes=num_seg,
        num_clf_classes=num_clf
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup(stage="test")
    dataset = dm.test_dataset
    
    save_dir = os.path.join(base_dir, f"top_5_visuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Using manifest: {manifest_files[-1]}")
    for res in top_5:
        visualize_sequence(model, dataset, res['clip_idx'], save_dir)

if __name__ == "__main__":
    main()
