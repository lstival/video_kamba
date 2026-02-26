import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import os
from datetime import datetime
from models.video_mamba import VideoMambaSystem

def visualize_batch(model, batch, save_dir, batch_idx=0):
    # Determine batch structure
    # HMDB51: (frames, labels)
    # DAVIS: (frames, masks)
    # VideoDataset: (frames, labels, boxes, box_labels)
    
    frames = batch[0].to(model.device)
    labels = batch[1] if len(batch) >= 2 else None
    
    with torch.no_grad():
        logits_clf, pred_boxes, pred_box_logits, logits_seg = model(frames)
    
    # Get predictions
    pred_classes = torch.argmax(logits_clf, dim=1)
    
    # Create directory for this batch
    batch_dir = os.path.join(save_dir, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)
    
    B, T, C, H, W = frames.shape
    
    for b in range(B):
        fig, axes = plt.subplots(1, T, figsize=(24, 5))
        if T == 1:
            axes = [axes]
        
        # Get scores
        # Detection scores: [B, T, num_boxes]
        box_scores = torch.softmax(pred_box_logits, dim=-1)
        # Max score per box (excluding background if class 0 is background)
        max_box_scores, max_box_classes = torch.max(box_scores[b], dim=-1)
        
        # Classification info
        gt_cls = labels[b].item() if labels is not None and labels.dim() == 1 else "N/A"
        pr_cls = pred_classes[b].item()
        
        # Segmentation probabilities for overlay intensity
        seg_probs = torch.softmax(logits_seg, dim=2) # [B, T, C, H, W]
        
        for t in range(T):
            # Frame [C, H, W] -> [H, W, C]
            img = frames[b, t].cpu().permute(1, 2, 0).numpy()
            
            # Simple de-normalization (assuming ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[t].imshow(img)
            
            # 1. Overlay Predicted Segmentation (as heatmap)
            mask_logits = logits_seg[b, t] # [C, H, W]
            mask_probs = torch.softmax(mask_logits, dim=0)
            # Sum probabilities of all classes except background (0)
            obj_prob = 1.0 - mask_probs[0].cpu().numpy()
            
            # Show heatmap
            if obj_prob.max() > 0.01: # Small threshold to avoid showing pure noise
                # Use a specific colormap for the heatmap (e.g., 'jet' or 'viridis')
                # Mask out very low probability areas
                masked_prob = np.ma.masked_where(obj_prob < 0.1, obj_prob)
                axes[t].imshow(masked_prob, alpha=0.5, cmap='jet', vmin=0, vmax=1)
            
            # 2. Overlay Ground Truth Mask (Contour)
            if labels is not None and labels.dim() == 4: # [B, T, H, W] (DAVIS)
                gt_mask = labels[b, t].cpu().numpy()
                if gt_mask.max() > 0:
                    axes[t].contour(gt_mask, levels=[0.5], colors='white', linewidths=0.8)

            # 3. Overlay Predicted Bounding Boxes
            # Show boxes with confidence > 0.3
            for box_idx in range(pred_boxes.shape[2]):
                score = max_box_scores[t, box_idx].item()
                if score > 0.3:
                    box = pred_boxes[b, t, box_idx].cpu().numpy()
                    cls_id = max_box_classes[t, box_idx].item()
                    
                    xc, yc, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
                    x1, y1 = xc - w/2, yc - h/2
                    
                    rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red', linewidth=1.2)
                    axes[t].add_patch(rect)
                    axes[t].text(x1, y1, f"{cls_id}:{score:.2f}", color='white', 
                                 fontsize=6, backgroundcolor='red')
            
            axes[t].axis('off')
            if t == 0:
                axes[t].set_title(f"GT: {gt_cls} | PR: {pr_cls}", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"sample_{b}.png"), bbox_inches='tight', dpi=150)
        plt.close()

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint={checkpoint_path} invalid.")
        return

    print(f"Loading model from {checkpoint_path}...")
    # Use strict=False to handle potential code changes.
    # We look for overrides in global config (CLI) or model sub-config.
    num_seg = cfg.get("num_seg_classes") or cfg.model.get("num_seg_classes", 11)
    num_clf = cfg.get("num_clf_classes") or cfg.model.get("num_clf_classes", 51)
    
    print(f"Instantiating model with: num_seg_classes={num_seg}, num_clf_classes={num_clf}")

    model = VideoMambaSystem.load_from_checkpoint(
        checkpoint_path, 
        strict=False,
        num_seg_classes=num_seg,
        num_clf_classes=num_clf
    )
    model.eval()
    
    print(f"Initializing datamodule: {cfg.datamodule._target_}")
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("eval_results", f"visuals_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating visualizations in {save_dir}...")
    
    # Take first batch
    batch = next(iter(test_loader))
    visualize_batch(model.to("cpu"), batch, save_dir)
    
    print("Visual evaluation completed.")

if __name__ == "__main__":
    main()
