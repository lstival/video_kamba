import torch
import lightning as L
import hydra
from omegaconf import DictConfig
import os
import json
from datetime import datetime
import numpy as np
from models.video_mamba import VideoMambaSystem
from utils.davis_metrics import db_eval_iou, db_eval_boundary

def evaluate_subset(pred_mask, gt_mask, obj_ids):
    """
    Evaluates J and F scores for a subset of object IDs.
    pred_mask: [T, H, W] tensor of class indices
    gt_mask: [T, H, W] tensor of class indices
    obj_ids: list of object IDs to evaluate
    """
    pred_np = pred_mask.cpu().numpy()
    gt_np = gt_mask.cpu().numpy()
    
    # Ignore void index (255)
    pred_np[gt_np == 255] = 0
    
    j_scores = []
    f_scores = []
    
    for obj_id in obj_ids:
        pred_bin = (pred_np == obj_id)
        gt_bin = (gt_np == obj_id)
        
        if gt_bin.sum() == 0:
            continue
            
        obj_j = []
        obj_f = []
        for t in range(pred_bin.shape[0]):
            j = db_eval_iou(gt_bin[t], pred_bin[t])
            f = db_eval_boundary(gt_bin[t], pred_bin[t])
            obj_j.append(j)
            obj_f.append(f)
        j_scores.append(np.mean(obj_j))
        f_scores.append(np.mean(obj_f))
            
    return j_scores, f_scores

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        print("Error: Please provide a checkpoint path via checkpoint=path/to/ckpt")
        return

    print(f"Loading model from {checkpoint_path}...")
    
    num_seg = cfg.get("num_seg_classes") or cfg.model.get("num_seg_classes", 11)
    num_clf = cfg.get("num_clf_classes") or cfg.model.get("num_clf_classes", 51)
    
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
    
    seen_j, seen_f = [], []
    unseen_j, unseen_f = [], []
    all_j, all_f = [], []
    
    print("Starting YouTube-VOS evaluation with Seen/Unseen split...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            ref_img, ref_mask, query_images, query_masks, _, metas = batch
            
            ref_img = ref_img.to(model.device)
            ref_mask = ref_mask.to(model.device)
            query_images = query_images.to(model.device)
            query_masks = query_masks.to(model.device)
            
            _, _, _, logits_seg = model(query_images, ref_frame=ref_img, ref_mask=ref_mask)
            preds = torch.argmax(logits_seg, dim=2) # [B, T, H, W]
            
            for b in range(preds.shape[0]):
                meta = metas[b]
                video_id = meta['video_id']
                seen_ids = meta.get('seen_obj_ids', [])
                unseen_ids = meta.get('unseen_obj_ids', [])
                
                # Filter IDs by identity bank size (n_id) if necessary
                n_id = cfg.datamodule.n_id
                seen_ids = [idx for idx in seen_ids if idx <= n_id]
                unseen_ids = [idx for idx in unseen_ids if idx <= n_id]
                
                # Evaluate Seen
                sj, sf = evaluate_subset(preds[b], query_masks[b], seen_ids)
                seen_j.extend(sj)
                seen_f.extend(sf)
                
                # Evaluate Unseen
                uj, uf = evaluate_subset(preds[b], query_masks[b], unseen_ids)
                unseen_j.extend(uj)
                unseen_f.extend(uf)
                
                # Overall
                all_j.extend(sj + uj)
                all_f.extend(sf + uf)
                
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches...")

    # Calculate means
    results = {
        "J_seen": np.mean(seen_j) if seen_j else 0.0,
        "F_seen": np.mean(seen_f) if seen_f else 0.0,
        "J_unseen": np.mean(unseen_j) if unseen_j else 0.0,
        "F_unseen": np.mean(unseen_f) if unseen_f else 0.0,
        "J_all": np.mean(all_j) if all_j else 0.0,
        "F_all": np.mean(all_f) if all_f else 0.0,
    }
    
    results["Mean"] = (results["J_all"] + results["F_all"]) / 2
    results["Overall_Mean"] = (results["J_seen"] + results["F_seen"] + results["J_unseen"] + results["F_unseen"]) / 4
    
    print("\n--- YouTube-VOS Evaluation Results ---")
    print(f"J Seen:   {results['J_seen']:.4f}")
    print(f"F Seen:   {results['F_seen']:.4f}")
    print(f"J Unseen: {results['J_unseen']:.4f}")
    print(f"F Unseen: {results['F_unseen']:.4f}")
    print(f"Mean J&F: {results['Overall_Mean']:.4f}")
    
    # Save results
    save_dir = "eval_results/youtubevos"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(save_dir, f"results_{timestamp}.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()
