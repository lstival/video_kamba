import time
import torch
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
from datetime import datetime
from PIL import Image
from models.video_mamba import VideoMambaSystem

WARMUP_BATCHES = 5

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Ensure checkpoint is provided
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        print("Error: Please provide a checkpoint path via checkpoint=path/to/ckpt")
        return

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist.")
        return

    print(f"Loading model from {checkpoint_path}...")
    
    # Load model from checkpoint with strict=False to handle potential code changes.
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
    
    # Initialize DataModule using Hydra
    print(f"Initializing datamodule: {cfg.datamodule._target_}")
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    # Initialize Trainer for testing
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(accelerator=accelerator, devices=1, logger=True)
    
    print("Starting quantitative evaluation...")
    # Track per-sequence metrics manually by iterating through the dataloader
    # since we need specific identifiers for the top-5 logic.
    sequence_results = []
    use_cuda = torch.cuda.is_available()

    # FPS tracking (GPU-only timing, excludes data loading)
    total_infer_time = 0.0  # seconds
    total_frames = 0        # query frames counted after warm-up

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # batch unpacking for MultiObjectVOSDataset (metas as list of dicts)
            if len(batch) == 6:
                ref_img, ref_mask, query_images, query_masks, _, metas = batch
            else:
                # Fallback for old/other datasets
                ref_img, ref_mask, query_images, query_masks = batch
                metas = None

            # Move tensors to device
            ref_img = ref_img.to(model.device)
            ref_mask = ref_mask.to(model.device)
            query_images = query_images.to(model.device)
            query_masks = query_masks.to(model.device)

            # --- FPS: GPU-synchronised timing after warm-up ---
            is_warmup = batch_idx < WARMUP_BATCHES
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Forward pass
            _, _, _, logits_seg = model(query_images, ref_frame=ref_img, ref_mask=ref_mask)

            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if not is_warmup:
                # query_images: [B, T, C, H, W] → T query frames per sample
                batch_frames = query_images.shape[0] * query_images.shape[1]
                total_frames += batch_frames
                total_infer_time += elapsed

            preds = torch.argmax(logits_seg, dim=2) # [B, T, H, W]

            # Update metrics per sample in batch
            for b in range(preds.shape[0]):
                # Identify sequence name
                if metas is not None:
                    seq_name = metas[b]['video_id']
                else:
                    clip_info = dm.test_dataset.clips[batch_idx * cfg.datamodule.batch_size + b]
                    seq_name = clip_info['seq']
                
                # Compute J&F for this specific sample
                num_objects = int(query_masks[b].max().item())
                if num_objects == 0:
                    continue
                    
                from utils.davis_metrics import evaluate_j_f
                j, f = evaluate_j_f(preds[b], query_masks[b], num_objects)
                
                sequence_results.append({
                    'seq': seq_name,
                    'clip_idx': batch_idx * cfg.datamodule.batch_size + b,
                    'J': float(j),
                    'F': float(f),
                    'JF': float((j + f) / 2)
                })
                
    # --- FPS summary ---
    fps = total_frames / total_infer_time if total_infer_time > 0 else 0.0
    fps_info = {
        "fps": round(fps, 2),
        "total_frames_measured": total_frames,
        "total_infer_time_s": round(total_infer_time, 4),
        "warmup_batches_skipped": WARMUP_BATCHES,
        "device": str(model.device),
    }
    print(f"\n--- FPS (forward pass only, after {WARMUP_BATCHES} warm-up batches) ---")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Infer time : {total_infer_time:.4f} s")

    # Sort and pick top 5
    top_5 = sorted(sequence_results, key=lambda x: x['JF'], reverse=True)[:5]

    # Aggregate J&F mean
    mean_j = sum(r['J'] for r in sequence_results) / len(sequence_results) if sequence_results else 0
    mean_f = sum(r['F'] for r in sequence_results) / len(sequence_results) if sequence_results else 0
    mean_jf = sum(r['JF'] for r in sequence_results) / len(sequence_results) if sequence_results else 0

    # Determine save directory
    output_subdir = cfg.get("output_subdir", "")
    save_dir = os.path.join("eval_results", output_subdir)
    os.makedirs(save_dir, exist_ok=True)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_file = os.path.join(save_dir, f"metrics_{timestamp}.json")
    manifest_file = os.path.join(save_dir, f"top_5_manifest_{timestamp}.json")
    fps_file = os.path.join(save_dir, f"fps_{timestamp}.json")

    with open(metric_file, "w") as f:
        json.dump(sequence_results, f, indent=4)

    with open(manifest_file, "w") as f:
        json.dump(top_5, f, indent=4)

    with open(fps_file, "w") as f:
        json.dump(fps_info, f, indent=4)

    print(f"\nEvaluation Results saved to {metric_file}")
    print(f"Top 5 Manifest saved to {manifest_file}")
    print(f"FPS info saved to {fps_file}")

    print(f"\n--- Overall Metrics ({len(sequence_results)} sequences) ---")
    print(f"  Mean J   : {mean_j:.4f}")
    print(f"  Mean F   : {mean_f:.4f}")
    print(f"  Mean J&F : {mean_jf:.4f}")

    print("\n--- Top 5 Performers ---")
    for i, res in enumerate(top_5):
        print(f"{i+1}. {res['seq']} | J: {res['J']:.4f} | F: {res['F']:.4f} | J&F: {res['JF']:.4f}")

if __name__ == "__main__":
    main()
