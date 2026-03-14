import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from data.coco_pretrain import COCOPretrainDataset

def save_visualisation(output_dir="vis_coco_pretrain"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO dataset (local cache)
    dataset = COCOPretrainDataset(
        img_size=448,
        seq_len=3,
        n_id=10,
        split="validation",
        cache_dir="./data/coco_cache",
        max_samples=10,
        streaming=False
    )
    
    # Get first valid sample
    iterator = iter(dataset)
    ref_img, ref_mask, q_imgs, q_masks, obj_present, meta = next(iterator)
    
    # helper to denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def to_img(t):
        t = t * std + mean
        return t.permute(1, 2, 0).numpy().clip(0, 1)

    # Color map for masks
    cmap = plt.cm.get_cmap('tab20', 11)
    
    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Ref Frame
    axes[0, 0].imshow(to_img(ref_img))
    axes[0, 0].set_title("COCO Ref (Original)")
    axes[1, 0].imshow(ref_mask.numpy(), cmap=cmap, interpolation='nearest')
    axes[1, 0].set_title("COCO Ref Mask (Rectangles)")
    
    # Query Frames
    for i in range(3):
        axes[0, i+1].imshow(to_img(q_imgs[i]))
        axes[0, i+1].set_title(f"COCO Query {i+1}")
        axes[1, i+1].imshow(q_masks[i].numpy(), cmap=cmap, interpolation='nearest')
        axes[1, i+1].set_title(f"COCO Mask {i+1}")
        
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coco_pretrain_sequence.png"), bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_dir}/coco_pretrain_sequence.png")

if __name__ == "__main__":
    save_visualisation()
