import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

def save_ytvos_visualisation(output_dir="vis_ytvos"):
    os.makedirs(output_dir, exist_ok=True)
    
    yt_root = "data/YouTubeVOS/train"
    img_dir = os.path.join(yt_root, "JPEGImages")
    ann_dir = os.path.join(yt_root, "Annotations")
    
    # Pick a random video
    videos = os.listdir(img_dir)
    video = random.choice(videos)
    
    v_img_dir = os.path.join(img_dir, video)
    v_ann_dir = os.path.join(ann_dir, video)
    
    # Get frames (sorted)
    frames = sorted([f for f in os.listdir(v_img_dir) if f.endswith('.jpg')])
    
    # Pick 4 frames with some gap to show movement
    if len(frames) < 10: # fallback
        indices = [0, 1, 2, 3]
    else:
        indices = [0, 3, 6, 9] 
    
    selected_frames = [frames[i] for i in indices]
    
    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    cmap = plt.cm.get_cmap('tab20', 11)
    
    for i, f_name in enumerate(selected_frames):
        # Load Image
        img_path = os.path.join(v_img_dir, f_name)
        img = Image.open(img_path).convert("RGB").resize((448, 448))
        
        # Load Mask
        mask_name = f_name.replace('.jpg', '.png')
        mask_path = os.path.join(v_ann_dir, mask_name)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).resize((448, 448), Image.NEAREST)
            mask_np = np.array(mask)
        else:
            mask_np = np.zeros((448, 448))

        axes[0, i].imshow(img)
        axes[1, i].imshow(mask_np, cmap=cmap, interpolation='nearest')
        
        title = "Ref Frame" if i == 0 else f"Video Frame t+{i*3}"
        axes[0, i].set_title(title)

    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ytvos_sequence.png"), bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_dir}/ytvos_sequence.png")

if __name__ == "__main__":
    save_ytvos_visualisation()
