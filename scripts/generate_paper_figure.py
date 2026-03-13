import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageChops

def extract_gif_frame(gif_path, frame_idx):
    """Extract a specific frame from a GIF."""
    with Image.open(gif_path) as img:
        img.seek(frame_idx)
        return img.convert("RGB")

def create_overlay(image, mask, color=(0, 255, 0), alpha=128):
    """Create a semi-transparent colored overlay on an image using a mask."""
    # Convert mask to L (grayscale) if it's palette-based
    mask_l = mask.convert("L")
    # Threshold mask to binary (0 and 255)
    mask_binary = mask_l.point(lambda p: 255 if p > 0 else 0)
    
    # Create colored overlay image
    overlay = Image.new("RGBA", image.size, color + (alpha,))
    
    # Base image in RGBA
    img_rgba = image.convert("RGBA")
    
    # Composite the overlay onto the image using the mask
    composite = Image.composite(overlay, img_rgba, mask_binary)
    return composite.convert("RGB")

def create_figure(args):
    # Parameters for the figure
    n_cols = 5  # Frame t, t+5, t+10, GT t+5, GT t+10
    cell_w, cell_h = 480, 480
    padding = 10
    label_h = 50
    label_margin = 200
    
    # Initialize Canvas
    total_w = cell_w * n_cols + label_margin + padding * (n_cols + 2)
    total_h = cell_h * 2 + label_h + padding * 3
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    
    if args.individual_dir:
        os.makedirs(args.individual_dir, exist_ok=True)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
        large_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
        large_font = ImageFont.load_default()

    # Column Headers
    headers = ["Pred Start", "Pred Mid", "Pred End", "GT Mid", "GT End"]
    for i, h in enumerate(headers):
        x = i * (cell_w + padding) + padding + label_margin
        draw.text((x + 20, 10), h, fill="black", font=font)

    # Sequence Mapping
    # eval visual folder name -> {dataset_type, dataset_name}
    seq_mapping = {
        "car-roundabout": {"type": "DAVIS", "name": "car-roundabout"},
        "34564d26d8": {"type": "YouTubeVOS", "name": "34564d26d8"} # Person on bicycle
    }

    def paste_row(seq_folder, display_name, y_offset):
        seq_info = seq_mapping.get(seq_folder)
        if not seq_info:
            print(f"Warning: No mapping for sequence {seq_folder}")
            return

        # Row Label
        draw.text((padding, y_offset + cell_h//2 - 20), display_name, fill="black", font=font)
        
        # 1. Prediction mapping
        visual_path = os.path.join(args.davis_visuals if seq_info['type'] == 'DAVIS' else args.ytb_visuals, seq_folder)
        gifs = [f for f in os.listdir(visual_path) if f.endswith('.gif')]
        if not gifs:
            print(f"No prediction GIF found in {visual_path}")
            return
        gif_path = os.path.join(visual_path, gifs[0])
        
        # Original dataset paths
        if seq_info['type'] == 'DAVIS':
            img_root = os.path.join(args.data_root, "DAVIS/DAVIS/JPEGImages/480p", seq_info['name'])
            ann_root = os.path.join(args.data_root, "DAVIS/DAVIS/Annotations/480p", seq_info['name'])
        else:
            img_root = os.path.join(args.data_root, "YouTubeVOS/valid/JPEGImages", seq_info['name'])
            ann_root = os.path.join(args.data_root, "YouTubeVOS/valid/Annotations", seq_info['name'])

        dataset_img_files = sorted(os.listdir(img_root))
        dataset_ann_files = sorted(os.listdir(ann_root))
        n_dataset = len(dataset_img_files)

        with Image.open(gif_path) as img:
            n_gif_frames = img.n_frames
        
        # Selection: Start, Middle, End
        gif_indices = [0, n_gif_frames // 2, n_gif_frames - 1]
        
        for i, gif_idx in enumerate(gif_indices):
            # Mapping logic: align GIF frame to dataset frame
            if seq_info['type'] == 'DAVIS':
                # Clip 23 for car-roundabout starts at a specific index. Let's use 16 as an approximation 
                # or just use the gif frame + offset if we know it.
                # Actually, `clip_idx` 23 in the dataset means `i = 23 * step`. But it depends on the total clips across ALL videos.
                # We'll use the known offset for this specific sequence's top clip.
                # Here we use an empirical offset for this sequence.
                # Because DAVIS is consecutive frames, we map directly to an offset
                offset = 16  # Top clip for car-roundabout starts here
                orig_idx = offset + gif_idx
            else:
                # YouTubeVOS uses a stride sample starting from index 1
                stride = max(1, (n_dataset - 1) // 5)
                # The clip has 5 query frames.
                # gif_idx maps directly to the queries.
                orig_idx = 1 + gif_idx * stride

            orig_idx = min(orig_idx, n_dataset - 1)
            orig_frame_name = os.path.splitext(dataset_img_files[orig_idx])[0]

            # Paste Prediction
            frame = extract_gif_frame(gif_path, gif_idx)
            frame_resized = frame.resize((cell_w, cell_h))
            canvas.paste(frame_resized, (i * (cell_w + padding) + padding + label_margin, y_offset))
            # Label with original frame number
            draw.text((i * (cell_w + padding) + padding + label_margin + 5, y_offset + 5), f"t+{orig_frame_name}", fill="white", font=font)
            
            if args.individual_dir:
                frame_resized.save(os.path.join(args.individual_dir, f"{seq_folder}_pred_v{orig_frame_name}.png"))

            # 2. Paste Ground Truth (for middle and end columns)
            if i > 0:
                img_path = os.path.join(img_root, dataset_img_files[orig_idx])
                ann_path = os.path.join(ann_root, dataset_ann_files[orig_idx])
                
                with Image.open(img_path) as raw_img:
                    with Image.open(ann_path) as mask:
                        gt_overlay = create_overlay(raw_img, mask, color=(0, 255, 0))
                        gt_resized = gt_overlay.resize((cell_w, cell_h))
                        canvas.paste(gt_resized, ((i+2) * (cell_w + padding) + padding + label_margin, y_offset))
                        
                        if args.individual_dir:
                            gt_resized.save(os.path.join(args.individual_dir, f"{seq_folder}_gt_v{orig_frame_name}.png"))

    # Process Rows
    paste_row("car-roundabout", "DAVIS 2017", label_h)
    paste_row("34564d26d8", "YouTube-VOS", label_h + cell_h + padding)

    canvas.save(args.output)
    print(f"Figure saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_visuals", default="eval_results/top_5_visuals_20260305_092529")
    parser.add_argument("--ytb_visuals", default="eval_results/top_5_visuals_20260305_092739")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output", default="docs/automated_figure.png")
    parser.add_argument("--individual_dir", default="docs/individual_figures")
    args = parser.parse_args()
    
    create_figure(args)
