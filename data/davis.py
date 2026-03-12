import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision.transforms import functional as F
import torchvision.transforms as T
import random

class DAVISDataset(Dataset):
    """
    Dataset for loading DAVIS 2017 high-resolution frames and masks.
    """
    def __init__(self, root_dir, image_set="train", seq_len=16, img_size=224, resolution="480p", augment=False):
        self.root_dir = root_dir
        self.image_set = image_set # "train", "val", or "test-dev"
        self.seq_len = seq_len
        self.img_size = img_size
        self.resolution = resolution
        self.augment = augment
        
        self.img_dir = os.path.join(root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(root_dir, 'Annotations', resolution)
        self.split_file = os.path.join(root_dir, 'ImageSets', '2017', f'{image_set}.txt')
        
        if not os.path.exists(self.split_file):
            print(f"Warning: Cannot find split file {self.split_file}. Ignoring if not needed.")
            self.sequences = []
        else:
            with open(self.split_file, 'r') as f:
                self.sequences = [line.strip() for line in f.readlines() if line.strip()]
            
        # Group frames by sequence
        self.sequence_frames = {}
        for seq in self.sequences:
            seq_img_dir = os.path.join(self.img_dir, seq)
            if os.path.isdir(seq_img_dir):
                frames = sorted([f for f in os.listdir(seq_img_dir) if f.endswith('.jpg')])
                self.sequence_frames[seq] = frames
                
        # Create a list of all valid fixed-length clips we can extract
        self.clips = []
        for seq, frames in self.sequence_frames.items():
            if len(frames) >= self.seq_len:
                # We can sample multiple overlapping clips from long sequences
                step = self.seq_len // 2 if image_set == "train" else self.seq_len
                for i in range(0, len(frames) - self.seq_len + 1, step):
                    self.clips.append({
                        'seq': seq,
                        'frames': frames[i:i+self.seq_len]
                    })
            else:
                # If sequence is shorter than seq_len, we will duplicate the last frame during loading
                self.clips.append({
                    'seq': seq,
                    'frames': frames
                })

    def __len__(self):
        return len(self.clips)

    def _transform_clip(self, images, masks):
        # Apply the SAME spatial transform to all frames and masks in the clip
        transformed_images = []
        transformed_masks = []

        # Sample augmentation params once per clip so all frames are transformed consistently
        do_hflip = self.augment and random.random() < 0.5
        do_crop = self.augment and random.random() < 0.5
        if do_crop:
            # Random resized crop: scale 0.7-1.0, ratio 3/4-4/3
            scale = random.uniform(0.7, 1.0)
            ratio = random.uniform(0.75, 1.333)
            base_w, base_h = images[0].size
            crop_h = int(base_h * scale)
            crop_w = int(min(base_w, crop_h * ratio))
            crop_w = max(crop_w, 1)
            crop_h = max(crop_h, 1)
            crop_i = random.randint(0, max(0, base_h - crop_h))
            crop_j = random.randint(0, max(0, base_w - crop_w))
        # Color jitter applied per-image (masks not affected)
        color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05) if self.augment else None

        for img, mask in zip(images, masks):
            if do_crop:
                img = F.crop(img, crop_i, crop_j, crop_h, crop_w)
                mask = F.crop(mask, crop_i, crop_j, crop_h, crop_w)
            if do_hflip:
                img = F.hflip(img)
                mask = F.hflip(mask)
            img = F.resize(img, (self.img_size, self.img_size))
            mask = F.resize(mask, (self.img_size, self.img_size), interpolation=F.InterpolationMode.NEAREST)
            if color_jitter is not None:
                img = color_jitter(img)
            img_tensor = F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Preserve semantic integer IDs (0=bg, 1=obj1, 2=obj2)
            mask_tensor = torch.from_numpy(np.array(mask)).long()

            transformed_images.append(img_tensor)
            transformed_masks.append(mask_tensor)

        return torch.stack(transformed_images), torch.stack(transformed_masks)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        seq = clip_info['seq']
        frames = clip_info['frames']
        
        # Handle padding if sequence was shorter than seq_len
        if len(frames) < self.seq_len:
            pad_len = self.seq_len - len(frames)
            frames = frames + [frames[-1]] * pad_len
            
        loaded_images = []
        loaded_masks = []
        
        # 1. Load the absolute first frame of the sequence as the reference
        ref_frame = self.sequence_frames[seq][0]
        ref_img_path = os.path.join(self.img_dir, seq, ref_frame)
        ref_mask_path = os.path.join(self.mask_dir, seq, ref_frame.replace('.jpg', '.png'))
        
        loaded_images.append(Image.open(ref_img_path).convert('RGB'))
        # If mask doesn't exist (e.g., test-dev queries), create empty mask
        if os.path.exists(ref_mask_path):
            loaded_masks.append(Image.open(ref_mask_path))
        else:
            w, h = loaded_images[-1].size
            loaded_masks.append(Image.new('L', (w, h), 0))
        
        # 2. Load the query frames
        for frame in frames:
            img_path = os.path.join(self.img_dir, seq, frame)
            mask_path = os.path.join(self.mask_dir, seq, frame.replace('.jpg', '.png'))
            
            loaded_images.append(Image.open(img_path).convert('RGB'))
            if os.path.exists(mask_path):
                loaded_masks.append(Image.open(mask_path))
            else:
                w, h = loaded_images[-1].size
                loaded_masks.append(Image.new('L', (w, h), 0))
            
        images, masks = self._transform_clip(loaded_images, loaded_masks)
        
        ref_img, ref_mask = images[0], masks[0]
        query_images, query_masks = images[1:], masks[1:]
        
        return ref_img, ref_mask, query_images, query_masks


class DAVISDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 2, num_workers: int = 4,
                 seq_len: int = 16, img_size: int = 224, test_split: str = "val",
                 augment_train: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.img_size = img_size
        self.test_split = test_split
        self.augment_train = augment_train
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DAVISDataset(
                self.data_dir, image_set="train", seq_len=self.seq_len, img_size=self.img_size,
                augment=self.augment_train)
            self.val_dataset = DAVISDataset(
                self.data_dir, image_set="val", seq_len=self.seq_len, img_size=self.img_size)
        
        if stage == "test" or stage is None:
            self.test_dataset = DAVISDataset(
                self.data_dir, image_set=self.test_split, seq_len=self.seq_len, img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)
