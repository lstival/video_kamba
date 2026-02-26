import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision.transforms import functional as F
import random

class DAVISDataset(Dataset):
    """
    Dataset for loading DAVIS 2017 high-resolution frames and masks.
    """
    def __init__(self, root_dir, image_set="train", seq_len=16, img_size=224, resolution="480p"):
        self.root_dir = root_dir
        self.image_set = image_set # "train" or "val"
        self.seq_len = seq_len
        self.img_size = img_size
        self.resolution = resolution
        
        self.img_dir = os.path.join(root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(root_dir, 'Annotations', resolution)
        self.split_file = os.path.join(root_dir, 'ImageSets', '2017', f'{image_set}.txt')
        
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Cannot find split file {self.split_file}")
            
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
        # We need to apply the SAME spatial transform to all frames and masks in the clip
        # For simplicity, we just use absolute resize for both
        
        # Determine random crop or similar if it was train, but here we just resize
        transformed_images = []
        transformed_masks = []
        
        for img, mask in zip(images, masks):
            img = F.resize(img, (self.img_size, self.img_size))
            mask = F.resize(mask, (self.img_size, self.img_size), interpolation=F.InterpolationMode.NEAREST)
            
            img_tensor = F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask_tensor = torch.from_numpy(np.array(mask)).long()
            
            # Binary segmentation masks: non-zero is object
            # DAVIS mask values represent integer object IDs (0=background, 1=obj1, 2=obj2...)
            # We turn it into a single classification map or dense binary map for semantic segmentation.
            
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
        
        for frame in frames:
            img_path = os.path.join(self.img_dir, seq, frame)
            mask_path = os.path.join(self.mask_dir, seq, frame.replace('.jpg', '.png'))
            
            loaded_images.append(Image.open(img_path).convert('RGB'))
            loaded_masks.append(Image.open(mask_path))
            
        images, masks = self._transform_clip(loaded_images, loaded_masks)
        
        return images, masks


class DAVISDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 2, num_workers: int = 4, 
                 seq_len: int = 16, img_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.img_size = img_size
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DAVISDataset(
                self.data_dir, image_set="train", seq_len=self.seq_len, img_size=self.img_size)
            self.val_dataset = DAVISDataset(
                self.data_dir, image_set="val", seq_len=self.seq_len, img_size=self.img_size)
        
        if stage == "test" or stage is None:
            self.test_dataset = DAVISDataset(
                self.data_dir, image_set="val", seq_len=self.seq_len, img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)
