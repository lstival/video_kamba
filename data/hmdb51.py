import os
import glob
import torch
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

class HMDB51Dataset(Dataset):
    """
    Dataset for loading HMDB51 videos for action classification.
    """
    def __init__(self, root_dir, split_mode="train", seq_len=16, frame_step=2, img_size=224, seed=42):
        self.root_dir = root_dir
        self.split_mode = split_mode
        self.seq_len = seq_len
        self.frame_step = frame_step
        self.img_size = img_size
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load all video paths and labels
        self.video_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for vid_name in os.listdir(cls_dir):
                if vid_name.endswith('.avi'):
                    self.video_paths.append(os.path.join(cls_dir, vid_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        # Simple random split for demonstration (70% train, 15% val, 15% test)
        rng = random.Random(seed)
        indices = list(range(len(self.video_paths)))
        rng.shuffle(indices)
        
        num_train = int(len(indices) * 0.7)
        num_val = int(len(indices) * 0.15)
        
        if split_mode == "train":
            split_indices = indices[:num_train]
        elif split_mode == "val":
            split_indices = indices[num_train:num_train+num_val]
        else: # test
            split_indices = indices[num_train+num_val:]
            
        self.video_paths = [self.video_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]
        
        # We will use simple spatial augmentations per frame. 
        # Ideally, we should use a VideoTransform that acts uniformly across time (T, C, H, W).
        # We apply Resize/Crop on the Height/Width dimension.
        self.transform = Compose([
            Resize(img_size),
            CenterCrop(img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Read video using torchvision
        # vframes shape: [T, H, W, C]
        vframes, aframes, info = torchvision.io.read_video(video_path, pts_unit='sec')
        
        total_frames = vframes.shape[0]
        required_span = self.seq_len * self.frame_step
        
        if total_frames > required_span:
            # Randomly sample a clip during training
            start_idx = random.randint(0, total_frames - required_span - 1)
            frame_indices = range(start_idx, start_idx + required_span, self.frame_step)
        else:
            # Pad by repeating or taking what we can
            frame_indices = [i % total_frames for i in range(required_span)]
            frame_indices = frame_indices[::self.frame_step]
            
        # Select frames and reshape to [T, C, H, W]
        frames = vframes[frame_indices]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        
        # Apply transforms
        # transform expects [..., C, H, W], so we can pass the entire [T, C, H, W] batch
        frames = self.transform(frames)
        
        return frames, torch.tensor(label, dtype=torch.long)


class HMDB51DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4, 
                 seq_len: int = 16, img_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.img_size = img_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = HMDB51Dataset(
                self.data_dir, split_mode="train", seq_len=self.seq_len, img_size=self.img_size)
            self.val_dataset = HMDB51Dataset(
                self.data_dir, split_mode="val", seq_len=self.seq_len, img_size=self.img_size)
            
        if stage == "test" or stage is None:
            self.test_dataset = HMDB51Dataset(
                self.data_dir, split_mode="test", seq_len=self.seq_len, img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)
