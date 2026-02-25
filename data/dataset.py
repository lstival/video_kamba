import torch
from torch.utils.data import Dataset
from typing import Tuple

class VideoDataset(Dataset):
    """
    Generic Video Dataset for classification and per-frame segmentation.
    This class currently yields synthetic data for testing the pipeline.
    """
    def __init__(self, num_samples: int = 100, seq_len: int = 16, 
                 img_size: int = 224, num_classes: int = 10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames: [T, 3, H, W]
            class_label: [] (scalar)
            masks: [T, H, W]
        """
        # [T, C, H, W]
        frames = torch.randn(self.seq_len, 3, self.img_size, self.img_size)
        
        # [1]
        class_label = torch.randint(0, self.num_classes, (1,)).squeeze()
        
        # [T, H, W]
        masks = torch.randint(0, self.num_classes, (self.seq_len, self.img_size, self.img_size))
        
        return frames, class_label, masks
