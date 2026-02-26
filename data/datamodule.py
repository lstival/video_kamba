import lightning as L
from torch.utils.data import DataLoader
from data.dataset import VideoDataset

class VideoDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for video understanding tasks.

    This module handles the 'Data Loading' step of the Training and Evaluation processes.
    It encapsulates the creation of training, validation, and test datasets and their
    corresponding DataLoaders.

    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
        seq_len: Number of frames in a video sequence.
        img_size: Spatial resolution of the frames (H=W=img_size).
        num_classes: Number of classes for the task (classification or segmentation).
    """

    def __init__(self, batch_size: int = 4, num_workers: int = 2, 
                 seq_len: int = 16, img_size: int = 224, num_classes: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_classes = num_classes
        
    def setup(self, stage: str = None):
        """Prepares datasets for different stages (fit/validate/test/predict).

        Part of the 'Data Loading' process step.

        Args:
            stage: The stage to setup ('fit', 'test', null).
        """
        if stage == "fit" or stage is None:
            self.train_dataset = VideoDataset(num_samples=100, seq_len=self.seq_len, 
                                              img_size=self.img_size, num_classes=self.num_classes)
            self.val_dataset = VideoDataset(num_samples=20, seq_len=self.seq_len, 
                                            img_size=self.img_size, num_classes=self.num_classes)
        
        if stage == "test" or stage is None:
            self.test_dataset = VideoDataset(num_samples=20, seq_len=self.seq_len, 
                                             img_size=self.img_size, num_classes=self.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)
