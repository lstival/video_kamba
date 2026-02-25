import torch
import lightning as L
from configs.datamodule.default import VideoDataModule
from models.video_mamba import VideoMambaSystem

def main():
    print("Initiating evaluation...")
    
    # Placeholder for loading from checkpoint
    # model = VideoMambaSystem.load_from_checkpoint("path/to/checkpoint.ckpt")
    
    # Initialize un-trained model for dry un
    model = VideoMambaSystem()
    model.eval()
    
    dm = VideoDataModule()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    trainer = L.Trainer(accelerator='cpu', devices=1, logger=False)
    
    # Run test
    trainer.test(model, dataloaders=test_loader)
    
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
