import torch
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
from datetime import datetime
from models.video_mamba import VideoMambaSystem

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
    
    # Load model from checkpoint
    model = VideoMambaSystem.load_from_checkpoint(checkpoint_path)
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
    results = trainer.test(model, dataloaders=test_loader)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("eval_results", exist_ok=True)
    metric_file = os.path.join("eval_results", f"metrics_{timestamp}.json")
    
    with open(metric_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation Results saved to {metric_file}:")
    for result in results:
        for k, v in result.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
