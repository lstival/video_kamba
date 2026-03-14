import os
import torch
import lightning as L
import hydra
from omegaconf import DictConfig
from models.video_mamba import VideoMambaSystem

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set matmul precision for A100 Tensor Core optimization
    torch.set_float32_matmul_precision("medium")

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    # Initialize logger
    logger = hydra.utils.instantiate(cfg.logger) if "logger" in cfg else None
    
    # Initialize trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=[] # We will instantiate callbacks here later
    )
    
    # Initialize datamodule (Placeholder for now)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # Initialize model
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # We use strict=False because some heads (like VOS) might be new 
        # compared to pre-training, or have different class counts.
        model: L.LightningModule = VideoMambaSystem.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            **cfg.model
        )
    else:
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Optional: Automatically find maximum batch size
    if cfg.get("auto_batch_size", False):
        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(trainer)
        # This automatically modifies model.hparams.batch_size or datamodule.batch_size
        tuner.scale_batch_size(model, datamodule=datamodule, mode="binsearch")
        print(f"Auto batch size found. Starting training...")
        
    # Train the model
    # If checkpoint is provided, we can either use it to resume training 
    # (including optimizer state) or just as weight initialization.
    # For fine-tuning on a different dataset, we usually want weight init only.
    trainer.fit(model=model, datamodule=datamodule)
    
    print("Project initialized successfully.")

if __name__ == "__main__":
    main()
