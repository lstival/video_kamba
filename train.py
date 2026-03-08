import comet_ml  # Must be imported before lightning/torch
import os
import lightning as L
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    # Initialize logger
    logger = hydra.utils.instantiate(cfg.logger) if "logger" in cfg else None
    if logger and hasattr(logger, "experiment") and hasattr(logger.experiment, "url"):
        print(f"Comet Logger initialized. Experiment: {logger.experiment.url}")
    
    # Initialize trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=[] # We will instantiate callbacks here later
    )
    
    # Initialize datamodule (Placeholder for now)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # Initialize model (Placeholder for now)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Load weights from checkpoint if provided (for fine-tuning)
    ckpt_path = cfg.get("checkpoint")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading pre-trained weights from: {ckpt_path}")
        import torch
        # Load the state_dict directly to allow fine-tuning on different datasets/configs
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        
        # Filter out keys if needed (e.g. classifier heads if classes change, 
        # but for VOS num_seg_classes should match)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded with results: {msg}")
    
    # Train the model
    trainer.fit(model=model, datamodule=datamodule)
    
    print("Project initialized successfully.")

if __name__ == "__main__":
    main()
