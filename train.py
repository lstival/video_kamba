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
    
    # Initialize trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=[] # We will instantiate callbacks here later
    )
    
    # Initialize datamodule (Placeholder for now)
    # datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # Initialize model (Placeholder for now)
    # model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Train the model
    # trainer.fit(model=model, datamodule=datamodule)
    
    print("Project initialized successfully.")

if __name__ == "__main__":
    main()
