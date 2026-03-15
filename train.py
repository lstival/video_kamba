import os
import torch
import lightning as L
import hydra
from omegaconf import DictConfig
from models.video_mamba import VideoMambaSystem


def _instantiate_callbacks(cfg: DictConfig) -> list:
    """Instantiate callbacks from Hydra config.

    Supports a single callback config (`_target_` at root) or a mapping/list
    of callback configs.
    """
    if cfg is None:
        return []

    callbacks = []

    if isinstance(cfg, DictConfig) and "_target_" in cfg:
        callbacks.append(hydra.utils.instantiate(cfg))
        return callbacks

    if isinstance(cfg, DictConfig):
        for cb_cfg in cfg.values():
            if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))
        return callbacks

    if isinstance(cfg, (list, tuple)):
        for cb_cfg in cfg:
            if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))

    return callbacks

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set matmul precision for A100 Tensor Core optimization
    torch.set_float32_matmul_precision("medium")

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    # Initialize logger
    logger = hydra.utils.instantiate(cfg.logger) if "logger" in cfg else None

    # Initialize callbacks from config (checkpointing, LR monitor, etc.)
    callbacks = _instantiate_callbacks(cfg.get("callbacks"))
    
    # Initialize trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )
    
    # Initialize datamodule (Placeholder for now)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # Initialize model
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # Instantiate model from config first, then do a shape-filtered weight
        # transfer.  This is necessary when the checkpoint was trained with a
        # different SSM / KAN hyperparameter set (e.g. different N or grid size)
        # — strict=False alone still raises RuntimeError on shape mismatches.
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt_sd   = ckpt["state_dict"]
        model_sd  = model.state_dict()
        matched, skipped_shape, skipped_missing = [], [], []
        filtered_sd = {}
        for k, v in ckpt_sd.items():
            if k not in model_sd:
                skipped_missing.append(k)
            elif v.shape != model_sd[k].shape:
                skipped_shape.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
            else:
                filtered_sd[k] = v
                matched.append(k)
        model.load_state_dict(filtered_sd, strict=False)
        print(f"[checkpoint] Loaded {len(matched)} matching tensors.")
        if skipped_shape:
            print(f"[checkpoint] Skipped {len(skipped_shape)} shape-mismatched tensors "
                  f"(will init from scratch):")
            for s in skipped_shape[:10]:
                print(f"  {s}")
            if len(skipped_shape) > 10:
                print(f"  ... and {len(skipped_shape) - 10} more")
        if skipped_missing:
            print(f"[checkpoint] Skipped {len(skipped_missing)} keys not in current model.")
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
