#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Smoke test for pixel matching
python train.py \
    datamodule=davis \
    trainer.fast_dev_run=true \
    trainer.accelerator=cpu \
    trainer.devices=1 \
    model.learning_rate=1e-4 \
    model.propagation_mode=pixel_matching \
    +model.use_ref_context=True \
    model.fusion_mode=kan_spatial \
    logger=none
