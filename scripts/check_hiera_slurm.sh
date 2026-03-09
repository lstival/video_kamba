#!/bin/bash
#SBATCH --job-name=hiera_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/hiera_check_%j.out
#SBATCH --error=logs/slurm/hiera_check_%j.err

source venv/bin/activate
export PYTHONPATH=.

python -c "
from transformers import HieraModel
import torch

for model_id in ['facebook/hiera-base-224-hf', 'facebook/hiera-base-plus-224-hf']:
    print(f'\n--- Checking: {model_id} ---')
    try:
        model = HieraModel.from_pretrained(model_id)
        print(f'SUCCESS: Loaded {model_id}')
        inputs = torch.randn(1, 3, 224, 224)
        outputs = model(pixel_values=inputs, output_hidden_states=True)
        rhs = getattr(outputs, 'reshaped_hidden_states', None)
        if rhs:
            for i, s in enumerate(rhs):
                print(f'  Stage {i}: {s.shape}')
        else:
            hs = getattr(outputs, 'hidden_states', None)
            for i, s in enumerate(hs):
                 print(f'  Hidden {i}: {s.shape}')
    except Exception as e:
        print(f'FAILED: {model_id} - {e}')
"
