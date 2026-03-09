from transformers import HieraModel
import torch

print("Attempting to download facebook/hiera-base-plus-224...")
model = HieraModel.from_pretrained("facebook/hiera-base-plus-224")
print("Successfully downloaded/loaded facebook/hiera-base-plus-224")
