import torch
from models.components.segmentation_decoder import KANSpatialGatingUpBlock, SegmentationDecoder

block = KANSpatialGatingUpBlock(in_channels=768, skip_channels=768, out_channels=256)
out = block(torch.randn(2, 768, 16, 16), torch.randn(2, 768, 32, 32))
assert out.shape == (2, 256, 32, 32), f"Bad shape: {out.shape}"
print(f"[OK] Block output: {out.shape}")

decoder = SegmentationDecoder(dim_ssm=768, dim_dinov2=768, num_classes=11, target_size=224, fusion_mode="kan_spatial")
dino = {k: torch.randn(1, 2, 768, 256) for k in ["layer_3", "layer_6", "layer_9", "layer_11"]}
logits = decoder(torch.randn(1, 2, 768, 256), dino)
assert logits.shape == (1, 2, 11, 224, 224), f"Bad shape: {logits.shape}"
print(f"[OK] Decoder output: {logits.shape}")

for name, m in decoder.named_modules():
    assert not isinstance(m, torch.nn.MultiheadAttention), f"O(N^2) attn found: {name}"
print("[OK] No MultiheadAttention in kan_spatial decoder path")
print("All checks passed.")
