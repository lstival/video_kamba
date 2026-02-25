import torch
import pytest
from models.components.dinov3_wrapper import DinoV3Wrapper
from models.components.kanga_ssm import KangaSSM
from models.video_mamba import VideoMambaSystem

def test_dinov3_wrapper_shapes():
    B, T, C, H, W = 2, 4, 3, 224, 224
    inputs = torch.randn(B, T, C, H, W)
    model = DinoV3Wrapper()
    
    cls_tokens, patch_tokens = model(inputs)
    
    assert cls_tokens.shape == (B, T, 768), f"Expected (2, 4, 768), got {cls_tokens.shape}"
    # P patches for 224x224 with 16x16 conv is 14x14 = 196
    assert patch_tokens.shape == (B, T, 768, 196), f"Expected (2, 4, 768, 196), got {patch_tokens.shape}"

def test_kanga_ssm_shapes():
    B, T, D = 2, 4, 768
    inputs = torch.randn(B, T, D)
    model = KangaSSM(d_model=D)
    
    outputs = model(inputs)
    
    assert outputs.shape == (B, T, D), f"Expected {(B, T, D)}, got {outputs.shape}"

def test_video_mamba_system():
    B, T, C, H, W = 2, 4, 3, 224, 224
    num_classes = 5
    inputs = torch.randn(B, T, C, H, W)
    model = VideoMambaSystem(dim_in=768, num_classes=num_classes, target_size=224)
    
    logits_clf, logits_seg = model(inputs)
    
    assert logits_clf.shape == (B, num_classes)
    assert logits_seg.shape == (B, T, num_classes, 224, 224)

    # test defensive assertions
    assert not torch.isnan(logits_clf).any()
    assert not torch.isnan(logits_seg).any()
