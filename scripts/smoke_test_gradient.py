import torch
import torch.nn as nn
from models.video_mamba import VideoMambaSystem

def test_gradient_flow(propagation_mode="soft_mask"):
    print(f"\n--- Smoke Test: Gradient Flow ({propagation_mode}) ---")
    
    # Initialize model with small dimensions for speed
    model = VideoMambaSystem(
        dim_in=768,
        dim_out=256,
        num_seg_classes=2,
        target_size=64,
        propagation_mode=propagation_mode,
        identity_mode="add"
    )
    model.train()
    
    # Dummy data: [B, T, C, H, W]
    B, T, C, H, W = 1, 3, 3, 64, 64
    x = torch.randn(B, T, C, H, W, requires_grad=True)
    ref_frame = torch.randn(B, C, H, W)
    ref_mask = torch.randn(B, H, W).long() % 2
    
    # Target masks for loss computation
    query_masks = torch.randn(B, T, H, W).long() % 2
    obj_present = torch.ones(B, T, 1).bool()
    
    # Forward pass
    logits_clf, pred_boxes, pred_box_logits, logits_seg = model(x, ref_frame=ref_frame, ref_mask=ref_mask)
    
    # Compute loss on the LAST frame to check temporal propagation
    loss = model.vos_loss_fn(logits_seg[:, -1:], query_masks[:, -1:], obj_present[:, -1:])
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # 1. Check if gradients reach the input patches of the FIRST frame
    # (Through the temporal model)
    # query_patch is extracted from Dino features, which are frozen.
    # So we check the temporal model weights.
    s1_grad = model.temporal_model.layers[0].B.grad
    print(f"Temporal Model B-matrix grad norm: {s1_grad.norm().item() if s1_grad is not None else 0:.6f}")
    
    # 2. Check if gradients reach the INITIAL mask embedding
    mask_emb_grad = model.mask_embedding.weight.grad
    print(f"Initial Mask Embedding grad norm: {mask_emb_grad.norm().item() if mask_emb_grad is not None else 0:.6f}")
    
    # 3. Check if gradients reach the feature extractor inputs (if it were unfrozen)
    # (Since Dino is frozen, we check the recursive path)
    
    if s1_grad is not None and s1_grad.norm() > 0:
        print("✅ Gradient is flowing through time!")
    else:
        print("❌ Gradient is DETACHED in temporal propagation!")

    if mask_emb_grad is not None and mask_emb_grad.norm() > 0:
        print("✅ Gradient is reaching initial mask embedding!")
    else:
        print("❌ Gradient is DETACHED from initial mask!")

if __name__ == "__main__":
    test_gradient_flow("soft_mask")
    test_gradient_flow("direct_feature")
