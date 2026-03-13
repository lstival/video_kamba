import torch
import torch.nn as nn
import torch.nn.functional as F
from models.video_mamba import VideoMambaSystem

def check_gradient_flow():
    print("--- Starting Hybrid Flow + Mask Guidance Gradient Check ---")
    
    # 1. Initialize Model
    model = VideoMambaSystem(
        propagation_mode="hybrid_flow",
        identity_mode="add",
        num_seg_classes=2,
        target_size=224,
        use_ref_context=True,
        fusion_mode="kan_spatial"
    )
    
    # 2. Dummy Data
    B, T, C, H, W = 1, 2, 3, 224, 224
    x = torch.randn(B, T, C, H, W)
    ref_frame = torch.randn(B, C, H, W)
    ref_mask = torch.randint(0, 2, (B, H, W)).float()
    query_masks = torch.randint(0, 2, (B, T, H, W)).float()
    
    # 3. Forward Pass
    print("Running Forward Pass...")
    model.train() 
    _, _, _, logits_seg = model(x, ref_frame=ref_frame, ref_mask=ref_mask, query_masks=query_masks)
    print(f"Logits Seg shape: {logits_seg.shape}")
    
    # 4. Backward Pass
    loss = (logits_seg**2).mean()
    print(f"Loss: {loss.item():.6f}")
    loss.backward()
    print("Backward Pass Complete.")
    
    # 5. Gradient Checks
    # Check matching path
    matching_grad = model.pixel_matching_attn.out_proj.weight.grad
    matching_sum = torch.abs(matching_grad).sum().item() if matching_grad is not None else 0
    print(f"Matching Grad Sum: {matching_sum:.6e}")
    
    # Check flow path
    flow_grad = model.flow_ssm.layers[0].B.grad
    flow_sum = torch.abs(flow_grad).sum().item() if flow_grad is not None else 0
    print(f"Flow Grad Sum: {flow_sum:.6e}")
    
    # Check KAN Decoder Gate (specifically the rbf weights)
    # up1.gate_kan.rbf_weight
    kan_grad = model.seg_decoder.up1.gate_kan.rbf_weight.grad
    kan_sum = torch.abs(kan_grad).sum().item() if kan_grad is not None else 0
    print(f"KAN Decoder Gate Grad Sum: {kan_sum:.6e}")
    
    # Check decoder final out
    decoder_grad = model.seg_decoder.up1.out_conv[0].weight.grad
    decoder_sum = torch.abs(decoder_grad).sum().item() if decoder_grad is not None else 0
    print(f"Decoder Out Conv Grad Sum: {decoder_sum:.6e}")
    
    all_ok = all(s > 1e-12 for s in [matching_sum, flow_sum, kan_sum, decoder_sum])
    
    if all_ok:
        print("\n✅ PHASE 3 GRADIENT FLOW VERIFIED: Identity, Flow, and Mask-Guided Decoder are all connected.")
    else:
        print("\n❌ PHASE 3 GRADIENT FLOW FAILED.")
        if matching_sum <= 1e-12: print(" - Matching path detached")
        if flow_sum <= 1e-12: print(" - Flow SSM path detached")
        if kan_sum <= 1e-12: print(" - KAN Gate (Mask guidance) detached")
        if decoder_sum <= 1e-12: print(" - Decoder output detached")

if __name__ == "__main__":
    check_gradient_flow()
