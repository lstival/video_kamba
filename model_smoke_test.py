import torch
from models.video_mamba import VideoMambaSystem

def test_model_forward():
    print("--- Testing VideoMambaSystem Forward Pass (Patch-Sequence SSM) ---")
    B, T, C, H, W = 2, 8, 3, 224, 224
    dim_in = 768
    num_clf_classes = 51
    num_seg_classes = 11
    
    model = VideoMambaSystem(
        dim_in=dim_in,
        num_clf_classes=num_clf_classes,
        num_seg_classes=num_seg_classes
    )
    model.eval()
    
    # Inputs
    x = torch.randn(B, T, C, H, W)
    ref_frame = torch.randn(B, C, H, W)
    ref_mask = torch.randint(0, num_seg_classes, (B, H, W))
    
    print(f"Input query: {x.shape}")
    print(f"Input ref: {ref_frame.shape}, mask: {ref_mask.shape}")
    
    with torch.no_grad():
        logits_clf, pred_boxes, pred_box_logits, logits_seg = model(x, ref_frame=ref_frame, ref_mask=ref_mask)
        
    print(f"Output logits_clf: {logits_clf.shape} (Expected: [{B}, {T}, {num_clf_classes}])")
    print(f"Output logits_seg: {logits_seg.shape} (Expected: [{B}, {T}, {num_seg_classes}, {H//16}, {W//16}] or similar)")
    
    # Check shapes
    assert logits_clf.shape == (B, T, num_clf_classes)
    # The decoder upsamples to target_size (default 224)
    assert logits_seg.shape == (B, T, num_seg_classes, 224, 224)
    
    print("✅ Model forward pass successful!")

if __name__ == "__main__":
    test_model_forward()
