from transformers import HieraModel
import torch
import sys

# Test candidates
candidates = [
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-base-plus-hf",
    "facebook/hiera-base-plus-224-hf", # Just in case
]

for model_id in candidates:
    print(f"\n--- Testing: {model_id} ---")
    try:
        model = HieraModel.from_pretrained(model_id)
        print(f"SUCCESS: Loaded {model_id}")
        
        # Check hidden states shapes
        inputs = torch.randn(1, 3, 224, 224)
        outputs = model(pixel_values=inputs, output_hidden_states=True)
        rhs = getattr(outputs, "reshaped_hidden_states", None)
        if rhs:
            for i, s in enumerate(rhs):
                print(f"  Stage {i}: {s.shape}")
        else:
            hs = getattr(outputs, "hidden_states", None)
            if hs:
                for i, s in enumerate(hs):
                    print(f"  Hidden {i}: {s.shape}")
        
        # Test passed for this model, we can stop or check others
    except Exception as e:
        print(f"FAILED: {model_id} - {e}")

print("\nDone testing candidates.")
