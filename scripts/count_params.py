from models.video_mamba import VideoMambaSystem
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

m = VideoMambaSystem(
    encoder_type="mobilenetv2",
    dim_in=256,
    ssm_layers=2,
    ssm_d_state=16
)

total_params = count_parameters(m)
local_ssm_params = count_parameters(m.temporal_model_local)
global_ssm_params = count_parameters(m.temporal_model_global)
fusion_params = count_parameters(m.ssm_output_fusion)
backbone_params = count_parameters(m.feature_extractor)
decoder_params = count_parameters(m.seg_decoder)

print(f"Total Trainable Params: {total_params / 1e6:.2f}M")
print(f"Local SSM Params: {local_ssm_params / 1e6:.2f}M")
print(f"Global SSM Params: {global_ssm_params / 1e6:.2f}M")
print(f"SSM Fusion Params: {fusion_params / 1e6:.2f}M")
print(f"Backbone Params: {backbone_params / 1e6:.2f}M")
print(f"Decoder Params: {decoder_params / 1e6:.2f}M")
print(f"Feature Fusion (Adapter) Params: {count_parameters(m.feature_fusion) / 1e6:.2f}M")

