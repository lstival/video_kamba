import torch

ckpt = torch.load(
    "/home/WUR/stiva001/WUR/video_kamba/checkpoints/ssm_mem_v2_aot/best_slim_v2_phase2_bl30k_full.ckpt",
    map_location="cpu",
    weights_only=False,
)
for k, v in ckpt["callbacks"].items():
    if "ModelCheckpoint" in str(k):
        print(f"Callback key : {k}")
        print(f"monitor      : {v.get('monitor')}")
        print(f"best_score   : {v.get('best_model_score')}")
        print(f"best_path    : {v.get('best_model_path')}")
