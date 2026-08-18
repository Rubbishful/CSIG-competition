# save_model.py
import torch
from diffusers import DiffusionPipeline

save_dir = "./stable-diffusion-2-1-base-local"  # 本地保存目录

pipe = DiffusionPipeline.from_pretrained(
    "Manojb/stable-diffusion-2-1-base",
    torch_dtype=torch.bfloat16,  # 原脚本里的 dtype 参数名是无效的，diffusers 用 torch_dtype
)

# 保存全部组件（unet / vae / text_encoder / tokenizer / scheduler / feature_extractor / safety_checker）
pipe.save_pretrained(save_dir, safe_serialization=True)

print(f"模型已保存到: {save_dir}")
