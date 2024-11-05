import torch 
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("feizhengcong/mochi-1-preview-diffusers", torch_dtype=torch.bfloat16)


pipe.to('cuda')
pipe.enable_vae_tiling()
frames = pipe("A camera follows a squirrel running around on a tree branch", num_inference_steps=100, guidance_scale=4.5, height=480, width=848, num_frames=161).frames[0]
export_to_video(frames, "mochi.mp4", fps=15)