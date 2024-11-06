import torch 
from diffusers import MochiPipeline ,MochiTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi
from diffusers.utils import export_to_video


transformer = MochiTransformer3DModel.from_pretrained("imnotednamode/mochi-1-preview-mix-nf4", torch_dtype=torch.bfloat16)
transformer = torch.compile(transformer,mode = 'max_autotune')
pipe = MochiPipeline.from_pretrained('/home/user/minimochi/models/diffusers_models',torch_dtype=torch.bfloat16, transformer=transformer)



pipe.to('cuda')
pipe.enable_vae_tiling()
frames = pipe("A camera follows a squirrel running around on a tree branch", num_inference_steps=100, guidance_scale=4.5, height=480, width=848, num_frames=161).frames[0]
export_to_video(frames, "mochi.mp4", fps=15)