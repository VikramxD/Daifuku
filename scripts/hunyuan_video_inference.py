"""
Hunyuan Video Generation Script

This script provides functionality for generating videos using the Hunyuan AI model.
It handles model initialization, video generation configuration, and output saving.

The script can be configured via environment variables (with VIDEO_GEN_ prefix) or a .env file.
"""

import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video


model_id = "hunyuanvideo-community/HunyuanVideo"


quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A cat walks on the grass, realistic style."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "cat.mp4", fps=15)