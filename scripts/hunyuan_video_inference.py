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
from hunyuan.hyvideo.utils.file_utils import save_videos_grid
from hunyuan.hyvideo.config import parse_args
from hunyuan.hyvideo.inference import HunyuanVideoSampler
from configs.hunyuan_settings import VideoGenSettings



def initialize_model(model_path: str):
    """
    Initialize the Hunyuan video generation model.
    
    Args:
        model_path (str): Path to the directory containing the model files
        
    Returns:
        HunyuanVideoSampler: Initialized video generation model
        
    Raises:
        ValueError: If the model_path directory doesn't exist
    """
    args = parse_args()
    models_root_path = Path(model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    return hunyuan_video_sampler

def generate_video(
    model,
    settings: VideoGenSettings
):
    """
    Generate a video using the Hunyuan model based on provided settings.
    
    Args:
        model (HunyuanVideoSampler): Initialized Hunyuan video model
        settings (VideoGenSettings): Configuration settings for video generation
        
    Returns:
        str: Path to the generated video file
        
    Notes:
        - The video will be saved in the specified output directory
        - The filename includes timestamp, seed, and truncated prompt
        - Videos are saved as MP4 files with 24 FPS
    """
    seed = None if settings.seed == -1 else settings.seed
    width, height = settings.resolution.split("x")
    width, height = int(width), int(height)
    negative_prompt = "" 

    outputs = model.predict(
        prompt=settings.prompt,
        height=height,
        width=width, 
        video_length=settings.video_length,
        seed=seed,
        negative_prompt=negative_prompt,
        infer_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        num_videos_per_prompt=1,
        flow_shift=settings.flow_shift,
        batch_size=1,
        embedded_guidance_scale=settings.embedded_guidance_scale
    )
    
    samples = outputs['samples']
    sample = samples[0].unsqueeze(0)
    save_path = os.path.join(os.getcwd(), settings.output_dir)
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    video_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][0]}_{outputs['prompts'][0][:100].replace('/','')}.mp4"
    save_videos_grid(sample, video_path, fps=24)
    logger.info(f'Sample saved to: {video_path}')
    
    return video_path

def main():
    """
    Main entry point for the video generation script.
    
    Workflow:
        1. Loads configuration from environment/settings
        2. Initializes the Hunyuan model
        3. Generates the video based on settings
        4. Prints the path to the generated video
    """
    settings = VideoGenSettings()
    model = initialize_model(settings.model_path)
    video_path = generate_video(model=model, settings=settings)
    print(f"Video generated successfully at: {video_path}")

if __name__ == "__main__":
    main()