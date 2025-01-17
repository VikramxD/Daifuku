"""
Hunyuan Video Generation Script

This script provides functionality for generating videos using the Hunyuan AI model.
It handles model initialization, video generation configuration, and output saving.
The script uses Pydantic settings for configuration management, which can be
configured via environment variables or programmatically.

Example:
    Basic usage with default settings:
        >>> from scripts.hunyuan_video_inference import HunyuanVideoInference
        >>> generator = HunyuanVideoInference()
        >>> video_path = generator.generate_video(
        ...     prompt="A cat walks on the grass, realistic style."
        ... )
        >>> print(f"Video saved to: {video_path}")
    
    Usage with custom configuration:
        >>> from configs.hunyuan_config import HunyuanConfig
        >>> config = HunyuanConfig(device_map="cuda", output_dir="custom_outputs")
        >>> generator = HunyuanVideoInference(config)
        >>> video_path = generator.generate_video(
        ...     prompt="A dog playing in the snow",
        ...     num_frames=30,
        ...     fps=30
        ... )

Note:
    This script requires the Hunyuan model to be available either locally or
    downloadable from the Huggingface Hub. The model requires significant
    GPU memory, especially when not using 8-bit quantization.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
from loguru import logger
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    HunyuanVideoTransformer3DModel,
    HunyuanVideoPipeline,
)
from diffusers.utils import export_to_video

from configs.hunyuan_config import HunyuanConfig


class HunyuanVideoInference:
    """Hunyuan Video Generation inference class.
    
    This class provides an interface for generating videos using the Hunyuan model.
    It handles model initialization, pipeline setup, and video generation with
    configurable parameters.
    
    The class uses the HunyuanConfig for configuration management, which can be
    provided during initialization or created with default values.
    
    Attributes:
        config (HunyuanConfig): Configuration instance for the inference pipeline.
        pipeline (HunyuanVideoPipeline): The loaded Hunyuan video generation pipeline.
    
    Example:
        >>> generator = HunyuanVideoInference()
        >>> video_path = generator.generate_video(
        ...     prompt="A cat walks on the grass",
        ...     num_frames=30,
        ...     fps=30
        ... )
        >>> print(f"Generated video: {video_path}")
    """
    
    def __init__(self, config: Optional[HunyuanConfig] = None):
        """Initialize the Hunyuan Video inference pipeline.
        
        Args:
            config (Optional[HunyuanConfig]): Configuration for the inference pipeline.
                If None, default configuration will be used.
        """
        self.config = config or HunyuanConfig()
        self.setup_pipeline()
        
    def setup_pipeline(self):
        """Set up the Hunyuan Video pipeline with the specified configuration.
        
        This method initializes the model and creates the pipeline with the
        configuration specified during class initialization. It handles:
        1. Setting up quantization configuration
        2. Loading the transformer model
        3. Creating the pipeline with the loaded model
        
        The method sets the pipeline attribute of the class, which is then
        used for video generation.
        
        Note:
            This method is automatically called during initialization and typically
            doesn't need to be called directly.
        """
        quant_config = DiffusersBitsAndBytesConfig(
            load_in_8bit=self.config.load_in_8bit
        )
        
        torch_dtype = getattr(torch, self.config.torch_dtype)
        
        transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
            self.config.model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
        )
        
        self.pipeline = HunyuanVideoPipeline.from_pretrained(
            self.config.model_id,
            transformer=transformer_8bit,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
        )
    
    def generate_video(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        num_frames: int = 61,
        num_inference_steps: int = 30,
        fps: Optional[int] = None,
    ) -> str:
        """Generate a video based on the given prompt.
        
        This method handles the entire video generation process, including:
        1. Creating the output directory if needed
        2. Generating the video frames using the Hunyuan pipeline
        3. Exporting the frames to a video file
        
        Args:
            prompt (str): Text prompt describing the video to generate.
            output_path (Optional[str]): Path to save the video. If None,
                a timestamped filename in the configured output directory
                will be used.
            num_frames (int): Number of frames to generate. More frames
                result in longer videos but increase generation time.
                Default: 61
            num_inference_steps (int): Number of denoising steps. Higher
                values may improve quality but increase generation time.
                Default: 30
            fps (Optional[int]): Frames per second for the output video.
                If None, uses the configured default_fps.
                Default: None
            
        Returns:
            str: Path to the generated video file.
            
        Example:
            >>> generator = HunyuanVideoInference()
            >>> path = generator.generate_video(
            ...     prompt="A cat walks on the grass",
            ...     num_frames=30,
            ...     fps=30
            ... )
            >>> print(f"Video saved to: {path}")
        """
        # Create output directory if it doesn't exist
        if output_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.output_dir,
                f"video_{timestamp}.mp4"
            )
        
        # Generate video
        video = self.pipeline(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps
        ).frames[0]
        
        # Export video
        export_to_video(
            video,
            output_path,
            fps=fps or self.config.default_fps
        )
        
        logger.info(f"Video generated and saved to: {output_path}")
        return output_path


def main():
    """Example usage of the HunyuanVideoInference class.
    
    This function demonstrates how to use the HunyuanVideoInference class
    with default settings to generate a simple video.
    """
    config = HunyuanConfig()
    generator = HunyuanVideoInference(config)
    
    prompt = "A cat walks on the grass, realistic style."
    video_path = generator.generate_video(prompt)
    print(f"Generated video saved at: {video_path}")


if __name__ == "__main__":
    main()