"""
Hunyuan Video Generation Engine

A high-performance implementation of the Hunyuan video generation model.
Provides enterprise-grade video generation capabilities with comprehensive
resource management and error handling.

Core Features:
    - Text-to-video generation using Hunyuan model
    - 8-bit quantization support for memory efficiency
    - Configurable generation parameters
    - GPU memory management
    - Comprehensive error handling

Technical Specifications:
    - Model: Hunyuan Video Transformer
    - Quantization: 8-bit support
    - Output Format: MP4
    - Device Support: CPU and GPU (CUDA)
    - Memory Optimization: Device map balancing
"""

import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
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
    """Enterprise-grade Hunyuan Video Generation engine.
    
    This class provides a robust implementation of video generation using the
    Hunyuan model with comprehensive error handling, resource management,
    and performance optimization.
    
    Features:
        - Configurable model parameters
        - Memory-efficient quantization
        - Automatic device mapping
        - Resource cleanup
        - Detailed error reporting
    
    Attributes:
        config (HunyuanConfig): Configuration instance
        pipeline (HunyuanVideoPipeline): Loaded model pipeline
        
    Example:
        >>> config = HunyuanConfig(device_map="cuda")
        >>> generator = HunyuanVideoInference(config)
        >>> video_path = generator.generate_video(
        ...     prompt="A cat walks on the grass",
        ...     num_frames=30
        ... )
    """
    
    def __init__(self, config: Optional[HunyuanConfig] = None):
        """Initialize the Hunyuan Video generation engine.
        
        Args:
            config: Configuration for the inference pipeline.
                If None, default configuration will be used.
                
        Raises:
            RuntimeError: If initialization fails
            ValueError: If configuration is invalid
        """
        try:
            self.config = config or HunyuanConfig()
            self._validate_config()
            self.setup_pipeline()
            logger.info("Hunyuan Video engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Hunyuan Video engine: {str(e)}")
            raise RuntimeError(f"Initialization failed: {str(e)}") from e
    
    def _validate_config(self) -> None:
        """Validate the configuration settings.
        
        Ensures all configuration parameters are valid and compatible.
        Creates necessary directories and validates paths.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate output directory
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Invalid output directory: {str(e)}")
        
        # Validate device settings
        if self.config.device_map not in ["auto", "balanced", "sequential", "cpu"]:
            raise ValueError(f"Invalid device_map: {self.config.device_map}")
        
        # Validate torch dtype
        if not hasattr(torch, self.config.torch_dtype):
            raise ValueError(f"Invalid torch_dtype: {self.config.torch_dtype}")
    
    def setup_pipeline(self) -> None:
        """Set up the Hunyuan Video pipeline.
        
        Initializes the model with memory optimization and proper device mapping.
        Handles quantization configuration and model loading.
        
        Raises:
            RuntimeError: If pipeline setup fails
        """
        try:
            logger.info("Setting up Hunyuan pipeline with quantization")
            quant_config = DiffusersBitsAndBytesConfig(
                load_in_8bit=self.config.load_in_8bit
            )
            
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            logger.info("Loading transformer model")
            transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
                self.config.model_id,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch_dtype,
            )
            
            logger.info("Creating pipeline")
            self.pipeline = HunyuanVideoPipeline.from_pretrained(
                self.config.model_id,
                transformer=transformer_8bit,
                torch_dtype=torch_dtype,
                device_map=self.config.device_map,
            )
            
            logger.info("Pipeline setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to set up pipeline: {str(e)}")
            raise RuntimeError(f"Pipeline setup failed: {str(e)}") from e
    
    def generate_video(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        num_frames: int = 61,
        num_inference_steps: int = 30,
        fps: Optional[int] = None,
    ) -> str:
        """Generate a video based on the given prompt.
        
        This method handles the complete video generation process with
        comprehensive error handling and resource management.
        
        Args:
            prompt: Text description of the video to generate
            output_path: Path to save the video. If None, uses timestamped
                filename in configured output directory
            num_frames: Number of frames to generate (default: 61)
            num_inference_steps: Number of denoising steps (default: 30)
            fps: Frames per second for output video. If None, uses
                configured default_fps
                
        Returns:
            str: Path to the generated video file
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If video generation fails
            
        Example:
            >>> generator = HunyuanVideoInference()
            >>> video_path = generator.generate_video(
            ...     prompt="A cat walks on the grass",
            ...     num_frames=30,
            ...     fps=30
            ... )
        """
        start_time = time.time()
        
        try:
            # Parameter validation
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            if num_frames < 1:
                raise ValueError("num_frames must be positive")
            if num_inference_steps < 1:
                raise ValueError("num_inference_steps must be positive")
            
            # Setup output path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    self.config.output_dir,
                    f"video_{timestamp}.mp4"
                )
            
            # Generate video
            logger.info(f"Generating video for prompt: {prompt}")
            logger.info(f"Parameters: frames={num_frames}, steps={num_inference_steps}")
            
            video = self.pipeline(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps
            ).frames[0]
            
            # Export video
            logger.info(f"Exporting video to: {output_path}")
            export_to_video(
                video,
                output_path,
                fps=fps or self.config.default_fps
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Video generation completed in {generation_time:.2f}s")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise RuntimeError(f"Video generation failed: {str(e)}") from e


def main():
    """Example usage of the HunyuanVideoInference class.
    
    Demonstrates basic video generation with error handling.
    """
    try:
        # Initialize with default config
        config = HunyuanConfig()
        generator = HunyuanVideoInference(config)
        
        # Generate a sample video
        prompt = "A cat walks on the grass, realistic style."
        video_path = generator.generate_video(prompt)
        print(f"Generated video saved at: {video_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()