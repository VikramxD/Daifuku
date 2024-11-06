"""
Configuration module for Mochi video generation model.
"""

from typing import Optional
from pydantic_settings import BaseSettings
import torch
from pathlib import Path

class MochiSettings(BaseSettings):
    """
    Configuration settings for Mochi video generation model.
    
    All settings can be overridden via environment variables using uppercase names:
    e.g., TRANSFORMER_PATH="path/to/model"
    
    Attributes:
        transformer_path (str): Path or HuggingFace repo ID for the transformer model.
        pipeline_path (str): Path to the diffusers pipeline model.
        dtype (torch.dtype): Torch data type for model precision.
        device (str): Device to run inference on.
        model_name (str): Name of the model for logging/tracking.
        
        # Optimization Settings
        enable_vae_tiling (bool): Enable VAE tiling for memory efficiency.
        enable_model_cpu_offload (bool): Enable CPU offloading for large models.
        enable_attention_slicing (bool): Enable attention slicing for memory efficiency.
        attention_slice_size (Optional[int]): Size of attention slices if enabled.
        
        # Video Generation Settings
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Classifier-free guidance scale.
        height (int): Output video height in pixels.
        width (int): Output video width in pixels.
        num_frames (int): Number of frames to generate.
        fps (int): Frames per second for video export.
    """
    model_name: str = 'Genmo-Mochi'
    transformer_path: str = "imnotednamode/mochi-1-preview-mix-nf4"
    pipeline_path: str = "/home/user/minimochi/models/diffusers_models"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    
    # Optimization Settings
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    attention_slice_size: Optional[int] = None
    
    # Video Generation Settings
    num_inference_steps: int = 100
    guidance_scale: float = 4.5
    height: int = 480
    width: int = 848
    num_frames: int = 161
    fps: int = 15
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        env_prefix = "MOCHI_"  # Environment variables will be prefixed with MOCHI_
    
    def validate_paths(self) -> None:
        """
        Validate model paths and CUDA availability.
        
        Raises:
            RuntimeError: If CUDA is not available when device is set to 'cuda'.
            ValueError: If local pipeline path does not exist.
        """
        if not torch.cuda.is_available() and self.device == "cuda":
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        
       