"""
Configuration module for LTX video generation model with HuggingFace Hub integration.
"""

from typing import Optional, Union
from pathlib import Path
import os
import torch
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator
from huggingface_hub import snapshot_download
from loguru import logger

class LTXVideoSettings(BaseSettings):
    """
    Configuration settings for LTX video generation model.
    """
    
    # Model Settings
    model_id: str = Field(
        default="Lightricks/LTX-Video",
        description="HuggingFace model ID"
    )
    ckpt_dir: Path = Field(
        default_factory=lambda: Path(os.getenv('LTX_CKPT_DIR', 'checkpoints')),
        description="Directory containing model checkpoints"
    )
    use_auth_token: Optional[str] = Field(
        default=None,
        description="HuggingFace auth token for private models"
    )
    
    
    input_video_path: Optional[Path] = Field(None, description="Path to input video file")
    input_image_path: Optional[Path] = Field(None, description="Path to input image file")
    output_path: Optional[Path] = Field(
        default_factory=lambda: Path("outputs"),
        description="Path to save output files"
    )
    
    # Generation Settings
    seed: int = Field(171198, description="Random seed for generation")
    num_inference_steps: int = Field(40, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(3.0, ge=1.0, le=20.0, description="Guidance scale")
    
    # Video Parameters
    height: int = Field(480, ge=256, le=720, description="Height of output video frames")
    width: int = Field(704, ge=256, le=1280, description="Width of output video frames")
    num_frames: int = Field(121, ge=1, le=257, description="Number of frames to generate")
    frame_rate: int = Field(25, ge=1, le=60, description="Frame rate of output video")
    num_images_per_prompt: int = Field(
        1,
        ge=1,
        le=4,
        description="Number of videos to generate per prompt"
    )
    
    # Model Settings
    bfloat16: bool = Field(False, description="Use bfloat16 precision")
    device: str = Field("cuda", description="Device to run inference on")
    
    # Prompt Settings
    prompt: Optional[str] = Field(None, description="Text prompt for generation")
    negative_prompt: str = Field(
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt for undesired features"
    )
    
    # Constants
    MAX_HEIGHT: int = 720
    MAX_WIDTH: int = 1280
    MAX_NUM_FRAMES: int = 257

    def download_model(self) -> Path:
        """
        Download model from HuggingFace Hub if not already present.
        
        Returns:
            Path: Path to the model checkpoint directory
        """
        try:
            logger.info(f"Checking for model in {self.ckpt_dir}")
            
            # Check if model files already exist
            if self._verify_model_files():
                logger.info("Model files already present")
                return self.ckpt_dir
            
            # Create checkpoint directory if it doesn't exist
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model from HuggingFace
            logger.info(f"Downloading model {self.model_id} to {self.ckpt_dir}")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.ckpt_dir,
                local_dir_use_symlinks=False,
                repo_type='model',
                token=self.use_auth_token
            )
            
            # Verify downloaded files
            if not self._verify_model_files():
                raise ValueError("Model download appears incomplete. Please check the files.")
            
            logger.info("Model downloaded successfully")
            return self.ckpt_dir
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise

    def _verify_model_files(self) -> bool:
        """
        Verify that all required model files are present.
        
        Returns:
            bool: True if all required files are present
        """
        required_dirs = ['unet', 'vae', 'scheduler']
        required_files = {
            'unet': ['config.json', 'unet_diffusion_pytorch_model.safetensors'],
            'vae': ['config.json', 'vae_diffusion_pytorch_model.safetensors'],
            'scheduler': ['scheduler_config.json']
        }
        
        try:
            # Check for required directories
            for dir_name in required_dirs:
                dir_path = self.ckpt_dir / dir_name
                if not dir_path.is_dir():
                    return False
                
                # Check for required files in each directory
                for file_name in required_files[dir_name]:
                    if not (dir_path / file_name).is_file():
                        return False
            
            return True
            
        except Exception:
            return False

    @field_validator("ckpt_dir")
    @classmethod
    def validate_ckpt_dir(cls, v: Path) -> Path:
        """Convert checkpoint directory to Path and create if needed."""
        return Path(v)

    # Other validators remain the same...

    class Config:
        """Pydantic configuration."""
        env_prefix = "LTX_"
        arbitrary_types_allowed = True
        validate_assignment = True

    def get_model_paths(self) -> tuple[Path, Path, Path]:
        """Get paths to model components after ensuring model is downloaded."""
        # Ensure model is downloaded
        self.download_model()
        
        unet_dir = self.ckpt_dir / "unet"
        vae_dir = self.ckpt_dir / "vae"
        scheduler_dir = self.ckpt_dir / "scheduler"
        
        return unet_dir, vae_dir, scheduler_dir
    
    def get_output_resolution(self) -> tuple[int, int]:
        """Get the output resolution as a tuple of (height, width)."""
        return (self.height, self.width)
    
    def get_padded_num_frames(self) -> int:
        """
        Calculate the padded number of frames.
        Ensures the number of frames is compatible with model requirements.
        """
        # Common video models often require frame counts to be multiples of 8
        FRAME_PADDING = 8
        
        # Calculate padding needed to reach next multiple of FRAME_PADDING
        remainder = self.num_frames % FRAME_PADDING
        if remainder == 0:
            return self.num_frames
            
        padding_needed = FRAME_PADDING - remainder
        return self.num_frames + padding_needed
    