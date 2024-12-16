"""
Configuration module for LTX video generation model with HuggingFace Hub integration.

This module provides a comprehensive configuration system for the LTX video generation model,
handling model downloads, parameter validation, and settings management. It uses Pydantic
for robust configuration validation and type checking.

Key Features:
    - Automatic model download from HuggingFace Hub
    - Configurable video generation parameters
    - Input/output path management
    - Model checkpoint verification
    - Device and precision settings
    - Prompt configuration for generation

Example:
    >>> settings = LTXVideoSettings(
    ...     model_id="Lightricks/LTX-Video",
    ...     prompt="A beautiful sunset over the ocean",
    ...     num_frames=60
    ... )
    >>> settings.download_model()
    >>> unet_path, vae_path, scheduler_path = settings.get_model_paths()
"""

# Constants
MAX_HEIGHT: int = 720
MAX_WIDTH: int = 1280
MAX_NUM_FRAMES: int = 257

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
    
    This class manages all configuration aspects of the LTX video generation pipeline,
    including model paths, generation parameters, and output settings. It provides
    validation and automatic type conversion for all settings.

    Attributes:
        model_id (str): HuggingFace model identifier (default: "Lightricks/LTX-Video")
        ckpt_dir (Path): Directory for model checkpoints
        use_auth_token (Optional[str]): HuggingFace authentication token
        input_video_path (Optional[Path]): Path to input video file
        input_image_path (Optional[Path]): Path to input image file
        output_path (Optional[Path]): Directory for output files
        seed (int): Random seed for reproducible generation
        num_inference_steps (int): Number of denoising steps (range: 1-100)
        guidance_scale (float): Classifier-free guidance scale (range: 1.0-20.0)
        height (int): Output video height in pixels (range: 256-720)
        width (int): Output video width in pixels (range: 256-1280)
        num_frames (int): Number of frames to generate (range: 1-257)
        frame_rate (int): Output video frame rate (range: 1-60)
        num_images_per_prompt (int): Number of videos per prompt (range: 1-4)
        bfloat16 (bool): Whether to use bfloat16 precision
        device (str): Device for inference ('cuda' or 'cpu')
        prompt (Optional[str]): Generation prompt text
        negative_prompt (str): Negative prompt for undesired features
        model_revision (str): Specific model revision to use for stability

    Example:
        >>> settings = LTXVideoSettings(
        ...     prompt="A serene mountain landscape",
        ...     num_frames=60,
        ...     height=480,
        ...     width=704
        ... )
        >>> settings.download_model()
    """
    
    # Model Settings
    model_id: str = Field(
        default="Lightricks/LTX-Video",
        description="HuggingFace model ID"
    )
    ckpt_dir: Path = Field(
        default_factory=lambda: Path(os.getenv('LTX_CKPT_DIR', '../checkpoints')),
        description="Directory containing model checkpoints"
    )
    use_auth_token: Optional[str] = Field(
        default=None,
        description="HuggingFace auth token for private models"
    )
    
    
    input_video_path: Optional[Path] = Field(None, description="Path to input video file")
    input_image_path: Optional[Path] = Field(None, description="Path to input image file")
    output_path: Optional[Path] = Field(None, description="Path to save output files")
    
    # Generation Settings
    seed: int = Field(171198, description="Random seed for generation")
    num_inference_steps: int = Field(40, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(3.0, ge=1.0, le=20.0, description="Guidance scale")
    
    # Video Parameters
    height: int = Field(
        480,
        ge=256,
        le=MAX_HEIGHT,
        description="Height of output video frames"
    )
    width: int = Field(
        704,
        ge=256,
        le=MAX_WIDTH,
        description="Width of output video frames"
    )
    num_frames: int = Field(
        121,
        ge=1,
        le=MAX_NUM_FRAMES,
        description="Number of frames to generate"
    )
    frame_rate: int = Field(25, ge=1, le=60, description="Frame rate of output video")
    num_images_per_prompt: int = Field(1, ge=1, le=4, description="Number of videos per prompt")
    
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

    # Add model revision to ensure stable downloads
    model_revision: str = Field(
        default="f1994f6731091f828ecc135923c978155928c031",
        description="Specific model revision to use for stability"
    )

    def download_model(self) -> Path:
        """
        Download model from HuggingFace Hub if not already present.
        
        This method checks for existing model files, downloads missing components,
        and verifies the integrity of the downloaded files. It handles authentication
        for private models using the provided token.

        Returns:
            Path: Path to the model checkpoint directory

        Raises:
            ValueError: If model download is incomplete or verification fails
            Exception: If download encounters network or permission errors

        Example:
            >>> settings = LTXVideoSettings()
            >>> model_path = settings.download_model()
            >>> print(f"Model downloaded to {model_path}")
        """
        try:
            logger.info(f"Checking for model in {self.ckpt_dir}")
            
            # Check if model files already exist
            if self._verify_model_files():
                logger.info("Model files already present")
                return self.ckpt_dir
            
            # Create checkpoint directory if it doesn't exist
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model from HuggingFace with specific revision
            logger.info(f"Downloading model {self.model_id} (revision: {self.model_revision}) to {self.ckpt_dir}")
            snapshot_download(
                repo_id=self.model_id,
                revision=self.model_revision,  # Use specific working revision
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
        Verify that all required model files are present in the checkpoint directory.
        
        Checks for the existence of essential model components including the UNet,
        VAE, and scheduler configurations and weights.

        Returns:
            bool: True if all required files are present and accessible

        Note:
            Required directory structure:
            - unet/
                - config.json
                - unet_diffusion_pytorch_model.safetensors
            - vae/
                - config.json
                - vae_diffusion_pytorch_model.safetensors
            - scheduler/
                - scheduler_config.json
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
        """
        Get paths to model components after ensuring model is downloaded.
        
        This method ensures the model is downloaded before returning paths to
        the essential model components.

        Returns:
            tuple[Path, Path, Path]: Paths to (unet_dir, vae_dir, scheduler_dir)

        Example:
            >>> settings = LTXVideoSettings()
            >>> unet, vae, scheduler = settings.get_model_paths()
            >>> print(f"UNet path: {unet}")
        """
        # Ensure model is downloaded
        self.download_model()
        
        unet_dir = self.ckpt_dir / "unet"
        vae_dir = self.ckpt_dir / "vae"
        scheduler_dir = self.ckpt_dir / "scheduler"
        
        return unet_dir, vae_dir, scheduler_dir
    
    def get_output_resolution(self) -> tuple[int, int]:
        """
        Get the output resolution as a tuple of (height, width).
        
        Returns:
            tuple[int, int]: Video dimensions as (height, width)

        Example:
            >>> settings = LTXVideoSettings(height=480, width=704)
            >>> h, w = settings.get_output_resolution()
            >>> print(f"Output resolution: {h}x{w}")
        """
        return (self.height, self.width)
    
    def get_padded_num_frames(self) -> int:
        """
        Calculate the padded number of frames.
        
        Ensures the number of frames is compatible with model requirements by
        padding to the nearest multiple of 8 frames if necessary.

        Returns:
            int: Padded frame count that's compatible with the model

        Example:
            >>> settings = LTXVideoSettings(num_frames=30)
            >>> padded = settings.get_padded_num_frames()
            >>> print(f"Padded frame count: {padded}")  # Will be 32
        """
        # Common video models often require frame counts to be multiples of 8
        FRAME_PADDING = 8
        
        # Calculate padding needed to reach next multiple of FRAME_PADDING
        remainder = self.num_frames % FRAME_PADDING
        if remainder == 0:
            return self.num_frames
            
        padding_needed = FRAME_PADDING - remainder
        return self.num_frames + padding_needed
    