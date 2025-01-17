"""
Configuration for Hunyuan Video Generation.

This module contains the configuration settings for the Hunyuan video generation model.
Settings can be configured via environment variables with the VIDEO_GEN_ prefix.

Example:
    To use with default settings:
        >>> from configs.hunyuan_config import HunyuanConfig
        >>> config = HunyuanConfig()
    
    To override via environment variables:
        $ export VIDEO_GEN_MODEL_ID="custom/model"
        $ export VIDEO_GEN_DEVICE_MAP="cuda"
        
    To override in code:
        >>> config = HunyuanConfig(
        ...     model_id="custom/model",
        ...     device_map="cuda"
        ... )
"""

from pydantic_settings import BaseSettings


class HunyuanConfig(BaseSettings):
    """Configuration settings for Hunyuan Video Generation.
    
    This class uses Pydantic's BaseSettings to manage configuration with environment
    variable support. All settings can be overridden via environment variables
    with the prefix VIDEO_GEN_.
    
    Attributes:
        model_id (str): The Huggingface model ID for the Hunyuan video model.
            Default: "hunyuanvideo-community/HunyuanVideo"
        device_map (str): Strategy for mapping model layers to devices.
            Options: "auto", "balanced", "sequential", or specific device like "cuda:0".
            Default: "balanced"
        load_in_8bit (bool): Whether to load the model in 8-bit quantization.
            Reduces memory usage at the cost of slight quality degradation.
            Default: True
        torch_dtype (str): PyTorch data type for model weights.
            Options: "float16", "float32", "bfloat16".
            Default: "float16"
        output_dir (str): Directory where generated videos will be saved.
            Default: "outputs"
        default_fps (int): Default frames per second for generated videos.
            Default: 15
    
    Example:
        >>> config = HunyuanConfig(
        ...     model_id="custom/model",
        ...     device_map="cuda",
        ...     output_dir="custom_outputs"
        ... )
        >>> print(config.model_id)
        'custom/model'
    """
    
    model_id: str = "hunyuanvideo-community/HunyuanVideo"
    device_map: str = "balanced"
    load_in_8bit: bool = True
    torch_dtype: str = "float16"
    output_dir: str = "outputs"
    default_fps: int = 15
    
    class Config:
        """Pydantic configuration class.
        
        Attributes:
            env_prefix: Prefix for environment variables.
                Example: VIDEO_GEN_MODEL_ID will set the model_id attribute.
        """
        env_prefix = "VIDEO_GEN_"
