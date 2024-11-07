from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class MochiWeightsSettings(BaseSettings):
    # Original download settings
    output_dir: Path = Path("weights")
    repo_id: str = "genmo/mochi-1-preview"
    model_file: str = "dit.safetensors"
    decoder_file: str = "decoder.safetensors"
    encoder_file: str = "encoder.safetensors"

    # Conversion specific settings (from original argparse)
    transformer_checkpoint_path: Path = Path("weights/dit.safetensors")
    vae_encoder_checkpoint_path: Path = Path("weights/encoder.safetensors")
    vae_decoder_checkpoint_path: Path = Path("weights/decoder.safetensors")
    output_path: Path = Path("weights/converted_models")
    push_to_hub: bool = False
    text_encoder_cache_dir: Optional[Path] = None
    dtype: Optional[str] = None  # Options: "fp16", "bf16", "fp32"

    class Config:
        env_prefix = "MOCHI_"
