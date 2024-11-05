from enum import Enum
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings


class DTypes(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class MochiConversionSettings(BaseSettings):
    transformer_checkpoint_path: Optional[Path] = None
    vae_encoder_checkpoint_path: Optional[Path] = None
    vae_decoder_checkpoint_path: Optional[Path] = None
    output_path: Path
    push_to_hub: bool = False
    text_encoder_cache_dir: Optional[Path] = None
    dtype: Optional[DTypes] = None

    model_config = {
        "env_prefix": "MOCHI_",
        "extra": "ignore",
    }

    def get_torch_dtype(self):
        if self.dtype is None:
            return None
        import torch
        dtype_map = {
            DTypes.FP16: torch.float16,
            DTypes.BF16: torch.bfloat16,
            DTypes.FP32: torch.float32
        }
        return dtype_map[self.dtype]