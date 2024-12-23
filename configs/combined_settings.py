"""
combined_settings.py

Central Pydantic models and optional unified config for combining Mochi and LTX requests/settings.

1) CombinedItemRequest:
   - Defines the schema for a single text-to-video request, including model_name (mochi/ltx)
     and common fields like prompt, negative_prompt, resolution, etc.

2) CombinedBatchRequest:
   - Defines a list of CombinedItemRequest items, handling batch-mode requests.

3) CombinedConfig (optional):
   - Demonstrates how you might wrap both mochi_settings and ltx_settings into a single config
     so everything can be accessed from a unified settings object if needed.

Usage Example (Batch):
    POST /predict
    {
      "batch": [
        {
          "model_name": "mochi",
          "prompt": "A calm ocean scene at sunset",
          "negative_prompt": "blurry, worst quality",
          "num_inference_steps": 50,
          "guidance_scale": 4.5,
          "height": 480,
          "width": 848
        },
        {
          "model_name": "ltx",
          "prompt": "Golden autumn leaves swirling",
          "num_inference_steps": 40,
          "guidance_scale": 3.0,
          "height": 480,
          "width": 704
        }
      ]
    }
"""

from pydantic import BaseModel, Field
from typing import List, Optional

# If you want to embed or reference them here:
from .mochi_settings import MochiSettings
from .ltx_settings import LTXVideoSettings
from pydantic_settings import BaseSettings


class CombinedItemRequest(BaseModel):
    """
    A single request object for either Mochi or LTX.

    Fields:
        model_name (str): Which model to use: 'mochi' or 'ltx'.
        prompt (str): Main prompt describing the video content.
        negative_prompt (Optional[str]): Text describing what to avoid.
        num_inference_steps (Optional[int]): Override for inference steps.
        guidance_scale (Optional[float]): Classifier-free guidance scale.
        height (Optional[int]): Video height in pixels.
        width (Optional[int]): Video width in pixels.
        (Add additional fields as needed for your models.)
    """
    model_name: str = Field(..., description="Model to use: 'ltx' or 'mochi'.")
    prompt: str = Field(..., description="Prompt describing the video content.")
    negative_prompt: Optional[str] = Field(None, description="Things to avoid in generation.")
    num_inference_steps: Optional[int] = Field(40, description="Number of denoising steps.")
    guidance_scale: Optional[float] = Field(3.0, description="Guidance scale for generation.")
    height: Optional[int] = Field(480, description="Video height in pixels.")
    width: Optional[int] = Field(704, description="Video width in pixels.")
    # Add any more fields your sub-models need, e.g. fps, frames, etc.


class CombinedBatchRequest(BaseModel):
    """
    A batched request containing multiple CombinedItemRequest items.

    Usage:
        {
          "batch": [
            { "model_name": "mochi", "prompt": "...", ... },
            { "model_name": "ltx",   "prompt": "...", ... }
          ]
        }
    """
    batch: List[CombinedItemRequest] = Field(
        ..., description="List of multiple CombinedItemRequest items to process in parallel."
    )


class CombinedConfig(BaseSettings):
    """
    Optional: A unified config that embeds or references your Mochi/LTX model settings.

    This can be used if you want to store and manipulate both sets of settings in one place.
    For example, you might define environment variables to override mochi or ltx defaults.

    Usage:
        from configs.combined_settings import CombinedConfig
        combined_config = CombinedConfig()
        # Access mochi or ltx settings: combined_config.mochi_config, combined_config.ltx_config
    """
    mochi_config: MochiSettings = MochiSettings()
    ltx_config: LTXVideoSettings = LTXVideoSettings()

    class Config:
        env_prefix = "COMBINED_"
        validate_assignment = True
        arbitrary_types_allowed = True
