from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal,Optional




class VideoGenSettings(BaseSettings):
    """
    Configuration settings for video generation using the Hunyuan model.
    
    Inherits from Pydantic BaseSettings to support loading from environment variables
    and .env files with the prefix VIDEO_GEN_.
    
    Attributes:
        model_path (str): Path to the pretrained model directory
        prompt (str): Text description of the video to generate
        resolution (str): Video dimensions in "WxH" format (e.g., "1280x720")
        video_length (int): Number of frames to generate (65 for 2s, 129 for 5s)
        seed (int): Random seed for generation (-1 for random seed)
        num_inference_steps (int): Number of denoising steps (1-100)
        guidance_scale (float): Classifier-free guidance scale (1.0-20.0)
        flow_shift (float): Flow shift parameter for motion control (0.0-10.0)
        embedded_guidance_scale (float): Scale for embedded guidance (1.0-20.0)
        output_dir (str): Directory path for saving generated videos
    """
    model_path: str = Field('/root/Daifuku/hunyuan/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt', description="Path to the model")
    prompt: str = Field(
        default="A cat walks on the grass, realistic style.",
        description="Prompt for video generation"
    )
    resolution: Literal[
        "1280x720", "720x1280", "1104x832", "832x1104", "960x960",
        "960x544", "544x960", "832x624", "624x832", "720x720"
    ] = Field(default="1280x720", description="Video resolution")
    video_length: Literal[65, 129] = Field(
        default=129,
        description="Video length in frames (65 for 2s, 129 for 5s)"
    )
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    num_inference_steps: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=20.0,
        description="Guidance scale"
    )
    flow_shift: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Flow shift"
    )
    embedded_guidance_scale: float = Field(
        default=6.0,
        ge=1.0,
        le=20.0,
        description="Embedded guidance scale"
    )
    output_dir: str = Field(
        default="outputs",
        description="Directory to save generated videos"
    )

    class Config:
        env_file = ".env"
        env_prefix = "VIDEO_GEN_"