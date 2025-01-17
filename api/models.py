"""
API models for Hunyuan Video Generation service.

This module defines the Pydantic models used for request/response handling
in the Hunyuan Video Generation API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class VideoGenerationRequest(BaseModel):
    """Request model for video generation endpoint.
    
    Attributes:
        prompt (str): Text description of the video to generate
        num_frames (int): Number of frames to generate (default: 61)
        num_inference_steps (int): Number of denoising steps (default: 30)
        fps (int): Frames per second for the output video (default: 15)
    """
    
    prompt: str = Field(
        ...,
        description="Text description of the video to generate",
        example="A cat walks on the grass, realistic style."
    )
    num_frames: int = Field(
        default=61,
        ge=1,
        le=120,
        description="Number of frames to generate"
    )
    num_inference_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of denoising steps"
    )
    fps: Optional[int] = Field(
        default=15,
        ge=1,
        le=60,
        description="Frames per second for the output video"
    )


class VideoGenerationResponse(BaseModel):
    """Response model for video generation endpoint.
    
    Attributes:
        video_path (str): Path to the generated video file
        duration_seconds (float): Duration of the generated video in seconds
        frames (int): Number of frames in the video
        fps (int): Frames per second of the video
    """
    
    video_path: str = Field(
        ...,
        description="Path to the generated video file"
    )
    duration_seconds: float = Field(
        ...,
        description="Duration of the generated video in seconds"
    )
    frames: int = Field(
        ...,
        description="Number of frames in the video"
    )
    fps: int = Field(
        ...,
        description="Frames per second of the video"
    )


class ErrorResponse(BaseModel):
    """Error response model.
    
    Attributes:
        error (str): Error message
        details (Optional[str]): Additional error details if available
    """
    
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
