"""
LitServe API implementation for Mochi video generation service.
"""

import io
import base64
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from litserve import LitAPI, LitServer
from loguru import logger

from configs.mochi_settings import MochiSettings
from scripts.mochi_diffusers import MochiInference
from scripts import mp4_to_s3_json

class VideoGenerationRequest(BaseModel):
    """
    Model representing a video generation request.

    Attributes:
        prompt: Text description of the video to generate.
        negative_prompt: Optional text description of what to avoid.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of frames to generate.
        fps: Frames per second for output video.
        seed: Random seed for reproducibility.
    """
    prompt: str = Field(..., description="Text description of the video to generate")
    negative_prompt: Optional[str] = Field(None, description="Text description of what to avoid")
    num_inference_steps: int = Field(50, ge=1, le=1000, description="Number of inference steps")
    guidance_scale: float = Field(4.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    height: int = Field(480, ge=64, le=1024, multiple_of=8, description="Video height in pixels")
    width: int = Field(848, ge=64, le=1024, multiple_of=8, description="Video width in pixels")
    num_frames: int = Field(161, ge=1, le=1000, description="Number of frames to generate")
    fps: int = Field(15, ge=1, le=120, description="Frames per second for output")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class MochiVideoAPI(LitAPI):
    """
    API for Mochi video generation using LitServer.
    
    This class implements the LitAPI interface to provide video generation
    functionality using the Mochi model.
    """

    def setup(self, device: str) -> None:
        """
        Initialize the Mochi video generation model.

        Args:
            device: The device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.settings = MochiSettings(
            model_name="Mini-Mochi",
            enable_vae_tiling=True,
            enable_attention_slicing=True,
            device=device
        )
        
        logger.info("Initializing Mochi inference engine")
        self.engine = MochiInference(self.settings)
        
        # Create output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the incoming request into a format suitable for processing.

        Args:
            request: The raw incoming request data.

        Returns:
            Dict containing the decoded request data.

        Raises:
            Exception: If there's an error in decoding the request.
        """
        try:
            generation_request = VideoGenerationRequest(**request)
            return generation_request.model_dump()
        except Exception as e:
            logger.error(f"Error in decode_request: {e}")
            raise

    def batch(self, inputs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Prepare a batch of inputs for processing.

        Args:
            inputs: List of individual input dictionaries.

        Returns:
            Dict containing batched inputs.
        """
        return {
            "prompt": [input["prompt"] for input in inputs],
            "negative_prompt": [input.get("negative_prompt") for input in inputs],
            "num_inference_steps": [input["num_inference_steps"] for input in inputs],
            "guidance_scale": [input["guidance_scale"] for input in inputs],
            "height": [input["height"] for input in inputs],
            "width": [input["width"] for input in inputs],
            "num_frames": [input["num_frames"] for input in inputs],
            "fps": [input["fps"] for input in inputs],
            "seed": [input.get("seed") for input in inputs]
        }

    def predict(self, inputs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of inputs and return the results.

        Args:
            inputs: Dictionary containing batched inputs.

        Returns:
            List of dictionaries containing the prediction results.
        """
        results = []
        for i in range(len(inputs["prompt"])):
            start_time = time.time()
            try:
                # Create unique output path
                output_path = self.output_dir / f"mochi_{int(time.time())}_{i}.mp4"
                
                # Generate video
                self.engine.generate(
                    prompt=inputs["prompt"][i],
                    negative_prompt=inputs["negative_prompt"][i],
                    num_inference_steps=inputs["num_inference_steps"][i],
                    guidance_scale=inputs["guidance_scale"][i],
                    height=inputs["height"][i],
                    width=inputs["width"][i],
                    num_frames=inputs["num_frames"][i],
                    fps=inputs["fps"][i],
                    seed=inputs["seed"][i],
                    output_path=str(output_path)
                )
                
                end_time = time.time()
                
                # Get memory usage
                allocated, peak = self.engine.get_memory_usage()
                
                results.append({
                    "video_path": str(output_path),
                    "prompt": inputs["prompt"][i],
                    "seed": inputs["seed"][i],
                    "time_taken": end_time - start_time,
                    "memory_usage": {
                        "allocated_gb": round(allocated, 2),
                        "peak_gb": round(peak, 2)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error in predict for item {i}: {e}")
                results.append(None)
                
            finally:
                self.engine.clear_memory()
                
        return results

    def unbatch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert batched outputs back to individual results.

        Args:
            outputs: List of output dictionaries from the predict method.

        Returns:
            The same list of output dictionaries.
        """
        return outputs

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode the output and prepare the response.

        Args:
            output: Dictionary containing the prediction output.

        Returns:
            Dictionary containing the encoded response with video path, generation info,
            and performance metrics.

        Raises:
            Exception: If there's an error in encoding the response.
        """
       

        try:
            return {
                "status": "success",
                "video_path": output["video_path"],
                "generation_info": {
                    "prompt": output["prompt"],
                    "seed": output["seed"],
                },
                "performance": {
                    "time_taken": round(output["time_taken"], 2),
                    "memory_usage": output["memory_usage"]
                }
            }
        except Exception as e:
            logger.error(f"Error in encode_response: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    import sys
    
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    api = MochiVideoAPI()
    server = LitServer(
        api,
        api_path='/api/v1/video/mochi',
        accelerator="auto",
        devices ="auto",
        max_batch_size=1, 
        batch_timeout=50000
    )
    server.run(port=8000)