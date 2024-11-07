"""
LitServe API implementation for Mochi video generation service.
"""

import io
import base64
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field
from litserve import LitAPI, LitServer
from loguru import logger
import os, time
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app, multiprocess
from configs.mochi_settings import MochiSettings
from scripts.mochi_diffusers import MochiInference
from scripts import mp4_to_s3_json
import litserve

# Set the directory for multiprocess mode
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"

# Ensure the directory exists
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")


# Use a multiprocess registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)



class PrometheusLogger(litserve.Logger):
    def __init__(self):
        super().__init__()
        self.function_duration = Histogram("request_processing_seconds", "Time spent processing request", ["function_name"], registry=registry)

    def process(self, key, value):
        print("processing", key, value)
        self.function_duration.labels(function_name=key).observe(value)

class VideoGenerationRequest(BaseModel):
    """
    Model representing a video generation request.
    """
    prompt: str = Field(..., description="Text description of the video to generate")
    negative_prompt: Optional[str] = Field("", description="Text description of what to avoid")
    num_inference_steps: int = Field(50, ge=1, le=1000, description="Number of inference steps")
    guidance_scale: float = Field(4.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    height: int = Field(480, ge=64, le=1024, multiple_of=8, description="Video height in pixels")
    width: int = Field(848, ge=64, le=1024, multiple_of=8, description="Video width in pixels")
    num_frames: int = Field(161, ge=1, le=1000, description="Number of frames to generate")
    fps: int = Field(15, ge=1, le=120, description="Frames per second for output")

class MochiVideoAPI(LitAPI):
    """
    API for Mochi video generation using LitServer.
    """

    def setup(self, device: str) -> None:
        """Initialize the Mochi video generation model."""
        try:
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
            
            logger.info("Setup completed successfully")
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            raise

    def decode_request(self, request: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Decode and validate the incoming request."""
        try:
            # Ensure the request is a list of dictionaries
            if not isinstance(request, list):
                request = [request]
            
            # Validate each request in the list
            validated_requests = [VideoGenerationRequest(**req).model_dump() for req in request]
            return validated_requests
        except Exception as e:
            logger.error(f"Error in decode_request: {e}")
            raise

    def batch(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, List[Any]]:
        """
        Prepare inputs for batch processing.
        """
        try:
            # Convert single input to list format
            if not isinstance(inputs, list):
                inputs = [inputs]

            # Initialize with default values
            defaults = VideoGenerationRequest().model_dump()
            
            batched = {
                "prompt": [],
                "negative_prompt": [],
                "num_inference_steps": [],
                "guidance_scale": [],
                "height": [],
                "width": [],
                "num_frames": [],
                "fps": []
            }
            
            # Fill batched dictionary
            for input_item in inputs:
                for key in batched.keys():
                    value = input_item.get(key, defaults.get(key))
                    batched[key].append(value)
            
            return batched
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process inputs and generate videos."""
        results = []
        try:
            for request in inputs:
                start_time = time.time()
                try:
                    # Validate and parse the request
                    generation_request = VideoGenerationRequest(**request)
                    
                    # Create unique output path
                    timestamp = int(time.time())
                    output_path = self.output_dir / f"mochi_{timestamp}.mp4"
                    
                    # Prepare generation parameters
                    generation_params = generation_request.dict()
                    generation_params["output_path"] = str(output_path)
                    
                    # Generate video
                    logger.info(f"Starting generation for prompt: {generation_params['prompt']}")
                    self.engine.generate(**generation_params)
                    
                    end_time = time.time()
                    self.log("inference_time", end_time - start_time)
                    
                    # Get memory usage
                    allocated, peak = self.engine.get_memory_usage()
                    
                    result = {
                        "status": "success",
                        "video_path": str(output_path),
                        "prompt": generation_params["prompt"],
                        "generation_params": generation_params,
                        "time_taken": end_time - start_time,
                        "memory_usage": {
                            "allocated_gb": round(allocated, 2),
                            "peak_gb": round(peak, 2)
                        }
                    }
                    results.append(result)
                    
                    logger.info(f"Generation completed for prompt: {generation_params['prompt']}")
                    
                except Exception as e:
                    logger.error(f"Error in generation for request: {e}")
                    error_result = {
                        "status": "error",
                        "error": str(e)
                    }
                    results.append(error_result)
                finally:
                    self.engine.clear_memory()
                    
        except Exception as e:
            logger.error(f"Error in predict method: {e}")
            results.append({
                "status": "error",
                "error": str(e)
            })
            
        return results if results else [{"status": "error", "error": "No results generated"}]

    def unbatch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert batched outputs back to individual results."""
        return outputs

    def encode_response(self, output: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        """Encode the output for response."""
        try:
            # If output is a list, take the first item
            if isinstance(output, list):
                output = output[0] if output else {"status": "error", "error": "No output generated"}
            
            # Handle error cases
            if output.get("status") == "error":
                return {
                    "status": "error",
                    "error": output.get("error", "Unknown error"),
                    "item_index": output.get("item_index")
                }
            
            # Upload video to S3 and get signed URL
            video_path = output.get("video_path")
            with open(video_path, 'rb') as video_file:
                video_bytes = io.BytesIO(video_file.read())
                s3_response = mp4_to_s3_json(video_bytes, Path(video_path).name)
            
            # Handle success cases
            return {
                "status": "success",
                "video_id": s3_response["video_id"],
                "video_url": s3_response["url"],
                "generation_info": {
                    "prompt": output.get("prompt"),
                    "parameters": output.get("generation_params", {})
                },
                "performance": {
                    "time_taken": round(output.get("time_taken", 0), 2),
                    "memory_usage": output.get("memory_usage", {
                        "allocated_gb": 0,
                        "peak_gb": 0
                    })
                }
            }
        except Exception as e:
            logger.error(f"Error in encode_response: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    import sys
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(path="/metrics", app=make_asgi_app(registry=registry))
    # Configure logging
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
    

    try:
        api = MochiVideoAPI()
        server = LitServer(
            api,
            api_path='/api/v1/video/mochi',
            accelerator="auto",
            devices="auto",
            max_batch_size=1,
            track_requests=True,
            loggers=prometheus_logger
        )
        logger.info("Starting server on port 8000")
        server.run(port=8000)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)