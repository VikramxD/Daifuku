"""
LitServe API implementation for LTX video generation service.
"""

import os
import sys
import time
import tempfile
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from litserve import LitAPI, LitServer, Logger
from loguru import logger
from prometheus_client import (
    CollectorRegistry,
    Histogram,
    make_asgi_app,
    multiprocess
)

from configs.ltx_settings import LTXVideoSettings
from scripts.ltx_inference import LTXInference
from scripts import mp4_to_s3_json 

# Set up prometheus multiprocess mode
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

# Initialize prometheus registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class PrometheusLogger(Logger):
    """Custom logger for Prometheus metrics."""
    
    def __init__(self):
        super().__init__()
        self.function_duration = Histogram(
            "ltx_request_processing_seconds",
            "Time spent processing LTX video request",
            ["function_name"],
            registry=registry
        )

    def process(self, key: str, value: float) -> None:
        """Process and record metric."""
        self.function_duration.labels(function_name=key).observe(value)

class VideoGenerationRequest(BaseModel):
    """Model representing a video generation request."""
    
    prompt: str = Field(..., description="Text description of the video to generate")
    negative_prompt: Optional[str] = Field(
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Text description of what to avoid"
    )
    num_inference_steps: int = Field(
        40,
        ge=1,
        le=100,
        description="Number of inference steps"
    )
    guidance_scale: float = Field(
        3.0,
        ge=1.0,
        le=20.0,
        description="Guidance scale for generation"
    )
    height: int = Field(
        480,
        ge=256,
        le=720,
        multiple_of=32,
        description="Video height in pixels"
    )
    width: int = Field(
        704,
        ge=256,
        le=1280,
        multiple_of=32,
        description="Video width in pixels"
    )
    num_frames: int = Field(
        121,
        ge=1,
        le=257,
        description="Number of frames to generate"
    )
    frame_rate: int = Field(
        25,
        ge=1,
        le=60,
        description="Frames per second for output"
    )
    seed: Optional[int] = Field(None, description="Random seed for generation")

class LTXVideoAPI(LitAPI):
    """API for LTX video generation using LitServer."""

    def setup(self, device: str) -> None:
        """Initialize the LTX video generation model."""
        try:
            logger.info(f"Initializing LTX video generation on device: {device}")
            
            # Initialize settings
            self.settings = LTXVideoSettings(
                device=device,
                ckpt_dir=os.environ.get("LTX_CKPT_DIR", "checkpoints"),
            )
            
            # Initialize inference engine
            self.engine = LTXInference(self.settings)
            
            # Create output directory
            self.output_dir = Path("outputs")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("LTX setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during LTX setup: {e}")
            raise

    def decode_request(
        self,
        request: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Decode and validate the incoming request."""
        try:
            # Ensure request is a list
            if not isinstance(request, list):
                request = [request]
            
            # Validate each request
            validated_requests = [
                VideoGenerationRequest(**req).model_dump()
                for req in request
            ]
            return validated_requests
            
        except Exception as e:
            logger.error(f"Error in decode_request: {e}")
            raise

    def batch(
        self,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, List[Any]]:
        """Prepare inputs for batch processing."""
        try:
            # Convert single input to list
            if not isinstance(inputs, list):
                inputs = [inputs]

            # Get default values
            defaults = VideoGenerationRequest().model_dump()
            
            # Initialize batch dictionary
            batched = {
                "prompt": [],
                "negative_prompt": [],
                "num_inference_steps": [],
                "guidance_scale": [],
                "height": [],
                "width": [],
                "num_frames": [],
                "frame_rate": [],
                "seed": []
            }
            
            # Fill batch dictionary
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
                    # Validate request
                    generation_request = VideoGenerationRequest(**request)
                    
                    # Create temporary directory for output
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_video_path = Path(temp_dir) / f"ltx_{int(time.time())}.mp4"
                        
                        # Update settings with request parameters
                        self.settings.prompt = generation_request.prompt
                        self.settings.negative_prompt = generation_request.negative_prompt
                        self.settings.num_inference_steps = generation_request.num_inference_steps
                        self.settings.guidance_scale = generation_request.guidance_scale
                        self.settings.height = generation_request.height
                        self.settings.width = generation_request.width
                        self.settings.num_frames = generation_request.num_frames
                        self.settings.frame_rate = generation_request.frame_rate
                        self.settings.seed = generation_request.seed
                        self.settings.output_path = str(temp_video_path)
                        
                        # Generate video
                        logger.info(f"Starting generation for prompt: {generation_request.prompt}")
                        self.engine.generate()
                        
                        end_time = time.time()
                        generation_time = end_time - start_time
                        self.log("inference_time", generation_time)
                        
                        # Get memory statistics
                        memory_stats = self.engine.get_memory_stats()
                        
                        # Upload to S3
                        s3_response = mp4_to_s3_json(
                            temp_video_path,
                            f"ltx_{int(time.time())}.mp4"
                        )
                        
                        result = {
                            "status": "success",
                            "video_id": s3_response["video_id"],
                            "video_url": s3_response["url"],
                            "prompt": generation_request.prompt,
                            "generation_params": generation_request.model_dump(),
                            "time_taken": generation_time,
                            "memory_usage": memory_stats
                        }
                        results.append(result)
                        
                        logger.info(f"Generation completed successfully")
                        
                except Exception as e:
                    logger.error(f"Error in generation: {e}")
                    results.append({
                        "status": "error",
                        "error": str(e)
                    })
                    
                finally:
                    # Cleanup
                    if hasattr(self.engine, 'clear_memory'):
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

    def encode_response(
        self,
        output: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Encode the output for response."""
        try:
            # Handle list output
            if isinstance(output, list):
                output = output[0] if output else {
                    "status": "error",
                    "error": "No output generated"
                }
            
            # Handle error cases
            if output.get("status") == "error":
                return {
                    "status": "error",
                    "error": output.get("error", "Unknown error"),
                    "item_index": output.get("item_index")
                }
            
            # Return successful response
            return {
                "status": "success",
                "video_id": output.get("video_id"),
                "video_url": output.get("video_url"),
                "generation_info": {
                    "prompt": output.get("prompt"),
                    "parameters": output.get("generation_params", {})
                },
                "performance": {
                    "time_taken": round(output.get("time_taken", 0), 2),
                    "memory_usage": output.get("memory_usage", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in encode_response: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

def main():
    """Main entry point for the API server."""
    # Initialize Prometheus logger
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(
        path="/api/v1/metrics",
        app=make_asgi_app(registry=registry)
    )
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/ltx_api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    try:
        # Initialize API and server
        api = LTXVideoAPI()
        server = LitServer(
            api,
            api_path='/api/v1/video/ltx',
            accelerator="auto",
            devices="auto",
            max_batch_size=1,
            track_requests=True,
            loggers=prometheus_logger,
        )
        
        # Start server
        logger.info("Starting LTX video generation server on port 8000")
        server.run(port=8000)
        
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()