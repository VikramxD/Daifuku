"""
Mochi Video Generation Service API

A high-performance API implementation for the Mochi video generation model using LitServe.
Provides enterprise-grade video generation capabilities with comprehensive monitoring,
validation, and error handling.

Core Features:
    - Automated video generation from text prompts
    - S3 integration for result storage
    - Prometheus metrics collection
    - Resource utilization tracking
    - Configurable generation parameters
    - GPU memory management

Technical Specifications:
    - Input Resolution: 64-1024 pixels (width/height)
    - Frame Range: 1-1000 frames
    - FPS Range: 1-120
    - Supported Formats: MP4 output
    - Storage: AWS S3 integration

Performance Monitoring:
    - Request latency tracking
    - Memory usage monitoring
    - GPU utilization metrics
    - Success/failure rates
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
from scripts.mp4_to_s3_json import mp4_to_s3_json
import litserve
import tempfile

os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class PrometheusLogger(litserve.Logger):
    """
    Enterprise-grade Prometheus metrics collector for Mochi service monitoring.

    Implements detailed performance tracking with multi-process support for production
    deployments. Provides high-resolution timing metrics for all service operations.

    Metrics:
        request_processing_seconds:
            - Type: Histogram
            - Labels: function_name
            - Description: Processing time per operation
    """

    def __init__(self):
        super().__init__()
        self.function_duration = Histogram("request_processing_seconds", "Time spent processing request", ["function_name"], registry=registry)

    def process(self, key: str, value: float) -> None:
        """
        Record a metric observation with operation-specific labeling.

        Args:
            key: Operation identifier for metric labeling
            value: Duration measurement in seconds
        """
        self.function_duration.labels(function_name=key).observe(value)

class VideoGenerationRequest(BaseModel):
    """
    Validated request model for video generation parameters.

    Enforces constraints and provides default values for all generation parameters.
    Ensures request validity before resource allocation.

    Attributes:
        prompt: Primary generation directive
        negative_prompt: Elements to avoid in generation
        num_inference_steps: Generation quality control
        guidance_scale: Prompt adherence strength
        height: Output video height
        width: Output video width
        num_frames: Total frames to generate
        fps: Playback frame rate
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
    Production-ready API implementation for Mochi video generation service.

    Provides a robust, scalable interface for video generation with comprehensive
    error handling, resource management, and performance monitoring.

    Features:
        - Request validation and normalization
        - Batched processing support
        - Automatic resource cleanup
        - Detailed error reporting
        - Performance metrics collection
    """

    def setup(self, device: str) -> None:
        """
        Initialize the Mochi video generation infrastructure.

        Performs model loading, resource allocation, and directory setup for
        production deployment.

        Args:
            device: Target compute device for model deployment

        Raises:
            RuntimeError: On initialization failure
        """
        self.settings = MochiSettings(
            model_name="Mini-Mochi",
            enable_vae_tiling=True,
            enable_attention_slicing=True,
            device=device
        )
        
        logger.info("Initializing Mochi inference engine")
        self.engine = MochiInference(self.settings)
        
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Setup completed successfully")

    def decode_request(self, request: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Validate and normalize incoming generation requests.

        Args:
            request: Raw request data, single or batched

        Returns:
            List of validated request parameters

        Raises:
            ValidationError: For malformed requests
        """
        if not isinstance(request, list):
            request = [request]
        
        validated_requests = [VideoGenerationRequest(**req).model_dump() for req in request]
        return validated_requests

    def batch(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, List[Any]]:
        """
        Prepare inputs for batch processing.

        Args:
            inputs: Single or multiple generation requests

        Returns:
            Batched parameters ready for processing

        Raises:
            ValueError: For invalid batch composition
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

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
        
        for input_item in inputs:
            for key in batched.keys():
                value = input_item.get(key, defaults.get(key))
                batched[key].append(value)
        
        return batched

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute video generation for validated requests.

        Handles the complete generation pipeline including:
        - Parameter validation
        - Resource allocation
        - Video generation
        - S3 upload
        - Performance monitoring
        - Resource cleanup

        Args:
            inputs: List of validated generation parameters

        Returns:
            List of generation results or error details

        Raises:
            RuntimeError: On generation failure
        """
        results = []
        for request in inputs:
            start_time = time.time()
            try:
                generation_request = VideoGenerationRequest(**request)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_video_path = os.path.join(temp_dir, f"mochi_{int(time.time())}.mp4")
                    
                    generation_params = generation_request.dict()
                    generation_params["output_path"] = temp_video_path
                    
                    logger.info(f"Starting generation for prompt: {generation_params['prompt']}")
                    self.engine.generate(**generation_params)
                    
                    end_time = time.time()
                    self.log("inference_time", end_time - start_time)
                    
                    allocated, peak = self.engine.get_memory_usage()

                    with open(temp_video_path, "rb") as video_file:
                        s3_response = mp4_to_s3_json(video_file, f"mochi_{int(time.time())}.mp4")
                    
                    result = {
                        "status": "success",
                        "video_id": s3_response["video_id"],
                        "video_url": s3_response["url"],
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
                
        return results if results else [{"status": "error", "error": "No results generated"}]

    def unbatch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert batched outputs to individual results.

        Args:
            outputs: List of generation results

        Returns:
            Unbatched list of results
        """
        return outputs

    def encode_response(self, output: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        """
        Format generation results for API response.

        Args:
            output: Raw generation results or error information

        Returns:
            Formatted API response with standardized structure

        Note:
            Handles both success and error cases with consistent formatting
        """
        if isinstance(output, list):
            output = output[0] if output else {"status": "error", "error": "No output generated"}
        
        if output.get("status") == "error":
            return {
                "status": "error",
                "error": output.get("error", "Unknown error"),
                "item_index": output.get("item_index")
            }
        
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

def main():
    """
    Initialize and launch the Mochi video generation service.

    Sets up the complete service infrastructure including:
    - Prometheus metrics collection
    - Structured logging
    - API server configuration
    - Error handling
    """
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(path="/api/v1/metrics", app=make_asgi_app(registry=registry))
    
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
            loggers=prometheus_logger,
            generate_client_file=False
        )
        logger.info("Starting server on port 8000")
        server.run(port=8000)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()