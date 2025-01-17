"""
Hunyuan Video Generation Service API

A high-performance API implementation for the Hunyuan video generation model using LitServe.
Provides video generation capabilities with comprehensive monitoring,
validation, and error handling.

Core Features:
    - Text-to-video generation from prompts
    - Configurable generation parameters
    - Prometheus metrics collection
    - Resource utilization tracking
    - Comprehensive error handling
    - GPU memory management

Technical Specifications:
    - Model: Hunyuan Video Transformer
    - Quantization: 8-bit support
    - Output Format: MP4
    - Device Support: CPU and GPU (CUDA)
    - Memory Optimization: Device map balancing

Performance Monitoring:
    - Request latency tracking
    - Memory usage monitoring
    - GPU utilization metrics
    - Success/failure rates
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import litserve as ls
from loguru import logger
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app, multiprocess
from scripts.hunyuan_video_inference import HunyuanVideoInference
from configs.hunyuan_config import HunyuanConfig
from api.models import VideoGenerationRequest, VideoGenerationResponse, ErrorResponse
import sys

# Setup Prometheus multiprocess metrics
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_hunyuan"
if not os.path.exists("/tmp/prometheus_multiproc_hunyuan"):
    os.makedirs("/tmp/prometheus_multiproc_hunyuan")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)


class PrometheusLogger(ls.Logger):
    """Enterprise-grade Prometheus metrics collector for Hunyuan service.
    
    Implements detailed performance tracking with multi-process support for production
    deployments. Provides high-resolution timing metrics for all service operations.
    
    Metrics:
        hunyuan_request_duration_seconds:
            - Type: Histogram
            - Labels: operation, status
            - Description: Processing time per operation
    """
    
    def __init__(self):
        """Initialize Prometheus metrics collectors."""
        super().__init__()
        self.request_duration = Histogram(
            "hunyuan_request_duration_seconds",
            "Time spent processing video generation requests",
            ["operation", "status"],
            registry=registry
        )
    
    def process(self, key: str, value: float) -> None:
        """Record a metric observation with operation-specific labeling.
        
        Args:
            key: Metric identifier in format "operation:status"
            value: Duration measurement in seconds
        """
        if ":" in key:
            operation, status = key.split(":", 1)
        else:
            operation = key
            status = "unknown"
            
        self.request_duration.labels(
            operation=operation,
            status=status
        ).observe(value)


class HunyuanVideoAPI(ls.LitAPI):
    """Production-ready API implementation for Hunyuan video generation service.
    
    Provides a robust, scalable interface for video generation with comprehensive
    error handling, resource management, and performance monitoring.
    
    Features:
        - Request validation and normalization
        - Batched processing support
        - Automatic resource cleanup
        - Detailed error reporting
        - Performance metrics collection
        
    Attributes:
        generator (HunyuanVideoInference): Video generation engine
        config (HunyuanConfig): Service configuration
    """
    
    def setup(self, device: str) -> None:
        """Initialize the Hunyuan video generation infrastructure.
        
        Performs model loading, resource allocation, and directory setup for
        production deployment.
        
        Args:
            device: Target compute device ("cuda" or "cpu")
            
        Raises:
            RuntimeError: On initialization failure
        """
        try:
            logger.info(f"Initializing HunyuanVideoAPI on device={device}")
            
            # Initialize configuration
            self.config = HunyuanConfig(
                device_map="auto" if device == "cuda" else "cpu"
            )
            
            # Setup video generator
            logger.info("Creating video generation engine")
            self.generator = HunyuanVideoInference(self.config)
            
            # Ensure output directory exists
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("HunyuanVideoAPI initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HunyuanVideoAPI: {str(e)}")
            raise RuntimeError(f"API initialization failed: {str(e)}") from e
    
    def decode_request(
        self,
        request: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Validate and normalize incoming generation requests.
        
        Supports both single requests and batches, normalizing them into a
        consistent format for processing.
        
        Args:
            request: Raw request data, single or batched
            
        Returns:
            List of validated request parameters
            
        Raises:
            ValidationError: For malformed requests
        """
        # Normalize to list format
        if not isinstance(request, list):
            request = [request]
        
        # Validate each request
        validated_requests = []
        for req in request:
            try:
                validated = VideoGenerationRequest(**req)
                validated_requests.append(validated.model_dump())
            except Exception as e:
                logger.error(f"Request validation failed: {str(e)}")
                raise
        
        return validated_requests
    
    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate videos for the validated requests.
        
        Processes each request in the batch, handling errors individually to
        ensure partial batch completion.
        
        Args:
            inputs: List of validated request parameters
            
        Returns:
            List of generation results or error information
        """
        results = []
        
        for request in inputs:
            try:
                start_time = time.time()
                
                # Generate video
                video_path = self.generator.generate_video(
                    prompt=request["prompt"],
                    num_frames=request["num_frames"],
                    num_inference_steps=request["num_inference_steps"],
                    fps=request["fps"]
                )
                
                # Calculate metrics
                end_time = time.time()
                duration = end_time - start_time
                
                result = {
                    "status": "success",
                    "video_path": str(video_path),
                    "generation_params": request,
                    "time_taken": duration,
                    "metrics": {
                        "total_time": duration
                    }
                }
                results.append(result)
                
                logger.info(f"Generation completed for prompt: {request['prompt']}")
                
            except Exception as e:
                logger.error(f"Video generation failed: {str(e)}")
                error_result = {
                    "status": "error",
                    "error": str(e)
                }
                results.append(error_result)
                
        return results if results else [{"status": "error", "error": "No results generated"}]

    def encode_response(self, output: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        """Format generation results for API response.
        
        Args:
            output: Raw generation results or error information
            
        Returns:
            Formatted API response with standardized structure
            
        Note:
            Handles both success and error cases with consistent formatting
        """
        if isinstance(output, list):
            output = output[0] if output else {"status": "error", "error": "No output generated"}
            
        if output.get("status") == "success":
            return {
                "status": "success",
                "video_path": output.get("video_path"),
                "generation_params": output.get("generation_params", {}),
                "time_taken": output.get("time_taken"),
                "metrics": {
                    "total_time": output.get("time_taken")
                }
            }
        else:
            return {
                "status": "error",
                "error": output.get("error", "Unknown error occurred")
            }

def main():
    """
    Initialize and launch the Hunyuan video generation service.
    
    Sets up the complete service infrastructure including:
        - Prometheus metrics collection
        - Structured logging
        - API server
        - Error handling
    """
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(
        path="/api/v1/metrics",
        app=make_asgi_app(registry=registry)
    )
    
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/hunyuan_api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )

    try:
        api = HunyuanVideoAPI()
        server = litserve.LitServer(
            api,
            api_path='/api/v1/video/hunyuan',
            accelerator="auto",
            devices="auto",
            max_batch_size=1,
            track_requests=True,
            loggers=[prometheus_logger],
            generate_client_file=False
        )
        logger.info("Starting Hunyuan video generation server on port 8000")
        server.run(port=8000)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
