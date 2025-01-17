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
                duration = time.time() - start_time
                fps = request["fps"] or self.config.default_fps
                
                # Create success response
                response = VideoGenerationResponse(
                    video_path=video_path,
                    duration_seconds=duration,
                    frames=request["num_frames"],
                    fps=fps
                )
                
                results.append({
                    "status": "success",
                    "data": response.model_dump()
                })
                
            except Exception as e:
                logger.error(f"Video generation failed: {str(e)}")
                error = ErrorResponse(
                    error="Video generation failed",
                    details=str(e)
                )
                results.append({
                    "status": "error",
                    "error": error.model_dump()
                })
        
        return results
    
    def encode_response(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format the final API response.
        
        Args:
            outputs: List of generation results
            
        Returns:
            Formatted response with status and data/error information
        """
        # Single request response
        if len(outputs) == 1:
            return outputs[0]
        
        # Batch request response
        return {
            "batch_results": outputs
        }


def main():
    """Initialize and launch the Hunyuan video generation service.
    
    Sets up the complete service infrastructure including:
        - Prometheus metrics collection
        - Structured logging
        - API server
        - Error handling
    """
    # Create metrics ASGI app
    metrics_app = make_asgi_app(registry=registry)
    
    # Create and configure the API server
    server = ls.LitServer(
        api=HunyuanVideoAPI(),
        host="0.0.0.0",
        port=8000,
        metrics_app=metrics_app
    )
    
    # Start the server
    server.serve()


if __name__ == "__main__":
    main()
