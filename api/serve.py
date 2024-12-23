"""
Combined API router for LTX and Mochi video generation services.

This module provides a unified endpoint that can handle requests for both
LTX and Mochi video generation models. Clients specify which model to use
via the 'model_name' parameter in their requests.

Usage:
    POST /predict
    {
        "model_name": "ltx",  # or "mochi"
        "prompt": "your video prompt here",
        ...other model-specific parameters...
    }
"""

import sys
import os
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field
from loguru import logger
import torch
import litserve as ls
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app, multiprocess
from ltx_serve import LTXVideoAPI
from mochi_serve import MochiVideoAPI

# Setup Prometheus multiprocess mode
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc"
if not os.path.exists("/tmp/prometheus_multiproc"):
    os.makedirs("/tmp/prometheus_multiproc")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class PrometheusLogger(ls.Logger):
    """Custom logger for tracking combined API metrics."""
    
    def __init__(self):
        super().__init__()
        self.function_duration = Histogram(
            "combined_request_duration_seconds",
            "Time spent processing video generation requests",
            ["model_name", "function_name"],
            registry=registry
        )

    def process(self, key: str, value: float) -> None:
        """Record metric observations."""
        model_name, func_name = key.split(":", 1) if ":" in key else ("unknown", key)
        self.function_duration.labels(
            model_name=model_name,
            function_name=func_name
        ).observe(value)

class CombinedRequest(BaseModel):
    """Request model for the combined API endpoint."""
    
    model_name: str = Field(
        ...,
        description="Model to use for video generation ('ltx' or 'mochi')"
    )
    
    class Config:
        extra = "allow"  # Allow additional fields for model-specific parameters

class CombinedVideoAPI(ls.LitAPI):
    """Combined API for serving both LTX and Mochi video generation models."""

    def setup(self, device: str) -> None:
        """Initialize both video generation models.
        
        Args:
            device: Target device for model execution
        """
        logger.info(f"Setting up combined video API on device: {device}")
        
        # Initialize both APIs
        self.ltx_api = LTXVideoAPI()
        self.mochi_api = MochiVideoAPI()
        
        # Setup each model
        self.ltx_api.setup(device=device)
        self.mochi_api.setup(device=device)
        
        # Register models for routing
        self.model_apis = {
            "ltx": self.ltx_api,
            "mochi": self.mochi_api
        }
        
        logger.info("Successfully initialized all models")

    def decode_request(
        self,
        request: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Validate request and determine target model.
        
        Args:
            request: Raw request data
            
        Returns:
            Decoded request with model selection
            
        Raises:
            ValueError: If model_name is invalid
        """
        if isinstance(request, list):
            request = request[0]  # Handle single requests for now
            
        validated = CombinedRequest(**request).dict()
        model_name = validated.pop("model_name").lower()
        
        if model_name not in self.model_apis:
            raise ValueError(
                f"Invalid model_name: {model_name}. "
                f"Available models: {list(self.model_apis.keys())}"
            )
            
        return {
            "model_name": model_name,
            "request_data": validated
        }

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate model and generate video.
        
        Args:
            inputs: Decoded request data
            
        Returns:
            Generation results from selected model
        """
        model_name = inputs["model_name"]
        request_data = inputs["request_data"]
        model_api = self.model_apis[model_name]
        
        try:
            # Process request through selected model
            decoded = model_api.decode_request(request_data)
            predictions = model_api.predict(decoded)
            result = predictions[0] if predictions else {
                "status": "error",
                "error": "No result returned"
            }
            
            return {
                "model_name": model_name,
                "result": result
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error in {model_name} prediction: {str(e)}")
            return {
                "model_name": model_name,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Encode final response using model-specific encoder.
        
        Args:
            output: Raw model output
            
        Returns:
            Encoded response ready for client
        """
        model_name = output.get("model_name")
        if model_name and model_name in self.model_apis:
            result = output.get("result", {})
            
            if result.get("status") == "error":
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error"),
                    "traceback": result.get("traceback")
                }
                
            encoded = self.model_apis[model_name].encode_response(result)
            encoded["model_name"] = model_name
            return encoded
        else:
            return {
                "status": "error",
                "error": output.get("error", "Unknown routing error"),
                "traceback": output.get("traceback")
            }

def main():
    """Initialize and start the combined video generation server."""
    # Setup Prometheus metrics
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(
        path="/metrics",
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
        "logs/combined_api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )

    # Start server
    logger.info("Starting Combined Video Generation Server")
    api = CombinedVideoAPI()
    server = ls.LitServer(
        api,
        api_path="/predict",
        accelerator="auto",
        devices="auto",
        max_batch_size=1,
        track_requests=True,
        loggers=[prometheus_logger]
    )
    server.run(port=8000)

if __name__ == "__main__":
    main()
