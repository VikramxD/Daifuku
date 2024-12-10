"""
Combined API router for multiple LitServe-based models.

This script imports multiple model-specific LitAPI classes (e.g., LTXVideoAPI
and MochiVideoAPI) and integrates them into a single endpoint. Clients specify
which model to invoke by providing a `model_name` field in the request body.

Features:
- Single endpoint routing for multiple models
- Prometheus metrics for request duration tracking
- Comprehensive logging (stdout and file) with loguru
- Detailed docstrings and structured JSON responses
- Extensible: Just add new model APIs and register them in `model_apis`.

Usage:
1. Ensure `ltx_serve.py` and `mochi_serve.py` are in the same directory.
2. Run `python combined_serve.py`.
3. Send POST requests to `http://localhost:8000/predict` with JSON like:
   {
     "model_name": "ltx",
     "prompt": "Generate a video about a sunny day at the beach"
   }

   or

   {
     "model_name": "mochi",
     "prompt": "Generate a video about a futuristic city"
   }
"""

import sys
import os
import time
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field
from loguru import logger

import torch
import litserve as ls
from prometheus_client import (
    CollectorRegistry,
    Histogram,
    make_asgi_app,
    multiprocess
)

# Import the individual model APIs
from ltx_serve import LTXVideoAPI
from mochi_serve import MochiVideoAPI

# Setup Prometheus multiprocess mode
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class PrometheusLogger(ls.Logger):
    """Custom logger for Prometheus metrics."""
    def __init__(self):
        super().__init__()
        self.function_duration = Histogram(
            "combined_request_processing_seconds",
            "Time spent processing combined API request",
            ["function_name"],
            registry=registry
        )

    def process(self, key: str, value: float) -> None:
        """Record metric observations for function durations."""
        self.function_duration.labels(function_name=key).observe(value)

class CombinedRequest(BaseModel):
    """
    Pydantic model for incoming requests to the combined endpoint.
    The `model_name` field is used to select which model to route to.
    Other fields depend on the target model, so they are optional here.
    """
    model_name: str = Field(..., description="Name of the model to use (e.g., 'ltx' or 'mochi').")
    # Any additional fields will be passed through to the selected model's decode_request.
    # We keep this flexible by using an extra allowed attributes pattern.
    # For more strict validation, define fields matching each model's requirements.
    class Config:
        extra = "allow"

class CombinedAPI(ls.LitAPI):
    """
    A combined API class that delegates requests to multiple model-specific APIs
    based on the `model_name` field in the request.

    This approach allows adding new models by:
    1. Importing their API class.
    2. Initializing and registering them in `model_apis` dictionary.
    """
    def setup(self, device: str) -> None:
        """Setup all sub-model APIs and logging/metrics."""

        logger.info(f"Initializing combined API with device={device}")
        
        # Initialize sub-model APIs
        self.ltx_api = LTXVideoAPI()
        self.mochi_api = MochiVideoAPI()

        # Setup each sub-model on the provided device
        self.ltx_api.setup(device=device)
        self.mochi_api.setup(device=device)

        # Register them in a dictionary for easy routing
        self.model_apis = {
            "ltx": self.ltx_api,
            "mochi": self.mochi_api
        }

        logger.info("Combined API setup completed successfully.")

    def decode_request(
        self,
        request: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Decode the incoming request to determine which model to use.
        We expect `model_name` to route the request accordingly.
        The rest of the fields will be passed to the chosen model's decode_request.
        """
        if isinstance(request, list):
            # We handle only single requests for simplicity
            request = request[0]

        validated = CombinedRequest(**request).dict()
        model_name = validated.pop("model_name").lower()

        if model_name not in self.model_apis:
            raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(self.model_apis.keys())}")

        # We'll store the selected model_name and request data
        return {
            "model_name": model_name,
            "request_data": validated
        }

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform prediction by routing to the chosen model API.

        Steps:
        1. Extract model_name and request_data.
        2. Pass request_data to the chosen model's decode_request -> predict pipeline.
        3. Return the predictions from the model.
        """
        model_name = inputs["model_name"]
        request_data = inputs["request_data"]
        model_api = self.model_apis[model_name]

        start_time = time.time()

        try:
            # The sub-model APIs typically handle lists of requests.
            # We'll wrap request_data in a list if needed.
            decoded = model_api.decode_request(request_data)
            # decoded is typically a list of requests for that model
            predictions = model_api.predict(decoded)
            # predictions is typically a list of results
            result = predictions[0] if predictions else {"status": "error", "error": "No result returned"}

            end_time = time.time()
            self.log("combined_inference_time", end_time - start_time)

            return {
                "model_name": model_name,
                "result": result
            }

        except Exception as e:
            import traceback
            logger.error(f"Error in combined predict: {e}\n{traceback.format_exc()}")
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
        """
        Encode the final response. We call the chosen model's encode_response if the result
        is from a model inference. If there's an error at the combined level, we return a generic error response.
        """
        model_name = output.get("model_name")
        if model_name and model_name in self.model_apis:
            # If there's a result from the model, encode it using the model's encoder
            result = output.get("result", {})
            if result.get("status") == "error":
                # Model-specific error case
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error"),
                    "traceback": result.get("traceback", None)
                }
            # Successful result
            encoded = self.model_apis[model_name].encode_response(result)
            # Add the model name to the final response for clarity
            encoded["model_name"] = model_name
            return encoded
        else:
            # If we got here, there's a top-level routing error
            return {
                "status": "error",
                "error": output.get("error", "Unknown top-level error"),
                "traceback": output.get("traceback", None)
            }


def main():
    """Main entry point to run the combined server."""
    # Set up Prometheus logger
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
        "logs/combined_api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )

    logger.info("Starting Combined Video Generation Server on port 8000")

    # Initialize and run the combined server
    api = CombinedAPI()
    server = ls.LitServer(
        api,
        api_path="/predict",  # A single endpoint for all models
        accelerator="auto",
        devices="auto",
        max_batch_size=1,
        track_requests=True,
        loggers=prometheus_logger
    )
    server.run(port=8000)

if __name__ == "__main__":
    main()
