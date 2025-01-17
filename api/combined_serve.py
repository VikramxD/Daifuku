"""
combined_serve.py

A unified API for Mochi, LTX, Hunyuan, and Allegro video generation using batch-based parallel processing.
All requests (single or multi) are sent as a batch, and we process them concurrently.

Usage:
  1. Place this file in your repo (e.g., in `api/combined_serve.py`).
  2. Run: python combined_serve.py
  3. POST requests to http://localhost:8000/api/v1/inference

Expected request format:
{
  "batch": [
    {
      "model_name": "mochi",
      "prompt": "A calm ocean scene at sunset",
      "negative_prompt": "blurry, worst quality",
      "num_inference_steps": 50,
      "guidance_scale": 4.5,
      "height": 480,
      "width": 848
      ...
    },
    {
      "model_name": "hunyuan",
      "prompt": "A beautiful mountain landscape",
      "num_frames": 16,
      "num_inference_steps": 50,
      "fps": 8
    }
    ...
  ]
}
"""

import sys
import os
import torch
import litserve as ls
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List
from loguru import logger
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app, multiprocess
from api.ltx_serve import LTXVideoAPI
from api.mochi_serve import MochiVideoAPI
from api.hunyuan_serve import HunyuanVideoAPI
from api.allegro_serve import AllegroVideoAPI
from configs.combined_settings import CombinedBatchRequest, CombinedItemRequest
import time
from typing import Union

os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc"
if not os.path.exists("/tmp/prometheus_multiproc"):
    os.makedirs("/tmp/prometheus_multiproc")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)


class PrometheusLogger(ls.Logger):
    """
    Enterprise-grade Prometheus metrics collector for combined service monitoring.

    Implements detailed performance tracking with multi-process support for production
    deployments. Provides high-resolution timing metrics for all service operations.

    Metrics:
        request_processing_seconds:
            - Type: Histogram
            - Labels: model_name, function_name
            - Description: Processing time per operation per model
    """

    def __init__(self):
        super().__init__()
        self.function_duration = Histogram(
            "request_processing_seconds",
            "Time spent processing request",
            ["model_name", "function_name"],
            registry=registry
        )

    def process(self, key: str, value: float) -> None:
        """
        Record a metric observation with operation-specific labeling.

        Args:
            key: Operation identifier in format "model_name:function_name"
            value: Duration measurement in seconds
        """
        if ":" in key:
            model_name, func_name = key.split(":", 1)
        else:
            model_name, func_name = "unknown", key

        self.function_duration.labels(
            model_name=model_name,
            function_name=func_name
        ).observe(value)


class CombinedVideoAPI(ls.LitAPI):
    """
    Combined Video Generation API for Mochi, LTX, Hunyuan, and Allegro models.
    This API handles requests in batch form, even for single items.

    Steps:
      1) setup(device): Initialize all sub-APIs on the specified device (CPU, GPU).
      2) decode_request(request): Parse the request body using Pydantic `CombinedBatchRequest`.
      3) predict(inputs): Parallel process each item in the batch.
      4) encode_response(outputs): Format the final JSON response.
    """

    def setup(self, device: str) -> None:
        """
        Called once at server startup.
        Initializes all model APIs on the same device.
        """
        logger.info(f"Initializing CombinedVideoAPI on device={device}")
        self.ltx_api = LTXVideoAPI()
        self.mochi_api = MochiVideoAPI()
        self.hunyuan_api = HunyuanVideoAPI()
        self.allegro_api = AllegroVideoAPI()

        self.ltx_api.setup(device=device)
        self.mochi_api.setup(device=device)
        self.hunyuan_api.setup(device=device)
        self.allegro_api.setup(device=device)

        self.model_apis = {
            "ltx": self.ltx_api,
            "mochi": self.mochi_api,
            "hunyuan": self.hunyuan_api,
            "allegro": self.allegro_api
        }

        logger.info("All sub-APIs (LTX, Mochi, Hunyuan, Allegro) successfully set up")

    def decode_request(self, request: Any) -> Dict[str, List[Dict[str, Any]]]:
        """
        Interprets the raw request body as a batch, then validates it.
        We unify single vs. multiple requests by requiring a `batch` array.

        Args:
            request: The raw request data (usually a dict from the body).

        Returns:
            A dictionary with key 'items' containing a list of validated dicts.

        Raises:
            ValidationError if the request doesn't match CombinedBatchRequest schema.
        """
        # If user directly posted an array, wrap it to match the expected schema
        if isinstance(request, list):
            request = {"batch": request}

        # Validate using CombinedBatchRequest
        validated_batch = CombinedBatchRequest(**request)

        # Convert each CombinedItemRequest into a dict for usage in predict
        items = [item.dict() for item in validated_batch.batch]
        return {"items": items}

    def predict(self, inputs: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Execute parallel inference for all items in the 'items' list.

        Args:
            inputs: Dictionary with key 'items' -> list of items.
                   Each item is a dict with fields like 'model_name', 'prompt', etc.

        Returns:
            List of generation results or error details
        """
        items = inputs["items"]
        logger.info(f"Processing batch of {len(items)} request(s) in parallel")
        results = []

        for item in items:
            try:
                start_time = time.time()
                model_name = item.get("model_name", "").lower()
                
                if model_name not in self.model_apis:
                    raise ValueError(f"Invalid model_name: {model_name}")

                sub_api = self.model_apis[model_name]
                sub_decoded = sub_api.decode_request(item)
                sub_pred = sub_api.predict(sub_decoded)
                
                if not sub_pred:
                    raise RuntimeError("No result returned from sub-API")
                
                end_time = time.time()
                result = {
                    "status": "success",
                    "model_name": model_name,
                    "generation_result": sub_pred[0],
                    "generation_params": item,
                    "time_taken": end_time - start_time
                }
                results.append(result)
                logger.info(f"Generation completed for model {model_name}, prompt: {item.get('prompt', '')}")
            
            except Exception as e:
                logger.error(f"Error in generation for model {model_name}: {e}")
                error_result = {
                    "status": "error",
                    "model_name": model_name,
                    "error": str(e)
                }
                results.append(error_result)
                
        return results if results else [{"status": "error", "error": "No results generated"}]

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
            
        if output.get("status") == "success":
            return {
                "status": "success",
                "model_name": output.get("model_name"),
                "video_path": output.get("generation_result", {}).get("video_path"),
                "generation_params": output.get("generation_params", {}),
                "time_taken": output.get("time_taken"),
                "metrics": {
                    "total_time": output.get("time_taken")
                }
            }
        else:
            return {
                "status": "error",
                "model_name": output.get("model_name", "unknown"),
                "error": output.get("error", "Unknown error occurred")
            }

def main():
    """
    Initialize and launch the combined video generation service.

    Sets up the complete service infrastructure including:
    - Prometheus metrics collection
    - Structured logging
    - API server configuration
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
        "logs/combined_api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )

    try:
        api = CombinedVideoAPI()
        server = litserve.LitServer(
            api,
            api_path='/api/v1/video/combined',
            accelerator="auto",
            devices="auto",
            max_batch_size=4,
            track_requests=True,
            loggers=[prometheus_logger],
            generate_client_file=False
        )
        logger.info("Starting combined video generation server on port 8000")
        server.run(port=8000)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
