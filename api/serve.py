"""
combined_serve.py

A unified API for both LTX and Mochi video generation using batch-based parallel processing.
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
from configs.combined_settings import CombinedBatchRequest, CombinedItemRequest
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc"
if not os.path.exists("/tmp/prometheus_multiproc"):
    os.makedirs("/tmp/prometheus_multiproc")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)


class PrometheusLogger(ls.Logger):
    """
    Custom logger for Prometheus metrics.
    Tracks request duration for each (model_name, function_name) pair.
    """

    def __init__(self):
        super().__init__()
        self.function_duration = Histogram(
            "combined_request_duration_seconds",
            "Time spent processing video generation requests",
            ["model_name", "function_name"],
            registry=registry
        )

    def process(self, key: str, value: float) -> None:
        """
        Record metric observations with labels for both model_name and function_name.
        `key` is expected to have the format "model_name:function_name".
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
    Combined Video Generation API for both LTX and Mochi models.
    This API handles requests in batch form, even for single items.

    Steps:
      1) setup(device): Initialize LTX and Mochi sub-APIs on the specified device (CPU, GPU).
      2) decode_request(request): Parse the request body using Pydantic `CombinedBatchRequest`.
      3) predict(inputs): Parallel process each item in the batch.
      4) encode_response(outputs): Format the final JSON response.
    """

    def setup(self, device: str) -> None:
        """
        Called once at server startup.
        Initializes both the LTX and Mochi APIs on the same device.
        """
        logger.info(f"Initializing CombinedVideoAPI on device={device}")
        self.ltx_api = LTXVideoAPI()
        self.mochi_api = MochiVideoAPI()

        self.ltx_api.setup(device=device)
        self.mochi_api.setup(device=device)

        self.model_apis = {
            "ltx": self.ltx_api,
            "mochi": self.mochi_api
        }

        logger.info("All sub-APIs (LTX, Mochi) successfully set up")

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

    def predict(self, inputs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Execute parallel inference for all items in the 'items' list.

        Args:
            inputs: Dictionary with key 'items' -> list of items. 
                    Each item is a dict with fields like 'model_name', 'prompt', etc.

        Returns:
            Dictionary with 'batch_results': a list of output dicts, 
            each containing status, video_id, video_url, etc.
        """
        items = inputs["items"]
        logger.info(f"Processing batch of {len(items)} request(s) in parallel")

        # We'll define a helper function for one item
        def process_single(item: Dict[str, Any]) -> Dict[str, Any]:
            """
            Takes a single request dict, delegates to the correct sub-API (LTX or Mochi).
            Returns the predicted result (video URL, etc.).
            """
            model_name = item.get("model_name", "").lower()
            if model_name not in self.model_apis:
                return {
                    "status": "error",
                    "error": f"Invalid model_name: {model_name}"
                }

            sub_api = self.model_apis[model_name]

            # Sub-API workflow: decode -> predict -> single result
            # Note: sub_api.decode_request() often returns a list. We'll handle that carefully.
            try:
                # Prepare sub-request in their expected format
                sub_decoded = sub_api.decode_request(item)
                sub_pred = sub_api.predict(sub_decoded)
                return sub_pred[0] if sub_pred else {
                    "status": "error", 
                    "error": "No result returned from sub-API."
                }
            except Exception as e:
                logger.error(f"[{model_name}] sub-api error: {e}")
                return {"status": "error", "error": str(e), "model_name": model_name}

        # Use a ProcessPoolExecutor to handle CPU-heavy tasks concurrently
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_idx = {}
            for idx, item in enumerate(items):
                future = executor.submit(process_single, item)
                future_to_idx[future] = idx

            for f in as_completed(future_to_idx):
                idx = future_to_idx[f]
                try:
                    out = f.result()
                    out["item_index"] = idx
                    if "model_name" not in out:
                        out["model_name"] = items[idx].get("model_name", "unknown")
                    results.append(out)
                except Exception as e:
                    # If something catastrophic happened in process_single
                    results.append({"status": "error", "error": str(e), "item_index": idx})

        # Sort results by item_index so response order matches input order
        results.sort(key=lambda x: x["item_index"])
        return {"batch_results": results}

    def encode_response(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the raw dictionary from `predict` into a final response.
        We unify single vs. multiple items:
          - The client always receives "batch_results" 
            (with 1 result if originally a single item).

        Sub-APIs often have their own encode_response() method to standardize the final JSON.
        We'll call that to keep consistent format.

        Returns:
            The final JSON-serializable dict.
        """
        if "batch_results" not in outputs:
            return {
                "status": "error", 
                "error": "No batch_results field found in predict output"
            }

        for item in outputs["batch_results"]:
            if item.get("status") == "error":
                continue

            model_name = item.get("model_name", "").lower()
            if model_name in self.model_apis:
                sub_encoded = self.model_apis[model_name].encode_response(item)
                item.update(sub_encoded)

        return outputs


def main():
    """
    Main entry point for the combined server, exposing /predict on port 8000.
    This version logs metrics to Prometheus and logs to console + file.
    """
    from litserve import LitServer

    # PROMETHEUS LOGGER
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(
        path="/metrics",
        app=make_asgi_app(registry=registry)
    )

    # LOGGING
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
               "| <level>{level: <8}</level> "
               "| <cyan>{name}</cyan>:<cyan>{function}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/combined_api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )

    api = CombinedVideoAPI()
    server = LitServer(
        api,
        api_path="/api/v1/inference",
        accelerator="auto",  
        devices="auto",      
        max_batch_size=4,    
        track_requests=True,
        loggers=[prometheus_logger]
    )

    logger.info("Starting combined video generation server on port 8000")
    server.run(port=8000)


if __name__ == "__main__":
    main()
