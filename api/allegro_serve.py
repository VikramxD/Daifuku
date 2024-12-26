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

from configs.allegro_settings import AllegroSettings
from scripts.allegro_diffusers import AllegroInference
from scripts.mp4_to_s3_json import mp4_to_s3_json
import torch

# Set up prometheus multiprocess mode
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

# Initialize prometheus registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class PrometheusLogger(Logger):
    """Custom logger for Prometheus metrics.
    
    Implements metric collection for request processing times
    using Prometheus Histograms.
    Metrics are stored in a multi-process compatible registry.
    
    Attributes:
        function_duration (Histogram): Prometheus histogram for tracking processing times
    """

    def __init__(self):
        super().__init__()
        self.function_duration = Histogram(
            "allegro_request_processing_seconds",
            "Time spent processing Allegro request",
            ["function_name"],
            registry=registry
        )

    def process(self, key: str, value: float) -> None:
        """Process and record a metric value.
        
        Args:
            key (str): The name of the function or operation being measured
            value (float): The duration or metric value to record
        """
        self.function_duration.labels(function_name=key).observe(value)

class AllegroRequest(BaseModel):
    """Model representing a request for the Allegro model.
    
    Validates input parameters for Allegro model inference.
    
    Attributes:
        prompt (str): Text prompt for inference
        negative_prompt (Optional[str]): Text prompt for elements to avoid
        num_inference_steps (int): Number of denoising steps (1-100)
        guidance_scale (float): Controls adherence to prompt (1.0-20.0)
        height (int): Image height (256-720, multiple of 32)
        width (int): Image width (256-1280, multiple of 32)
        seed (Optional[int]): Random seed for reproducibility
    """
    prompt: str = Field(..., description="Main text prompt for generation")
    negative_prompt: Optional[str] = Field(
        "worst quality, blurry, distorted",
        description="Text description of what to avoid"
    )
    num_inference_steps: int = Field(50, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    height: int = Field(512, ge=256, le=720, multiple_of=32, description="Image height")
    width: int = Field(512, ge=256, le=1280, multiple_of=32, description="Image width")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class AllegroAPI(LitAPI):
    """API implementation for Allegro model inference using LitServe.
    
    Attributes:
        settings (AllegroSettings): Configuration for Allegro model
        engine (AllegroInference): Inference engine for Allegro
    """

    def setup(self, device: str) -> None:
        """Initialize the Allegro inference engine.
        
        Args:
            device (str): Target device for inference ('cuda', 'cpu', etc.)
        """
        try:
            logger.info(f"Initializing Allegro model on device: {device}")
            self.settings = AllegroSettings(device=device)
            self.engine = AllegroInference(self.settings)
            logger.info("Allegro setup completed successfully")
        except Exception as e:
            logger.error(f"Error during Allegro setup: {e}")
            raise

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode and validate the incoming request.
        
        Args:
            request (dict): Input request dictionary
        
        Returns:
            dict: Validated request
        """
        try:
            return AllegroRequest(**request).dict()
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            raise

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform inference using the Allegro model.
        
        Args:
            inputs (list): List of validated requests
        
        Returns:
            list: Results with URLs and metadata
        """
        results = []
        for request in inputs:
            start_time = time.time()
            try:
                self.settings.update(request)
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / "output.mp4"
                    self.settings.output_path = output_path
                    self.engine.generate()

                    if not output_path.exists():
                        raise FileNotFoundError(f"Output not found at {output_path}")

                    with open(output_path, 'rb') as video_file:
                        s3_response = mp4_to_s3_json(video_file, output_path.name)

                    generation_time = time.time() - start_time

                    results.append({
                        "status": "success",
                        "video_url": s3_response["url"],
                        "prompt": request["prompt"],
                        "time_taken": generation_time
                    })

            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                results.append({"status": "error", "error": str(e)})
        return results

    def encode_response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encode the results into a response format.
        
        Args:
            output (list): Results list
        
        Returns:
            dict: Encoded response
        """
        return {"results": output}

def main():
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(
        path="/api/v1/metrics",
        app=make_asgi_app(registry=registry)
    )

    logger.remove()
    logger.add(sys.stdout, format="<green>{time}</green> | <level>{message}</level>", level="INFO")
    logger.add("logs/error.log", format="<red>{time}</red> | <level>{message}</level>", level="ERROR")

    try:
        api = AllegroAPI()
        server = LitServer(
            api,
            api_path='/api/v1/allegro',
            accelerator="auto",
            devices="auto",
            max_batch_size=4,
            loggers=prometheus_logger,
        )

        logger.info("Starting Allegro API server on port 8000")
        server.run(port=8000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)



if __name__ == "__main__":
    main()
