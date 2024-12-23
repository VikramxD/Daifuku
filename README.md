<div align = "center">
<h2>Daifuku: A Sweet Way to Serve Multiple Text-to-Video Models </h2>
</div>



<div align="center">
  <img src = https://github.com/user-attachments/assets/ef233ca4-275a-4817-9042-60e53045821e width=250, height=250)>
  <pre>
     ✅ Multi-model T2V                 ✅ GPU offload & BF16
     ✅ Parallel batch processing       ✅ Prometheus metrics
     ✅ Docker-based deployment         ✅ Pydantic-based config
     ✅ S3 integration for MP4s         ✅ Minimal code, easy to extend
  </pre>

</div>

 
   



---

## Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Running the Servers](#running-the-servers)
- [Usage Examples](#usage-examples)
  - [Single Prompt Requests](#single-prompt-requests)
  - [Batch Requests](#batch-requests)
- [Features](#features)
- [Prompt Engineering](#prompt-engineering)
- [Docker Support](#docker-support)
- [Monitoring & Logging](#monitoring--logging)
- [License](#license)

---

## Introduction

**Daifuku** is a versatile framework designed to serve multiple Text-to-Video (T2V) models like **Mochi** and **LTX**. It simplifies T2V model deployment by providing:

- A unified API for multiple models.
- Support for parallel batch processing.
- GPU optimizations for efficiency.
- Easy Docker-based deployment.
- Integrated monitoring and logging.

Whether you're a developer or researcher, Daifuku is your go-to solution for scalable and efficient T2V workflows.

---

## Quick Start

### Installation

Follow these steps to set up Daifuku locally:

```bash
git clone https://github.com/YourUserName/Daifuku.git
cd Daifuku

# Create a virtual environment
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e . --no-build-isolation
```

**Optional:** Download Mochi weights for faster first use:

```bash
python scripts/download_weights.py
```

> **Note:** LTX weights download automatically on first usage.

### Running the Servers

Daifuku can serve models individually or combine them behind one endpoint:

<details>
<summary><strong>Mochi-Only Server</strong></summary>

```bash
python api/mochi_serve.py
# Endpoint: http://127.0.0.1:8000/api/v1/video/mochi
```

</details>

<details>
<summary><strong>LTX-Only Server</strong></summary>

```bash
python api/ltx_serve.py
# Endpoint: http://127.0.0.1:8000/api/v1/video/ltx
```

</details>

<details>
<summary><strong>Combined Server</strong></summary>

```bash
python api/serve.py
# Endpoint: http://127.0.0.1:8000/predict
# Specify "model_name": "mochi" or "model_name": "ltx" in the request payload.
```

</details>

---

## Usage Examples

### Single Prompt Requests

#### Mochi Model Example

```python
import requests

url = "http://127.0.0.1:8000/api/v1/video/mochi"
payload = {
    "prompt": "A serene beach at dusk, gentle waves, dreamy pastel colors",
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "height": 480,
    "width": 848,
    "num_frames": 120,
    "fps": 10
}

response = requests.post(url, json=payload)
print(response.json())
```

#### LTX Model Example

```python
import requests

url = "http://127.0.0.1:8000/api/v1/video/ltx"
payload = {
    "prompt": "A cinematic scene of autumn leaves swirling around the forest floor",
    "negative_prompt": "blurry, worst quality",
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "height": 480,
    "width": 704,
    "num_frames": 121,
    "frame_rate": 25
}

response = requests.post(url, json=payload)
print(response.json())
```

### Batch Requests

Process multiple requests simultaneously with Daifuku's parallel capabilities:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "batch": [
      {
        "model_name": "mochi",
        "prompt": "A calm ocean scene, sunrise, realistic",
        "num_inference_steps": 40
      },
      {
        "model_name": "ltx",
        "prompt": "A vintage film style shot of the Eiffel Tower",
        "height": 480,
        "width": 704
      }
    ]
  }'
```

---

## Features

1. **Multi-Model T2V**
   - Serve **Mochi** or **LTX** individually or unify them under one endpoint.

2. **Parallel Batch Processing**
   - Handle multiple requests concurrently for high throughput.

3. **GPU Optimizations**
   - Features like BF16 precision, attention slicing, and VAE tiling for efficient GPU use.

4. **Prometheus Metrics**
   - Monitor request latency, GPU usage, and more.

5. **S3 Integration**
   - Automatically upload `.mp4` files to Amazon S3 and return signed URLs.

6. **Pydantic Config**
   - Configurable schemas for Mochi (`mochi_settings.py`), LTX (`ltx_settings.py`), and combined setups.

7. **Advanced Logging**
   - Uses [Loguru](https://github.com/Delgan/loguru) for detailed and structured logging.

---

## Prompt Engineering

- **Mochi**:
  - Optimized for creative and artistic prompts.
  - Recommended: ~50 steps, guidance scale 4.0–7.5, resolution up to 768×768.

- **LTX**:
  - Ideal for cinematic or photo-realistic scenes.
  - Recommended: Height/width multiples of 32, frame counts like `8n+1` (e.g., 121, 161), guidance scale ~3.0.

---

## Docker Support

Daifuku provides a [**Dockerfile**](./DockerFileFolder/Dockerfile) for streamlined deployment:

```bash
docker build -t daifuku -f DockerFileFolder/Dockerfile .
docker run --gpus all -p 8000:8000 daifuku
```

### Customization
Modify the `CMD` in the Dockerfile to switch between Mochi, LTX, or combined server modes.

---

## Monitoring & Logging

### Prometheus Metrics

Key metrics include:

- GPU memory usage (allocated & peak).
- Inference duration (histogram).
- Request throughput.

Endpoints:

- Mochi: `/api/v1/metrics`
- LTX: `/api/v1/metrics`
- Combined: `/metrics`

### Loguru Logging

- Logs rotate at **100 MB** and retain up to **1 week**.
- Find logs in:
  - `logs/api.log` (Mochi)
  - `logs/ltx_api.log` (LTX)
  - `logs/combined_api.log`

---

## License

Daifuku is licensed under the [MIT License](./LICENSE).

