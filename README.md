<div align="center">
   <img src="https://github.com/user-attachments/assets/ea97ff3a-39b3-418a-a62c-5687e7222117" alt="Cute Mochi Logo" width="200" height="200">
   <h1>MinMochi</h1>
   <h3>Minimalist API Server for Mochi and LTX Text-to-Video Generation</h3>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Torch 2.0+](https://img.shields.io/badge/torch-2.0%2B-orange.svg)](https://pytorch.org/)
</div>

## 🚀 Overview

**MinMochi** serves both the Genmo Mochi and Lightricks LTX text-to-video models as a production-ready API. Generate high-quality videos from text prompts with minimal setup.

## 🛠️ System Requirements

- 🐍 Python 3.10+
- 🎮 GPU Requirements:
  - Recommended: NVIDIA A100 or H100
  - Minimum: NVIDIA A6000 or A40
  - Mochi: 16GB+ VRAM
  - LTX: 24GB+ VRAM
- ☁️ Active AWS account
- 🐳 Docker

## 📦 Installation

```bash
# Get the code
git clone https://github.com/VikramxD/Minimochi
cd minimochi

# Set up environment
pip install uv
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-build-isolation
```

## ⚙️ Configuration

MinMochi uses Pydantic settings for configuration management. The configuration is split into multiple modules:

### 1. Mochi Settings (`mochi_settings.py`)
```python
# Default settings, can be overridden with MOCHI_ prefixed env variables
model_name = "Genmo-Mochi"
transformer_path = "imnotednamode/mochi-1-preview-mix-nf4"
pipeline_path = "VikramxD/mochi-diffuser-bf16"
dtype = torch.bfloat16
device = "cuda"

# Optimization Settings
enable_vae_tiling = True
enable_model_cpu_offload = True
enable_attention_slicing = False

# Video Generation Settings
num_inference_steps = 20
guidance_scale = 7.5
height = 480
width = 848
num_frames = 150
fps = 10
```

### 2. LTX Settings (`ltx_settings.py`)
```python
# Default settings, can be overridden with LTX_ prefixed env variables
model_name = "LTX-Video"
ckpt_dir = "checkpoints"  # Directory containing model components
device = "cuda"

# Video Generation Settings
num_inference_steps = 40
guidance_scale = 3.0
height = 480
width = 704
num_frames = 121
frame_rate = 25
```

### 3. AWS Settings (`aws_settings.py`)
```python
# Override with environment variables
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = "ap-south-1"
AWS_BUCKET_NAME = "diffusion-model-bucket"
```

## 🎨 Prompt Engineering Guide

### For LTX Model
Structure your prompts focusing on cinematic details:
1. Start with main action
2. Add specific movement details
3. Describe visual elements precisely
4. Include environment details
5. Specify camera angles
6. Describe lighting and colors

Example LTX Prompt:
```
A red maple leaf slowly falls through golden autumn sunlight in a serene forest. The leaf twirls and dances as it descends, casting delicate shadows. Sunbeams filter through trees, creating a warm, dappled lighting effect. The camera follows the leaf in a gentle downward tracking shot.
```

Parameter Guidelines (LTX):
- Resolution: Must be divisible by 32 (e.g., 480x704)
- Frames: Must follow pattern 8n+1 (e.g., 121, 161)
- Guidance Scale: 3.0-3.5 recommended
- Steps: 40+ for quality, 20-30 for speed

## 🎬 Usage

### Launch Servers

```bash
# Launch Mochi Server
python3 api/mochi_serve.py

# Launch LTX Server
python api/ltx_serve.py
```

### Generate Videos

#### Mochi API
```python
url = "http://localhost:8000/api/v1/video/mochi"
payload = {
    "prompt": "A beautiful sunset over the mountains",
    "num_inference_steps": 100,
    "guidance_scale": 7.5,
    "height": 480,
    "width": 848,
    "num_frames": 150,
    "fps": 10
}
```

#### LTX API
```python
url = "http://localhost:8000/api/v1/video/ltx"
payload = {
    "prompt": "A red maple leaf slowly falls...",
    "negative_prompt": "worst quality, inconsistent motion, blurry",
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "height": 480,
    "width": 704,
    "num_frames": 121,
    "frame_rate": 25,
    "seed": 42
}

response = requests.post(url, json=payload)
print(response.json())
```

### CURL Example (LTX)
```bash
curl -X POST http://localhost:8000/api/v1/video/ltx \
-H "Content-Type: application/json" \
-d '{
    "prompt": "A red maple leaf slowly falls...",
    "height": 480,
    "width": 704,
    "num_frames": 121,
    "num_inference_steps": 40,
    "guidance_scale": 3.0
}'
```

## 📊 Monitoring

### Metrics
Prometheus metrics available at `/metrics`:
- Request processing time
- GPU memory usage
- Inference time

### Logging
- Structured logging with loguru
- Log rotation at 100MB
- 1-week retention period
- Logs stored in `logs/api.log` and `logs/ltx_api.log`

## 🎛️ GPU Memory Requirements

### Mochi Model
| Resolution | Frames | Min GPU Memory |
|------------|--------|----------------|
| 480x480 | 60 | 16GB |
| 576x576 | 60 | 20GB |
| 768x768 | 60 | 24GB |

### LTX Model
| Resolution | Frames | Min GPU Memory |
|------------|--------|----------------|
| 480x704 | 121 | 24GB |
| 576x832 | 121 | 32GB |
| 720x1280 | 121 | 40GB |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Genmo.ai](https://genmo.ai) for the Mochi model
- [Lightricks](https://www.lightricks.com/) for the LTX-Video model
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [LitServe](https://github.com/Lightning-AI/litserve) - API framework

---

<div align="center">
Made with ❤️ by VikramxD
</div>
