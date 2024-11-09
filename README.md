<div align="center">
   <img src="https://github.com/user-attachments/assets/ea97ff3a-39b3-418a-a62c-5687e7222117" alt="Cute Mochi Logo" width="200" height="200">
   <h1>MinMochi</h1>
   <h3>Minimalist API Server for Mochi Text-to-Video Generation</h3>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Torch 2.0+](https://img.shields.io/badge/torch-2.0%2B-orange.svg)](https://pytorch.org/)
</div>

## 🚀 Overview

**MinMochi** serves the Genmo Mochi text-to-video model as a production-ready API. Generate high-quality videos from text prompts with minimal setup.

## 🛠️ System Requirements

- 🐍 Python 3.10+
- 🎮 GPU Requirements:
  - Recommended: NVIDIA A100 or H100
  - Suitable: NVIDIA A6000 or A40
- ☁️ Active AWS account
- 🐳 Docker

## 📦 Installation

```bash
# Get the code
git clone [your-repo-url]
cd minimochi

# Set up environment
pip install uv
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-build-isolation
```

## ⚙️ Configuration

MinMochi uses Pydantic settings for configuration management. The configuration is split into three main modules:

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

### 2. AWS Settings (`aws_settings.py`)
```python
# Override with environment variables
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = "ap-south-1"
AWS_BUCKET_NAME = "diffusion-model-bucket"
```

### 3. Model Weights Settings (`mochi_weights.py`)
```python
output_dir = Path("weights")
repo_id = "genmo/mochi-1-preview"
model_file = "dit.safetensors"
decoder_file = "decoder.safetensors"
encoder_file = "encoder.safetensors"
dtype = "bf16"  # Options: "fp16", "bf16"
```

## 🎬 Usage

### Launch Server

```bash
python src/api/mochi_serve.py
```

### Generate Videos

```python
import requests
import json

url = "http://localhost:8000/api/v1/video/mochi"
payload = {
    "prompt": "A beautiful sunset over the mountains",
    "negative_prompt": "",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "height": 480,
    "width": 848,
    "num_frames": 150,
    "fps": 10
}

response = requests.post(url, json=[payload])
print(response.json())
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
- Logs stored in `logs/api.log`

## 🎛️ GPU Memory Requirements

| Resolution | Frames | Min GPU Memory |
|------------|--------|----------------|
| 480x480 | 60 | 16GB |
| 576x576 | 60 | 20GB |
| 768x768 | 60 | 24GB |
| 1024x1024 | 60 | 40GB |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Genmo.ai](https://genmo.ai) for the original Mochi model
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [LitServe](https://github.com/Lightning-AI/litserve) - API framework

---

<div align="center">

[Report Bug](https://github.com/your-repo/minimochi/issues) • [Request Feature](https://github.com/your-repo/minimochi/issues)

</div>
