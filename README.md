

# 🍡 Minimochi

### Serving Genmo.ai Mochi Model in production

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red.svg?style=flat-square)](https://github.com/VikramxD/minimochi)

[⚙️ Installation](#%EF%B8%8F-installation) • 
[🎮 Usage](#-usage) • 
[📖 Docs](#-documentation) • 


---

</div>

## ✨ Overview

**Minimochi** is your gateway to next-generation video creation. Powered by state-of-the-art diffusion models, our API transforms simple text descriptions into captivating video content. Whether you're a developer, content creator, or researcher, Minimochi provides the tools you need for seamless AI video integration.


## 📋 Prerequisites

Before diving in, ensure you have:

- 🐍 Python 3.10 or higher
- 🎮 GPU Requirements:
  - Recommended: NVIDIA A100 or H100
  - Suitable: NVIDIA A6000 or A40
- ☁️ Active AWS account
- 🐳 Docker 

## 🛠️ Installation

### Quick Start

```bash
# Get the code
git clone https://github.com/VikramxD/minimochi.git
cd minimochi

# Set up environment
pip install uv
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-build-isolation
```


## 🎬 Usage

### Launch Server

```bash
python src/api/mochi_serve.py
```

### Create Videos

```python
import requests
import json

url = "http://localhost:8000/api/v1/video/mochi"

payload = json.dumps([
  {
    "prompt": "A beautiful sunset over the mountains",
    "negative_prompt": "No clouds",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "height": 480,
    "width": 480,
    "num_frames": 150,
    "fps": 10
  }
])
headers = {
  'Content-Type': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

```

## 📖 Documentation

### Available Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/api/v1/video/mochi'` | POST | Generate video | Required |
| `/status` | GET | Service health | Optional |


### Configuration Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|--------|
| num_frames | int | Frame count | 60 | 30-120 |
| fps | int | Frames per second | 30 | 15-60 |
| height | int | Video height | 480 | 256-1024 |
| width | int | Video width | 640 | 256-1024 |




## 🙏 Acknowledgments

- [Genmo.ai](https://github.com/genmoai)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [LitServe](https://github.com/Lightning-AI/litserve) 
---

