

# 🍡 Minimochi

### Serving Genmo.ai Mochi Model in production

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red.svg?style=flat-square)](https://github.com/VikramxD/minimochi)

[📚 Features](#-features) • 
[⚙️ Installation](#%EF%B8%8F-installation) • 
[🎮 Usage](#-usage) • 
[📖 Docs](#-documentation) • 
[🤝 Contribute](#-testing)

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
from typing import Optional, Dict, Any

def generate_video(
    prompt: str,
    frames: int = 60,
    fps: int = 30,
    height: int = 480,
    width: int = 640
) -> Optional[Dict[str, Any]]:
    """
    Generate a video from a text prompt.
    
    Args:
        prompt (str): Text description of the desired video
        frames (int): Number of frames to generate
        fps (int): Frames per second
        height (int): Video height in pixels
        width (int): Video width in pixels
    
    Returns:
        Optional[Dict[str, Any]]: Response data or None on error
    """
    url = "http://localhost:8000/api/v1/mochi"
    
    payload = {
        "prompt": prompt,
        "num_frames": frames,
        "fps": fps,
        "height": height,
        "width": width
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None

# Example
result = generate_video(
    prompt="A serene mountain landscape with a flowing river at sunset"
)
print(result)
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

