

<div align="center">
   <h1> MiniMochi </h1>
   <img src="https://github.com/user-attachments/assets/ea97ff3a-39b3-418a-a62c-5687e7222117" alt="Cute Mochi Logo" width="200" height="200">

### Serve Genmo\.ai Mochi Model in Production

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red.svg?style=flat-square)](https://github.com/VikramxD/minimochi)

[⚙️ Installation](#%EF%B8%8F-installation) • 
[🎮 Usage](#-usage) • 
[📖 Docs](#-documentation)

</div>

---

## ✨ Overview

**MinMochi** serves the Mochi text-to-video model as an API. Generate high-quality videos from text prompts with minimal setup\.

## 📋 Prerequisites

Before diving in, ensure you have:
- 🐍 Python 3\.10 or higher
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
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)
```

## 📖 Documentation

### Available Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/api/v1/video/mochi` | POST | Generate video | Required |
| `/health` | GET | Service health | Optional |

### Configuration Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|--------|
| prompt | str | Text description of desired video | Required | - |
| negative_prompt | str | What to avoid in generation | Optional | - |
| num_frames | int | Frame count | 60 | 30-120 |
| fps | int | Frames per second | 30 | 15-60 |
| height | int | Video height | 480 | 256-480 |
| width | int | Video width | 640 | 256-840 |
| num_inference_steps | int | Generation quality steps | 50 | 1-100 |
| guidance_scale | float | Prompt adherence strength | 7\.5 | 1-20 |


## 🔧 Advanced Configuration

### Environment Variables

Create a `.env` file in the project root:

```plaintext
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_preferred_region
AWS_BUCKET_NAME=your_s3_bucket
```

### GPU Memory Requirements

| Resolution | Frames | Min GPU Memory |
|------------|--------|----------------|
| 480x480 | 60 | 16GB |
| 576x576 | 60 | 20GB |
| 768x768 | 60 | 24GB |
| 1024x1024 | 60 | 40GB |





## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details\.

## 🙏 Acknowledgments

- [Genmo\.ai](https://github.com/genmoai) - For the original Mochi model
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) 
- [LitServe](https://github.com/Lightning-AI/litserve)

---

<div align="center">
[Report Bug](https://github.com/VikramxD/minimochi/issues) • [Request Feature](https://github.com/VikramxD/minimochi/issues)
</div>
