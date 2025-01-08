
<div align="center">
<h2>Daifuku: A Sweet Way to Serve Multiple Text-to-Video Models</h2>
</div>

<div align="center">
  <p></p>
  <img src="https://github.com/user-attachments/assets/ef233ca4-275a-4817-9042-60e53045821e" width="250" height="250" alt="Daifuku Logo" />
  <p></p>

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
- [Monitoring](#monitoring--logging)
- [License](#license)

---

## Introduction

**Daifuku** is a versatile framework designed to serve multiple Text-to-Video (T2V) models (e.g., **Mochi**, **LTX**, and more). It streamlines T2V model deployment by providing:

- A unified API for multiple models  
- Parallel batch processing  
- GPU optimizations for efficiency  
- Easy Docker-based deployment  
- Integrated monitoring, logging, and metrics  

Inspired by the concept of *daifuku mochi*—a sweet stuffed treat—this framework “stuffed” with multiple T2V capabilities aims to make your video generation *as sweet and satisfying as possible*.

---

## Quick Start

### Installation

```bash
git clone https://github.com/VikramxD/Daifuku.git
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

Daifuku can serve models individually or combine them behind a single endpoint:

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
# Must supply "model_name": "mochi" or "model_name": "ltx" in the request payload.
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

Process multiple requests simultaneously with Daifuku’s parallel capabilities:

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
   Serve **Mochi** or **LTX** individually, or unify them under one endpoint.

2. **Parallel Batch Processing**  
   Handle multiple requests concurrently for high throughput.

3. **GPU Optimizations**  
   BF16 precision, attention slicing, VAE tiling, CPU offload, etc.

4. **Prometheus Metrics**  
   Monitor request latency, GPU usage, and more.

5. **S3 Integration**  
   Automatically upload `.mp4` files to Amazon S3 and return signed URLs.

6. **Pydantic Config**  
   Configurable schemas for Mochi (`mochi_settings.py`), LTX (`ltx_settings.py`), and combined setups.

7. **Advanced Logging**  
   Uses [Loguru](https://github.com/Delgan/loguru) for detailed and structured logging.

---

## Prompt Engineering

### Mochi 1 Prompt Engineering Guide

Daifuku currently ships with **Genmo’s Mochi model** as one of the primary text-to-video generation options. **Crafting effective prompts** is crucial to producing high-quality, consistent, and predictable results. Below is a product-management-style guide with detailed tips and illustrative examples:

#### 1. Goal-Oriented Prompting

**Ask yourself:** What is the end experience or visual story you want to convey?
- Example: “I want a short clip showing a hand gently picking up a lemon and rotating it in mid-air before placing it back.”
- Pro Tip: Write prompts with the final user experience in mind—like describing a scene for a storyboard.

---

#### 2. Technical Guidelines

1. **Precise Descriptions**  
   - Include motion verbs and descriptors (e.g., “gently tosses,” “rotating,” “smooth texture”).
   - Use specifics for objects (e.g., “a bright yellow lemon in a wooden bowl”).

2. **Scene Parameters**  
   - Define environment details: lighting (soft sunlight, tungsten glow), camera position (top-down, eye-level), and any background elements.
   - Focus on how these details interact (e.g., “shadows cast by the overhead lamp moving across the marble table”).

3. **Motion Control**  
   - Specify movement timing or speed (e.g., “the camera pans at 0.3m/s left to right,” “the object rotates 90° every second”).
   - For multi-step actions, break them down into time-coded events (e.g., “t=1.0s: the hand appears, t=2.0s: the hand gently tosses the lemon...”).

4. **Technical Parameters**  
   - Provide explicit numeric values for lighting conditions or camera angles (e.g., “5600K color temperature,” “f/2.8 aperture,” “ISO 400”).
   - If controlling atmospheric or environmental effects (e.g., fog density, volumetric lighting), add them as key-value pairs for clarity.

---

#### 3. Reference Prompts

Below are **extended** examples showing how you can move from a simple directive to a fully descriptive, technical prompt.

<details>
<summary><strong>Example 1: Controlled Motion Sequence</strong></summary>

- **Simple Prompt**:
  ```
  PRECISE OBJECT MANIPULATION
  ```

- **Detailed Prompt**:
  ```
  A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and fresh mint sprigs against a peach-colored background.
  The hand gently tosses the lemon up and catches it mid-air, highlighting its smooth texture.
  A beige string bag rests beside the bowl, adding a rustic touch.
  Additional lemons, including one halved, are scattered around the bowl’s base.
  Even, diffused lighting accentuates vibrant colors, creating a fresh, inviting atmosphere.
  Motion Sequence:
  - t=0.0 to 0.5s: Hand enters from left
  - t=1.0 to 1.2s: Lemon toss in slow motion
  - t=1.2 to 2.0s: Hand exits, camera remains static
  ```

**Why It Works**  
- Provides both **visual** (color, environment) and **temporal** (timing, motion) details.  
- Mentions **lighting** explicitly for consistent results.  
- The final action is clearly staged with micro-timings.

</details>

<details>
<summary><strong>Example 2: Technical Scene Setup</strong></summary>

- **Simple Prompt**:
  ```
  ARCHITECTURAL VISUALIZATION
  ```

- **Detailed Prompt**:
  ```
  Modern interior space with precise lighting control.
  The camera tracks laterally at 0.5m/s, maintaining a 1.6m elevation from the floor.
  Natural light at 5600K color temperature casts dynamic shadows across polished surfaces,
  while secondary overhead lighting at 3200K adds a warm glow.
  The scene uses soft ambient occlusion for depth,
  and focus remains fixed on the primary subject: a minimalist white sofa placed near full-height windows.
  ```

**Why It Works**  
- Encourages a **photo-realistic** interior shot.  
- Combines color temperature specifics and motion parameters for consistent lighting and camera movement.

</details>

<details>
<summary><strong>Example 3: Environmental Control</strong></summary>

- **Simple Prompt**:
  ```
  ATMOSPHERIC DYNAMICS
  ```

- **Detailed Prompt**:
  ```
  Volumetric lighting with carefully controlled particle density.
  The camera moves upward at 0.3m/s, starting at ground level and ending at 2.0m elevation.
  Light scatter coefficient: 0.7, atmospheric transmission: 85%.
  Particles glisten under a single overhead spotlight, forming dynamic light beams.
  The scene remains in gentle motion, focusing on drifting dust motes that convey a dreamy atmosphere.
  ```

**Why It Works**  
- **Volumetric** and **particle** details reinforce a cinematic environment.  
- Inclusion of **scatter** and **transmission** values shapes a more consistent outcome.

</details>

---

#### 4. Prompt Architecture

1. **Scene Configuration**  
   Define core environmental parameters, e.g.  
   *"Interior setting, 5600K color temperature, f/4 aperture"*

2. **Motion Parameters**  
   Specify camera or object movements, e.g.  
   *"Camera tracks at 0.5m/s, 1.6m elevation"*

3. **Lighting Setup**  
   Detail lighting conditions, e.g.  
   *"Natural sunlight from east windows, overhead tungsten fill at 3200K"*

4. **Temporal Flow**  
   Outline time-coded actions, e.g.  
   *"Action sequence: t=0.0–0.8s approach, t=1.0–2.0s main interaction"*

---

#### 5. Advanced Techniques

- **Use precise numerical values**: Encourages the model to maintain consistent shapes, motions, and lighting across frames.
- **Incorporate scientific or cinematographic parameters**: e.g., specifying “diffuse reflectivity at 0.3,” or “shutter speed 1/60s.”
- **Define exact measurements** for spatial relationships: e.g. “The table is 1m wide, with objects placed 0.25m apart.”
- **Acknowledge model limitations**: If you see repeated artifacts, simplify the scene or reduce complex geometry references.
- **Aim for photorealism** over extreme fantasy: The more physically plausible your prompt, the more stable the outcome.

---
### LTX Video Prompt Engineering Guide

#### 📝 Prompt Engineering

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

* Start with main action in a single sentence
* Add specific details about movements and gestures
* Describe character/object appearances precisely
* Include background and environment details
* Specify camera angles and movements
* Describe lighting and colors
* Note any changes or sudden events


#### 🎮 Parameter Guide

* Resolution Preset: Higher resolutions for detailed scenes, lower for faster generation and simpler scenes. The model works on resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1 (e.g. 257). In case the resolution or number of frames are not divisible by 32 or 8 + 1, the input will be padded with -1 and then cropped to the desired resolution and number of frames. The model works best on resolutions under 720 x 1280 and number of frames below 257
* Seed: Save seed values to recreate specific styles or compositions you like
* Guidance Scale: 3-3.5 are the recommended values
* Inference Steps: More steps (40+) for quality, fewer steps (20-30) for speed

## Docker Support

Daifuku provides a [**Dockerfile**](./DockerFileFolder/Dockerfile) for streamlined deployment:

```bash
docker build -t daifuku -f DockerFileFolder/Dockerfile .
docker run --gpus all -p 8000:8000 daifuku
```




### Customization

Modify the `CMD` in the Dockerfile to switch between Mochi, LTX, or a combined server mode.

---

## Monitoring & Logging

### Prometheus Metrics

Key metrics include:

- **GPU memory usage** (allocated & peak)  
- **Inference duration** (histogram)  
- **Request throughput**  

Endpoints:
- **Mochi**: `/api/v1/metrics`
- **LTX**: `/api/v1/metrics`
- **Combined**: `/metrics`

### Loguru Logging

- Logs rotate at **100 MB** and are retained for **1 week**.
- Find logs in:
  - `logs/api.log` (Mochi)
  - `logs/ltx_api.log` (LTX)
  - `logs/combined_api.log` (Combined)

---

## License

Daifuku is licensed under the [MIT License](./LICENSE).

## Star History

<a href="https://star-history.com/#VikramxD/Daifuku">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=VikramxD/Daifuku&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=VikramxD/Daifuku" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=VikramxD/Daifuku" />
 </picture>
</a>

