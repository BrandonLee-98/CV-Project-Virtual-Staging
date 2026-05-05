# RE.Ai Solution: Intelligent Virtual Staging Pipeline

**Institution:** Houston City College  
**Course:** AI and Robotics - Computer Vision Capstone  
**Developer:** Brandon Matias


## Project Overview
RE.Ai Solution is a high-fidelity virtual staging pipeline designed to transform vacant real estate photography into fully furnished, market-ready assets. Unlike standard generative tools, this system utilizes a **dual-pass Computer Vision architecture** to ensure architectural integrity and spatial realism.

The pipeline integrates **YOLO-World** (Zero-Shot Object Detection) for environmental context and the **SeeDream 4.5** diffusion model for high-resolution 2K staging.

---

## Key Technical Features

### 1. Zero-Shot Architectural Anchor Detection
Standard YOLO models often fail to recognize permanent fixtures in empty rooms. This project implements **YOLO-World** to detect custom vocabulary anchors:
* **Fireplaces & Ceiling Fans:** Used as "architectural anchors" to correctly identify room types (e.g., Living Room vs. Bedroom) when furniture is absent.
* **Window & Door Detection:** Identifies structural boundaries to prevent the generative model from obstructing natural light sources or entryways.

### 2. Intelligent Room Mapping Logic
The system features a custom heuristic engine that maps detected objects to staging prompts:
* **Weighted Thresholding:** Structural anchors (fireplaces) are prioritised over temporary items to determine room identity.
* **Fallback Logic:** Statistically defaults to "Bedroom" for empty rooms to align with standard residential listing ratios.

### 3. 2K High-Resolution Staging
Utilizes the **SeeDream 4.5** API via Replicate to generate 2K resolution staged images. The system injects dynamic prompt constraints (Safety Heuristics) to maintain structural materials (walls, floors, windows) while adding realistic 3D furniture.

### 4. CV Verification Pass (The "Turing Test")
A unique feature of this pipeline is the **Verification Pass**. After staging, the system runs a second round of YOLO detection on the AI-generated image. This empirically validates the realism of the staged furniture; if the CV model can recognize the "AI bed" with high confidence, the generation is deemed spatially and geometrically accurate.

---

## Technical Stack
* **Language:** Python 3.11+
* **CV Model:** YOLO-World (via Ultralytics)
* **Generative Engine:** SeeDream 4.5 (Replicate API)
* **Fallback Engine:** OpenAI GPT-Image-1
* **Image Processing:** PIL (Pillow) for normalization and multi-pass filtering.
* **Hardware Management:** PyTorch (VRAM management/CUDA cache clearing for stability).

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/re-ai-solution.git
   cd re-ai-solution

## Setup & Installation

### Install dependencies:

```bash
pip install -r requirements.txt
```
---
## Environment Variables

Configure your `.env` file or Colab Secrets:

* **REPLICATE_API_TOKEN**: Your Replicate API key.
* **OPENAI_API_KEY**: Your OpenAI API key.
* **USE_LIVE_API**: Set to `1` for production runs.

---

## Methodology & Results

The development of **RE.Ai Solution** involved significant iteration on Prompt Engineering and Confidence Threshold Tuning.

### Challenges Overcome:

* **API Drift**: Updated the pipeline to handle schema changes in SeeDream 4.5 (migrating from `image_urls` to `image_input` arrays).
* **Window Occlusion**: Solved using a "Safety Heuristic" that appends architectural protection rules to the end of prompts, leveraging model recency bias.
* **Memory Optimization**: Implemented `torch.cuda.empty_cache()` to resolve Replicate "Director" errors during high-resolution processing.

---

*Developed as a final project for **Houston City College***.
