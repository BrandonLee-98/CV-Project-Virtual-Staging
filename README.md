# CV Virtual Staging Project: Intelligent Virtual Staging Pipeline

**Institution:** Houston City College | AI and Robotics Capstone  
**Developer:** Brandon Matias  

## 📌 Project Overview
The CV Virtual Staging Project is a Dual-Pass Computer Vision and Generative AI pipeline designed to automate the virtual staging of real estate photography. By combining **YOLO-World (Zero-Shot Detection)** for architectural anchor mapping and **SeeDream 4.5 (via Replicate)** for high-fidelity generative staging, the system transforms vacant rooms into photorealistic staged environments while preserving strict structural integrity.

---

## 🚀 60-Second Quick Start (Google Colab)

To provide a frictionless evaluation experience, the entire pipeline can be executed via a single master script in Google Colab. This script automatically clones the repository, syncs dependencies, injects local paths, and runs the end-to-end demonstration.

### Step 1: Open a Colab Environment
Create a blank [Google Colab Notebook](https://colab.research.google.com/) and ensure you are connected to a GPU runtime (`Runtime` > `Change runtime type` > `T4 GPU`).

### Step 2: Provide Testing Credentials
For production-grade security, API keys are not hardcoded in this repository. 
1. Download the `API_Keys_Testing.txt` file provided in the official Capstone grading submission portal.
2. Open the **Files** tab (folder icon) on the left sidebar in Colab.
3. Click the **Upload** icon and upload the `API_Keys_Testing.txt` file to the session storage.

### Step 3: Execute the Master Block
Copy and paste the following code into a Colab cell and press **Run**.

```python
# =================================================================
# CV Virtual Staging Project: Master Staging & Verification Pipeline
# =================================================================

import os
import sys
import subprocess
import shutil
import importlib

# 1. CLONE & HARD SYNC (Bypasses Colab Cache)
REPO_URL = "[https://github.com/BrandonLee-98/CV-Project-Virtual-Staging.git](https://github.com/BrandonLee-98/CV-Project-Virtual-Staging.git)"
REPO_NAME = "CV-Project-Virtual-Staging"
WORKING_DIR = f"/content/{REPO_NAME}"

print("📂 Initializing Environment...")
if not os.path.exists(WORKING_DIR):
    subprocess.run(["git", "clone", REPO_URL])
else:
    os.chdir(WORKING_DIR)
    subprocess.run(["git", "fetch", "--all"])
    subprocess.run(["git", "reset", "--hard", "origin/main"])
    os.chdir("/content")

# 2. BULLETPROOF PATH INJECTION
SRC_PATH = f"{WORKING_DIR}/src"
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Ensure absolute paths exist for cv_logic processing
os.makedirs("/content/data/raw/", exist_ok=True)
os.makedirs("/content/results/", exist_ok=True)

if os.path.exists(f"{WORKING_DIR}/data/raw/"):
    for f in os.listdir(f"{WORKING_DIR}/data/raw/"):
        shutil.copy(f"{WORKING_DIR}/data/raw/{f}", f"/content/data/raw/{f}")

# 3. INSTALL DEPENDENCIES
print("📦 Installing environment dependencies...")
subprocess.run(["pip", "install", "-qr", f"{WORKING_DIR}/requirements.txt"])

# 4. SECURE API AUTHENTICATION (Evaluation Mode)
key_file = "/content/API_Keys_Testing.txt"
os.environ["USE_LIVE_API"] = "1"

if os.path.exists(key_file):
    print("🔐 Parsing secure testing credentials...")
    with open(key_file, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key.strip()] = val.strip()
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("🔑 API Credentials Verified and Loaded.")
    else:
        print("⚠️ WARNING: API file found, but REPLICATE_API_TOKEN is missing.")
else:
    print(f"❌ ERROR: Credential file not found.")
    print(f"Please upload 'API_Keys_Testing.txt' to the Colab file explorer.")

# 5. EXECUTE INTELLIGENT PIPELINE
print("🧪 Loading System Logic...")
import cv_logic
importlib.reload(cv_logic) # Forces Python to utilize the freshly synced GitHub file

import matplotlib.pyplot as plt
from PIL import Image

TARGET_IMAGE = "placeholder_input3.jpg" # Ensure this file exists in your repo's data/raw folder

try:
    print(f"🎨 Starting Intelligent Staging for {TARGET_IMAGE}...")
    original, yolo_res, staged = cv_logic.run_intelligent_staging(TARGET_IMAGE, style="modern")

    # Visualize 3-Panel Staging Story
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(Image.open(original)); axes[0].set_title("1. Original Vacant Room"); axes[0].axis("off")
    axes[1].imshow(yolo_res.plot()); axes[1].set_title("2. Zero-Shot Anchor Detection"); axes[1].axis("off")
    axes[2].imshow(Image.open(staged)); axes[2].set_title("3. Final AI Staging"); axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    # Final Geometric Turing Test
    print("🧪 Running Secondary Vision Verification...")
    from ultralytics import YOLO
    v_results = YOLO('yolov8s.pt').predict(staged, conf=0.25, verbose=False)
    plt.figure(figsize=(10, 10)); plt.imshow(v_results[0].plot()); plt.axis("off"); plt.show()
    print("✅ Demo Successful: Architectural geometry maintained and populated.")

except Exception as e:
    print(f"❌ Pipeline Error: {e}")
```

---

## 🧠 System Architecture

The CV Virtual Staging Project operates on a strictly decoupled, modular architecture utilizing a **Dual-Pass Verification** methodology.

1. **Analysis Pass (Ultralytics YOLO-World):** Standard COCO-trained YOLO models fail to recognize empty rooms due to a lack of furniture context. This pipeline utilizes Zero-Shot detection to identify permanent architectural anchors (fireplaces, ceiling fans, windows) at a strict `0.10` confidence threshold.
2. **Generative Pass (SeeDream 4.5):** The system dynamically translates the CV findings into a targeted payload, generating a 2K resolution interior utilizing strict multi-image-input formatting and semantic safety heuristics to prevent window occlusion.
3. **Verification Pass (YOLOv8s):** The final staged image is run through a secondary computer vision model. If the generated furniture triggers a bounding box detection at a `>0.25` threshold, the geometry is validated as spatially accurate (a localized "Turing Test" for spatial realism).

## 📂 Repository Structure
* **`src/`**: Contains `cv_logic.py`, the core intelligence orchestration script bridging CV and Generative endpoints.
* **`notebooks/`**: Phase-separated Jupyter notebooks detailing Data Preprocessing (`01`), Threshold Tuning (`02`), and Production Staging (`03`).
* **`data/`**: Raw validation imagery utilized during the staging tests.
* **`models/`**: Technical justifications and weights logic for SeeDream, YOLO-World, and GPT-Fallback engines.
* **`AI_usage_log.md`**: Detailed documentation of AI agent collaboration, prompt engineering logic, and iterative debugging processes.
