# Model Documentation: CV Virtual Staging Project

This directory contains the technical specifications, access links, and configuration logic for the Computer Vision and Generative AI models used in the staging pipeline.

---

## 1. Primary Generative Engine: SeeDream 4.5
* **Provider:** Bytedance (via Replicate API)
* **Model Link:** [replicate.com/bytedance/seedream-4.5](https://replicate.com/bytedance/seedream-4.5)
* **Role:** High-fidelity image-to-image diffusion. 
* **Configuration:**
    * **Resolution:** 2K (2048x2048)
    * **Preprocessing:** Custom PIL-based normalization to 1024x768 before inference.
    * **Logic:** Receives a "Structural Integrity" prompt + Dynamic CV Constraints to generate furniture while preserving existing room geometry.

## 2. Computer Vision Engine: YOLO-World (Zero-Shot)
* **Provider:** Ultralytics
* **Model Link:** [docs.ultralytics.com/models/yolo-world](https://docs.ultralytics.com/models/yolo-world)
* **Role:** Real-time architectural anchor detection.
* **Why YOLO-World?** Unlike standard YOLO models (COCO-trained), YOLO-World allows for **Zero-Shot detection** of custom vocabulary. This is critical for identifying non-standard real estate features like "fireplaces," "ceiling fans," and "glass doors" in empty rooms.
* **Thresholding Logic:**
    * **Architectural Anchors:** 0.10 Confidence (High Sensitivity)
    * **Generic Furniture:** 0.20 Confidence (High Precision)

## 3. Verification Model: YOLOv8s
* **Provider:** Ultralytics
* **Model Link:** [docs.ultralytics.com/models/yolov8](https://docs.ultralytics.com/models/yolov8)
* **Role:** Post-generation validation ("Turing Test").
* **Logic:** Executes a secondary inference pass on the 2K staged result. It is used to programmatically verify if the AI-generated objects are "real" enough to be recognized by a secondary vision model.

## 4. Fallback Engine: OpenAI GPT-Image-1
* **Provider:** OpenAI (via API)
* **Role:** Secondary staging fallback for complex lighting scenarios.
* **Status:** Inactive in current production build; reserved for edge-case redundancy.

---

## Technical Weights & Weights Management
The models are loaded dynamically within the `cv_logic.py` script. To manage hardware constraints on the deployment environment:
1.  **VRAM Clearing:** `torch.cuda.empty_cache()` is called after each YOLO inference to prevent memory leakage.
2.  **Model Pinned Version:** The system currently pins `yolov8s-world.pt` to ensure consistent bounding box coordinates across different testing batches.

---
*CV Virtual Staging: Developed for Houston City College*
