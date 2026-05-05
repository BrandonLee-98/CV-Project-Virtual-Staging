"""
cv_logic.py
===========
Standalone Python port of the RE.Ai Solutions virtual-staging pipeline.
Includes YOLO-World Zero-Shot detection for automated room categorization
and architectural anchor preservation.
"""

from __future__ import annotations
import os
import io
import shutil
from pathlib import Path
from typing import Literal, Optional
from PIL import Image
import replicate
from ultralytics import YOLOWorld

# ---------------------------------------------------------------------------
# Core CV Logic & API Integration
# ---------------------------------------------------------------------------

def prepare_image_for_staging(image_path: str | Path, max_dim: int = 1024) -> str:
    """Normalizes image dimensions to prevent API timeouts."""
    img = Image.open(image_path)
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    out_path = "prepared_input.jpg"
    img.save(out_path, "JPEG", quality=95)
    return out_path

def call_replicate_seedream(image_path: str, prompt: str, out_path: str) -> str:
    """Handles the SeeDream 4.5 API call with strict schema validation."""
    # Strict API fix: 'size' must be exact uppercase "2K"
    output = replicate.run(
        "bytedance/seedream-4.5",
        input={
            "image": open(image_path, "rb"),
            "prompt": prompt,
            "size": "2K", 
            "guidance_scale": 7.5
        }
    )
    
    # Robust URL parsing for replicate outputs
    if hasattr(output, 'url'):
        image_url = output.url
    elif isinstance(output, list) and len(output) > 0 and hasattr(output[0], 'url'):
        image_url = output[0].url
    else:
        image_url = str(output)

    import requests
    response = requests.get(image_url)
    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path

def build_staging_prompt(room_type: str, style: str) -> str:
    return f"A beautifully staged {room_type} in a {style} interior design style, extremely realistic, highly detailed, photorealistic."

def stage_image(
    image_path: str | Path,
    room_type: str = "living_room",
    style: str = "modern",
    out_path: Optional[str | Path] = None,
) -> tuple[Path, Path, str]:
    """Generative execution engine."""
    prompt = build_staging_prompt(room_type, style)
    prepared = prepare_image_for_staging(image_path)
    staged = call_replicate_seedream(prepared, prompt, out_path or "staged_seedream.jpg")
    return prepared, staged, prompt

# ---------------------------------------------------------------------------
# The Intelligent Orchestrator (The Handshake)
# ---------------------------------------------------------------------------

def run_intelligent_staging(img_filename: str, style: str = "modern"):
    """
    Automated pipeline:
    1. Detects architectural anchors using YOLO-World (0.10 threshold).
    2. Maps anchors to RoomType logic.
    3. Executes the generative staging engine.
    """
    print("🔍 Initializing YOLO-World Zero-Shot Vision...")
    model = YOLOWorld('yolov8s-world.pt') 
    model.set_classes(["window", "fireplace", "ceiling fan", "refrigerator"])
    
    img_path = f"/content/data/raw/{img_filename}"
    results = model.predict(img_path, conf=0.10, verbose=False)
    
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
    
    # Architectural Anchor Mapping Heuristic
    if "fireplace" in labels:
        room_type = "living_room"
    elif "refrigerator" in labels:
        room_type = "kitchen"
    elif "ceiling fan" in labels:
        room_type = "bedroom"
    else:
        room_type = "bedroom" # Default structural fallback
        
    print(f"✅ CV Analysis Complete: Detected {labels} -> Categorized as {room_type.replace('_', ' ').title()}")
    
    # Trigger generative engine
    prepared_path, staged_path, _ = stage_image(img_path, room_type=room_type, style=style)
    
    return img_path, results[0], staged_path
