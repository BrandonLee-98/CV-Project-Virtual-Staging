# AI Usage Log: RE.Ai Solution

**Project:** Intelligent Virtual Staging Pipeline  
**Course:** AI and Robotics Capstone  
**Institution:** Houston City College  
**Developer:** Brandon Matias

## Overview
This document tracks the iterative development of the RE.Ai Solution, detailing the collaboration between the developer and AI agents to solve complex computer vision, architectural decoupling, and generative AI hurdles.

---

## Log Entry 1: Project Scaffolding & Decoupling
**Date:** [Insert Date]
**Task:** Extracting core Computer Vision (CV) logic from a production Replit web application into a standalone Python module.
**AI Tool:** Replit Agent (Claude-based)
**Hurdle:** The original staging logic was tightly coupled with a PostgreSQL database and Flask web server, making isolated testing difficult.
**Solution:** Prompted the AI to "identify and extract all core CV and Image Processing logic into a standalone Python port."
**What I Learned:** I learned how to separate "business logic" from "functional logic" to create a reproducible environment for testing and deployment.

## Log Entry 2: Implementing Dual-Mode Logic (Offline/Live)
**Date:** [Insert Date]
**Task:** Creating a "Gated" API flag for cost management and testing reliability.
**AI Tool:** Replit Agent
**Logic:** Implemented a `USE_LIVE_API` flag at the top level of the module.
**What I Learned:** I learned how to use `os.getenv` to securely handle sensitive API keys in a professional environment rather than hardcoding them, and how to build "placeholder" fallbacks for cost-efficient debugging.

## Log Entry 3: Core Pipeline & API Handshake
**Date:** [Insert Date]
**Task:** Migrating to SeeDream 4.5 for 2K High-Resolution Staging.
**Hurdle:** Encountered an `Unexpected output shape` error after the model returned a `replicate.helpers.FileOutput` object instead of a standard URL string.
**Solution:** Refactored `cv_logic.py` to include a robust parsing block that checks for `.url` attributes on output objects, ensuring stable image downloads in modern SDK versions.

## Log Entry 4: Architectural Integrity & Zero-Shot Detection
**Date:** [Insert Date]
**Task:** Implementing YOLO-World for structural anchor detection.
**Hurdle:** Standard models often defaulted empty bedrooms to "Living Rooms" due to a lack of furniture context.
**Solution:** Integrated YOLO-World for zero-shot detection of custom anchors (fireplaces, ceiling fans). Established a "Weighted Threshold" logic where architectural anchors trigger specific room mapping at lower confidence (0.10) than temporary furniture.

## Log Entry 5: Stability & Error Handling (E9243)
**Date:** [Insert Date]
**Task:** Resolving Replicate "Director" errors during batch processing.
**Hurdle:** The API returned `E9243` errors when VRAM was under pressure or when prompts exceeded token limits.
**Solution:** 1. Implemented `torch.cuda.empty_cache()` after every detection pass to clear GPU memory.
2. Optimized prompt structure by removing redundant parameters like `aspect_ratio` to reduce payload complexity.

## Log Entry 6: Prompt Engineering & Recency Bias
**Date:** [Insert Date]
**Task:** Preventing "Window Occlusion" in generative staging.
**Hurdle:** Generative models frequently ignored early-prompt instructions and placed furniture over windows.
**Solution:** Engineered a "Safety Heuristic" that appends critical architectural protection rules to the *end* of the prompt, leveraging "Recency Bias" to ensure window visibility remained the highest priority for the AI.

## Log Entry 7: Verification & The "Turing Test"
**Date:** [Insert Date]
**Task:** Developing a secondary validation layer for spatial realism.
**Logic:** Developed a "Double-Pass" architecture. Round 1 detects the empty room; Round 2 runs a CV pass on the final staged result. 
**Verification:** If the secondary CV pass detects the AI-generated furniture with >0.25 confidence, the staging is validated as geometrically and spatially accurate.

---

## Honest Analysis of AI Collaboration
The development of **RE.Ai Solution** was a hybrid effort. AI tools were used for rapid scaffolding and debugging API drift, while human intervention was required to design the "architectural anchors" and safety heuristics that prevent hallucinations. The project successfully evolved from a simple web-tool extraction to a sophisticated, self-verifying computer vision pipeline.

---
*Developed as a final project for Houston City College.*
