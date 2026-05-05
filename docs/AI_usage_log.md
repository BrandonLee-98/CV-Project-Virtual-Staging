# AI Usage Log - RE.AI Solutions
**Course:** ITAI 1378 Computer Vision
**Student:** Brandon Matias

## Log Entries

### Entry 1: Project Scaffolding & Decoupling
* **Date:** [Insert Date]
* **Task:** Extracting core Computer Vision (CV) logic from a production Replit web application into a standalone Python module.
* **AI Tool:** Replit Agent (Claude-based)
* **Prompt Used:** "Please identify and extract all the core Computer Vision and Image Processing logic from this project... provide the isolated Python port... in a standalone directory."
* **What I Learned:** I learned how to separate "business logic" (PostgreSQL, Flask) from "functional logic" (PIL preprocessing, API calls) to create a reproducible environment for testing.
* **Verification:** I manually tested the extracted `cv_logic.py` with a sample image to ensure the API handshake was successful without the rest of the web server running.

---

### Entry 2: Implementing Dual-Mode Logic (Offline/Live)
* **Date:** [Insert Date]
* **Task:** Creating a "Gated" API flag for cost management and testing reliability.
* **AI Tool:** Replit Agent
* **Prompt Used:** "Both — gate behind a USE_LIVE_API flag at the top of the notebook."
* **What I Learned:** I learned how to use `os.getenv` to securely handle sensitive API keys in a professional environment rather than hardcoding them.
* **Verification:** I ran the code with the flag set to `False` to verify that the placeholder image logic worked correctly before burning API credits.
