"""
cv_logic.py
===========

Standalone Python port of the virtual-staging pipeline used by the
RE.Ai Solutions web app (a TypeScript/Express project).

This module is a faithful translation of two server-side files from the
parent repository:

  * ``server/controlnet.ts``        — Replicate / SeeDream 4.5 path
  * ``server/imageEnhancement.ts``  — OpenAI ``gpt-image-1`` fallback path

Both source files are pure API-orchestration code. There is **no
in-house computer-vision logic** in the parent project (no segmentation,
no YOLO/ultralytics, no OpenCV, no ControlNet preprocessing despite the
historic file name). Accordingly this port contains no CV logic of its
own — only:

  1. PIL-based image resizing / format conversion that mirrors the
     ``sharp`` calls in the TypeScript code.
  2. Thin wrappers around the ``replicate`` and ``openai`` Python SDKs.
  3. The exact prompt strings (verbatim copies, not paraphrases) that
     the production app sends to the model.

Use it from a Jupyter notebook (see ``final_demo.ipynb``) to run a
single image through the same pipeline the web app uses, with a
``USE_LIVE_API`` flag for offline iteration.

Nothing in this file imports from the parent project; it is fully
standalone.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# When False, ``call_replicate_seedream`` and ``call_openai_fallback`` skip
# the network entirely and return the bundled placeholder output. Flip to
# True (or set the environment variable USE_LIVE_API=1) to actually hit
# Replicate / OpenAI. The notebook also exposes this flag in its first
# cell so it can be toggled interactively.
USE_LIVE_API: bool = os.getenv("USE_LIVE_API", "0").lower() in {"1", "true", "yes"}

# Folder that holds the offline placeholder pair. The notebook also reads
# the input image from here when USE_LIVE_API is False.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLACEHOLDER_INPUT = RESULTS_DIR / "placeholder_input.jpg"
PLACEHOLDER_OUTPUT = RESULTS_DIR / "placeholder_output.jpg"


# ---------------------------------------------------------------------------
# Prompt constants
#
# These strings are copied verbatim from server/controlnet.ts (Apache-licensed
# project source). They drive the entire "additive staging" behaviour — the
# model is told in plain English to leave walls / windows / floors alone and
# only place new freestanding furniture on top of the existing surfaces.
# There is no segmentation mask or inpainting mask backing this; the
# "structural preservation" is purely a prompt-engineering technique.
# ---------------------------------------------------------------------------

# server/controlnet.ts:35 — Unified Structural Integrity Rule
STRUCTURAL_INTEGRITY_RULE = (
    "STRICT ARCHITECTURAL RULE: Maintain the exact original walls, ceilings, "
    "floor materials, window frames, glass surfaces, doors, door openings, "
    "archways, and built-in cabinetry. Do not replace, paint over, or modify "
    "any existing surfaces. All staged items must be placed realistically on "
    "top of the existing floor and against existing walls without altering "
    "the room's base structure or camera perspective. ABSOLUTE WINDOW AND "
    "DOOR RULE: NEVER place any wall art, mirror, painting, photograph, "
    "poster, decor, shelving, cabinet, headboard, plant, floor lamp, sconce, "
    "or tall furniture in front of, over, on top of, leaning against, or in "
    "any way covering or obstructing any window, glass door, sliding door, "
    "French door, transom, skylight, or door opening. Every window, glass "
    "surface, and door opening visible in the source photo must remain fully "
    "visible and unobstructed in the output. ABSOLUTE BED RULE: NEVER place "
    "any wall art, painting, mirror, photograph, poster, or wall hanging "
    "directly above any bed or headboard, on the wall the bed sits against, "
    "or anywhere within the wall area immediately behind the bed. Wall art "
    "belongs ONLY on side walls or unoccupied walls. This rule applies even "
    "if the bed is positioned against a window, archway, or any other "
    "architectural feature — when in doubt, omit wall art behind the bed "
    "entirely."
)

# server/controlnet.ts:38 — appended to indoor prompts
CONSTRAINT_CLAUSE = (
    f"{STRUCTURAL_INTEGRITY_RULE} CRITICAL WINDOW AND OPENING PRESERVATION: "
    "Treat every window, glass door, and door opening as a strict "
    "no-placement zone for any added object. Do not hang, mount, lean, or "
    "place any painting, mirror, photograph, wall art, decor, shelving, "
    "tall furniture, headboard, plant, or lamp in front of, over, or "
    "covering any window, glass door, or door opening. Keep the full glass "
    "area of every window completely uncovered (no objects in front of the "
    "glass). Ensure all added items are scaled realistically to the room's "
    "dimensions. Maintain the original camera angle, perspective, and "
    "natural lighting sources."
)

# server/controlnet.ts:41 — exterior/utility constraint
EXTERIOR_CONSTRAINT_CLAUSE = (
    "NEGATIVE CONSTRAINTS: Do NOT add any floor lamps, table lamps, wall "
    "art, paintings, or indoor rugs. No indoor-specific furniture or "
    "lighting. Use only exterior-grade materials like powder-coated steel, "
    f"teak, or concrete. {STRUCTURAL_INTEGRITY_RULE}"
)

# server/controlnet.ts:44 — specialised prompts that override style selection
EXTERIOR_PROMPTS: dict[str, str] = {
    "outdoor": (
        "Photorealistic virtual staging of this outdoor patio in a custom "
        "aesthetic. Add a weather-resistant lounge seating set with durable "
        "fabric and a heavy-duty metal firepit. " + EXTERIOR_CONSTRAINT_CLAUSE
    ),
    "garage": (
        "High-fidelity staging of this garage interior. Add heavy-duty "
        "industrial metal shelving units with organized storage bins and a "
        "rolling tool cabinet. Ensure all items are placed against walls "
        "without blocking door hallways or window light. "
        + EXTERIOR_CONSTRAINT_CLAUSE
    ),
}

SPECIAL_ROOM_TYPES = {"garage", "outdoor"}

# server/controlnet.ts:53 — additive room/style matrix. Trimmed here to the
# four indoor room types that drive 95% of traffic in the production app.
# Add more entries from the TS source if you need them.
ROOM_STYLE_PROMPTS: dict[str, dict[str, str]] = {
    "living_room": {
        "modern": (
            "Additive virtual staging for a Modern Living Room. Place a "
            "low-profile leather sectional and matching area rug over the "
            "existing floor. Do not cover or modify windows with paintings "
            "or tall furniture. Ensure the original wall color and flooring "
            "texture remain visible and unchanged."
        ),
        "contemporary": (
            "Additive virtual staging for a Contemporary Living Room. Place "
            "curved velvet armchairs and a geometric wool rug over the "
            "existing floor. Do not cover or modify windows with paintings "
            "or tall furniture. Ensure the original wall color and flooring "
            "texture remain visible and unchanged."
        ),
        "traditional": (
            "Additive virtual staging for a Traditional Living Room. Place "
            "a tufted rolled-arm sofa with patterned throw pillows, a "
            "wingback armchair, a polished cherry-wood coffee table, and a "
            "classic Persian-style area rug over the existing floor. Add a "
            "brass table lamp. Ensure the original wall color and flooring "
            "texture remain visible and unchanged."
        ),
        "minimalist": (
            "Additive virtual staging for a Minimalist Living Room. Place a "
            "single light-grey modular sofa and a neutral-toned textured "
            "rug over the existing floor, focusing on open space. Ensure "
            "the original wall color and flooring texture remain visible "
            "and unchanged."
        ),
    },
    "bedroom": {
        "modern": (
            "Additive virtual staging for a Modern Bedroom. Place a "
            "floating walnut platform bed and large area rug over the "
            "existing floor. Add charcoal grey bedding. Do not cover or "
            "modify windows. Ensure the original wall color and flooring "
            "texture remain visible and unchanged."
        ),
        "contemporary": (
            "Additive virtual staging for a Contemporary Bedroom. Place a "
            "tufted navy headboard bed with mirrored nightstands and a "
            "plush cream area rug over the existing floor. Do not cover or "
            "modify windows. Ensure the original wall color and flooring "
            "texture remain visible and unchanged."
        ),
        "traditional": (
            "Additive virtual staging for a Traditional Bedroom. Place a "
            "button-tufted upholstered bed with crisp layered linens and "
            "decorative shams, matching wooden nightstands with brass "
            "lamps, and a patterned area rug over the existing floor. If "
            "wall art is added, place it on side walls only — never above "
            "the bed or headboard. Ensure the original wall color and "
            "flooring texture remain visible and unchanged."
        ),
        "minimalist": (
            "Additive virtual staging for a Minimalist Bedroom. Place a "
            "low platform bed with white organic cotton linens and a simple "
            "jute mat over the existing floor. Ensure the original wall "
            "color and flooring texture remain visible and unchanged."
        ),
    },
    "kitchen": {
        "modern": (
            "Additive virtual staging for a Modern Kitchen. Add sleek bar "
            "stools at the counter, a geometric fruit bowl, and small "
            "potted herbs. Ensure the original cabinetry, countertops, and "
            "flooring remain visible and unchanged."
        ),
        "contemporary": (
            "Additive virtual staging for a Contemporary Kitchen. Add "
            "modern bar seating, decorative countertop accessories, and "
            "fresh greenery. Ensure the original cabinetry, countertops, "
            "and flooring remain visible and unchanged."
        ),
        "traditional": (
            "Additive virtual staging for a Traditional Kitchen. Add a "
            "classic ceramic fruit bowl, a wooden cutting board, a polished "
            "brass kettle, and upholstered counter stools with nailhead "
            "trim. Ensure the original cabinetry, countertops, and flooring "
            "remain visible and unchanged."
        ),
        "minimalist": (
            "Additive virtual staging for a Minimalist Kitchen. Add clean "
            "countertop accessories, simple bar stools, and one sculptural "
            "plant. Ensure the original cabinetry, countertops, and "
            "flooring remain visible and unchanged."
        ),
    },
    "dining_room": {
        "modern": (
            "Additive virtual staging for a Modern Dining Room. Place a "
            "sleek glass or marble dining table and contemporary chairs "
            "over the existing floor. Add a minimalist centerpiece. Ensure "
            "the original wall color and flooring remain visible and "
            "unchanged."
        ),
        "contemporary": (
            "Additive virtual staging for a Contemporary Dining Room. "
            "Place an elegant wooden table and upholstered chairs over the "
            "existing floor. Ensure the original wall color and flooring "
            "remain visible and unchanged."
        ),
        "traditional": (
            "Additive virtual staging for a Traditional Dining Room. Place "
            "a polished wooden trestle dining table with upholstered Queen "
            "Anne chairs over the existing floor. Add a classic floral "
            "centerpiece and a patterned area rug. Ensure the original "
            "wall color and flooring remain visible and unchanged."
        ),
        "minimalist": (
            "Additive virtual staging for a Minimalist Dining Room. Place "
            "a simple wooden table and clean-lined chairs over the existing "
            "floor, focusing on open space. Ensure the original wall color "
            "and flooring remain visible and unchanged."
        ),
    },
    "bathroom": {
        "modern": (
            "Additive virtual staging for a Modern Bathroom. Add a sleek "
            "teak shower mat, plush white towels, and minimalist countertop "
            "accessories. Ensure the original wall tiles, flooring, and "
            "fixtures remain visible and unchanged."
        ),
        "contemporary": (
            "Additive virtual staging for a Contemporary Bathroom. Add a "
            "decorative stool, patterned bath runner, and modern vanity "
            "trays. Ensure the original wall tiles, flooring, and fixtures "
            "remain visible and unchanged."
        ),
        "traditional": (
            "Additive virtual staging for a Traditional Bathroom. Add "
            "plush white towels with embroidered detail, a polished brass "
            "tray with classic apothecary jars, and a small framed botanical "
            "print on a side wall. Ensure the original wall tiles, flooring, "
            "and fixtures remain visible and unchanged."
        ),
        "minimalist": (
            "Additive virtual staging for a Minimalist Bathroom. Add a "
            "single bamboo bath mat and a uniform set of ceramic dispensers. "
            "Ensure the original wall tiles, flooring, and fixtures remain "
            "visible and unchanged."
        ),
    },
    "office": {
        "modern": (
            "Additive virtual staging for a Modern Home Office. Place a "
            "sleek desk, ergonomic chair, and minimalist shelving over the "
            "existing floor. Ensure the original wall color and flooring "
            "remain visible and unchanged."
        ),
        "contemporary": (
            "Additive virtual staging for a Contemporary Home Office. "
            "Place a wooden desk with metal accents and comfortable task "
            "chair over the existing floor. Ensure the original wall color "
            "and flooring remain visible and unchanged."
        ),
        "traditional": (
            "Additive virtual staging for a Traditional Home Office. Place "
            "a solid mahogany executive desk, a leather-tufted office "
            "chair, and matching wooden bookcases over the existing floor. "
            "Add a classic green-shaded banker's lamp. Ensure the original "
            "wall color and flooring remain visible and unchanged."
        ),
        "minimalist": (
            "Additive virtual staging for a Minimalist Home Office. Place "
            "a clean-lined desk and simple chair over the existing floor "
            "with maximum negative space. Ensure the original wall color "
            "and flooring remain visible and unchanged."
        ),
    },
}

# server/controlnet.ts:93 — fallback style descriptions
STYLE_DESCRIPTIONS: dict[str, str] = {
    "modern": "Modern style with sleek clean lines, neutral tones, geometric shapes, and minimalist aesthetic",
    "minimalist": "Minimalist style with essential items only, maximum negative space, simple forms, and zen aesthetic",
    "contemporary": "Contemporary style with current trends, mixed materials, comfortable design, and subtle color accents",
    "traditional": "Traditional style with timeless classic furniture, rich wood tones, layered patterns, refined symmetry, and warm tailored fabrics with elegant details",
}

# server/controlnet.ts:101 — human-readable room names for fallback assembly
ROOM_DESCRIPTIONS: dict[str, str] = {
    "living_room": "living room",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "dining_room": "dining room",
    "office": "home office",
    "garage": "garage or utility space",
    "outdoor": "outdoor patio or backyard",
}

RoomType = Literal[
    "living_room",
    "bedroom",
    "kitchen",
    "dining_room",
    "bathroom",
    "office",
    "garage",
    "outdoor",
]
StyleName = Literal["modern", "contemporary", "traditional", "minimalist"]


def build_staging_prompt(room_type: RoomType, style: StyleName) -> str:
    """
    Assemble the full staging prompt the way ``virtualStageWithControlNet``
    in ``server/controlnet.ts`` does (lines 271-285).

    Parameters
    ----------
    room_type
        One of the indoor types ("living_room", "bedroom", "kitchen",
        "dining_room") or one of the special exterior types ("garage",
        "outdoor"). Special types ignore ``style`` and use their own
        canned prompt.
    style
        One of "modern", "contemporary", "traditional", "minimalist".
        Ignored for special room types.

    Returns
    -------
    str
        The full multi-paragraph prompt — room/style sentence first, then
        the structural-integrity constraint clause, then the photorealism
        suffix. This is exactly what the production app sends to
        SeeDream 4.5.
    """
    if room_type in SPECIAL_ROOM_TYPES:
        return (
            EXTERIOR_PROMPTS[room_type]
            + "\n\nPhotorealistic photography, perfect natural lighting, "
            "high resolution, sharp details, professional real estate photo "
            "quality."
        )

    room_styles = ROOM_STYLE_PROMPTS.get(room_type, ROOM_STYLE_PROMPTS["living_room"])
    # Mirror TS fallback at server/controlnet.ts:283 — when a room/style
    # combination isn't in the matrix, synthesize one from STYLE_DESCRIPTIONS
    # and the human-readable room name rather than silently swapping styles.
    room_name = ROOM_DESCRIPTIONS.get(room_type, "interior space")
    style_prompt = room_styles.get(
        style, f"{STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS['modern'])} staging for {room_name}"
    )
    return (
        f"{style_prompt} {CONSTRAINT_CLAUSE}\n\n"
        "Photorealistic interior photography, perfect natural lighting, "
        "high resolution, sharp details, professional real estate photo "
        "quality."
    )


# ---------------------------------------------------------------------------
# Preprocessing — PIL ports of the sharp() calls in the TS code
# ---------------------------------------------------------------------------


def prepare_image_for_staging(
    image_path: str | Path,
    out_path: Optional[str | Path] = None,
    max_size: int = 1024,
) -> Path:
    """
    PIL port of ``prepareImageForStaging`` from
    ``server/controlnet.ts`` (lines 117-177).

    What it does
    ------------
    1. Opens the image (any PIL-supported format: JPG/PNG/WebP/etc.).
    2. Computes a target size that fits inside ``max_size`` x ``max_size``
       while preserving aspect ratio. Default 1024 px matches the
       SeeDream 4.5 ``size="1024x1024"`` parameter.
    3. Snaps both dimensions down to the nearest multiple of 8. Diffusion
       models generally need width/height divisible by 8 (or 64) at the
       latent layer; the original TS code does the same arithmetic.
    4. Re-encodes as JPEG quality 90 — the format SeeDream 4.5 ingests
       most reliably.

    Parameters
    ----------
    image_path
        Source photo on disk. Any format PIL can open.
    out_path
        Where to write the prepared JPEG. If None, writes
        ``<image_path>.prepared.jpg`` next to the input.
    max_size
        Long-edge cap in pixels. Defaults to 1024 to match the production
        SeeDream call.

    Returns
    -------
    pathlib.Path
        The path of the prepared JPEG.
    """
    src = Path(image_path)
    if out_path is None:
        out_path = src.with_suffix(".prepared.jpg")
    dst = Path(out_path)

    img = Image.open(src)
    # Preserve EXIF orientation, then drop alpha so JPEG encoding works.
    img = img.convert("RGB")
    original_w, original_h = img.size

    target_w, target_h = original_w, original_h
    if original_w > max_size or original_h > max_size:
        if original_w >= original_h:
            target_w = max_size
            target_h = round(original_h / original_w * max_size)
        else:
            target_h = max_size
            target_w = round(original_w / original_h * max_size)

    # Snap to multiples of 8 (matches sharp() pipeline; floor, not round).
    target_w = max(8, (target_w // 8) * 8)
    target_h = max(8, (target_h // 8) * 8)

    if (target_w, target_h) != (original_w, original_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    img.save(dst, format="JPEG", quality=90, optimize=True)
    print(
        f"[prepare] {original_w}x{original_h} -> {target_w}x{target_h} "
        f"({dst.stat().st_size / 1024:.1f} KB)"
    )
    return dst


def optimize_image_for_ai(
    image_path: str | Path,
    out_path: Optional[str | Path] = None,
    max_size: int = 1536,
) -> Path:
    """
    PIL port of ``optimizeImageForAI`` from
    ``server/imageEnhancement.ts`` (lines 82-118), used by the OpenAI
    fallback path.

    Differences from ``prepare_image_for_staging``:
      * Uses ``max_size`` 1536 instead of 1024 (OpenAI's
        ``images.edit`` accepts a larger envelope).
      * Encodes as PNG (lossless) instead of JPEG, because
        ``gpt-image-1`` is more sensitive to JPEG compression artifacts
        in the input.
      * Does NOT snap to multiples of 8 — the OpenAI endpoint handles
        odd dimensions internally.

    Parameters
    ----------
    image_path : str or Path
    out_path : str or Path or None
    max_size : int

    Returns
    -------
    pathlib.Path
        Path of the prepared PNG.
    """
    src = Path(image_path)
    if out_path is None:
        out_path = src.with_suffix(".optimized.png")
    dst = Path(out_path)

    img = Image.open(src).convert("RGB")
    original_w, original_h = img.size

    target_w, target_h = original_w, original_h
    if original_w > max_size or original_h > max_size:
        if original_w >= original_h:
            target_w = max_size
            target_h = round(original_h / original_w * max_size)
        else:
            target_h = max_size
            target_w = round(original_w / original_h * max_size)

    if (target_w, target_h) != (original_w, original_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    img.save(dst, format="PNG", optimize=True, compress_level=6)
    print(
        f"[optimize] {original_w}x{original_h} -> {target_w}x{target_h} "
        f"({dst.stat().st_size / 1024:.1f} KB)"
    )
    return dst


# ---------------------------------------------------------------------------
# API wrappers
# ---------------------------------------------------------------------------


def _placeholder_copy(dest: Path) -> Path:
    """Copy the bundled placeholder output to ``dest`` and return it."""
    if not PLACEHOLDER_OUTPUT.exists():
        raise FileNotFoundError(
            f"Placeholder output missing at {PLACEHOLDER_OUTPUT}. "
            "Re-run the export's image-generation step."
        )
    shutil.copyfile(PLACEHOLDER_OUTPUT, dest)
    return dest


def call_replicate_seedream(
    image_path: str | Path,
    prompt: str,
    out_path: str | Path = "staged_seedream.jpg",
    size: str = "1024x1024",
) -> Path:
    """
    Mirror of ``generateStagedImage`` in ``server/controlnet.ts``
    (lines 189-247).
    """
    out = Path(out_path)
    if not USE_LIVE_API:
        print("[seedream] USE_LIVE_API=False — using placeholder output.")
        return _placeholder_copy(out)

    # Late imports so the module is importable without the SDK installed.
    import random
    import requests
    import replicate  # type: ignore

    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Either export it or set "
            "USE_LIVE_API=False to use the offline placeholder."
        )

    client = replicate.Client(api_token=token)
    print(f"[seedream] Calling bytedance/seedream-4.5 ({size})...")
    with open(image_path, "rb") as fh:
        # UPDATED: 'image_input' is the correct field name and must be a list [fh].
        output = client.run(
            "bytedance/seedream-4.5",
            input={
                "prompt": prompt,
                "image_input": [fh],
                "size": size,
                "aspect_ratio": "match_input_image"
            },
        )

    # Replicate returns the URL(s) in several shapes — handle the common ones,
    # mirroring the TS code's defensive parsing (server/controlnet.ts:214-233).
# Replicate returns the URL(s) in several shapes — handle the common ones.
    image_url: str | None = None
    if isinstance(output, str):
        image_url = output
    elif isinstance(output, list) and output:
        # UPDATED: Handle replicate.helpers.FileOutput objects safely
        if hasattr(output[0], "url"):
            image_url = output[0].url
        elif isinstance(output[0], str):
            image_url = output[0]
        else:
            image_url = str(output[0]) # Fallback cast
    elif isinstance(output, dict):
        data = output.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            image_url = data[0].get("url")
        else:
            image_url = (
                output.get("output") or output.get("image") or output.get("url")
            )

    if not image_url:
        raise RuntimeError(f"Unexpected SeeDream output shape: {output!r}")

    print(f"[seedream] Downloading {image_url[:60]}...")
    resp = requests.get(image_url, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print(f"[seedream] Wrote {out} ({len(resp.content) / 1024:.1f} KB)")
    return out


def call_openai_fallback(
    image_path: str | Path,
    prompt: str,
    out_path: str | Path = "staged_openai.png",
    size: str = "1024x1024",
) -> Path:
    """
    Mirror of the OpenAI fallback in ``virtuallyStageRoom``,
    ``server/imageEnhancement.ts`` (lines 200-235).

    When ``USE_LIVE_API`` is True
        Calls ``openai.images.edit`` with model ``gpt-image-1``, the
        prepared image, the prompt, and a 1024x1024 size. Writes the
        decoded base64 result to ``out_path``. Requires
        ``OPENAI_API_KEY`` in the environment and the ``openai`` package
        installed.

    When ``USE_LIVE_API`` is False
        Returns the placeholder output, identical to the SeeDream stub.

    Parameters
    ----------
    image_path
        Path to the *optimised* image (call ``optimize_image_for_ai``
        first; gpt-image-1 prefers PNG input at <= 1536 px).
    prompt
        Full staging prompt.
    out_path
        Where to write the resulting staged image.
    size
        Output dimensions. The production app uses "1024x1024".

    Returns
    -------
    pathlib.Path
        Path of the saved staged image.
    """
    out = Path(out_path)
    if not USE_LIVE_API:
        print("[openai] USE_LIVE_API=False — using placeholder output.")
        return _placeholder_copy(out)

    import base64
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either export it or set "
            "USE_LIVE_API=False to use the offline placeholder."
        )

    # The production app routes through Replit's OpenAI-compatible
    # gateway via AI_INTEGRATIONS_OPENAI_BASE_URL. Standalone use just
    # hits api.openai.com directly.
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    print(f"[openai] Calling images.edit gpt-image-1 ({size})...")
    with open(image_path, "rb") as fh:
        response = client.images.edit(
            model="gpt-image-1",
            image=fh,
            prompt=prompt,
            size=size,
        )

    b64 = response.data[0].b64_json if response.data else None
    if not b64:
        raise RuntimeError("OpenAI returned no image data.")
    out.write_bytes(base64.b64decode(b64))
    print(f"[openai] Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return out


# ---------------------------------------------------------------------------
# Convenience: end-to-end one-shot
# ---------------------------------------------------------------------------


def stage_image(
    image_path: str | Path,
    room_type: RoomType = "living_room",
    style: StyleName = "modern",
    backend: Literal["seedream", "openai"] = "seedream",
    out_path: Optional[str | Path] = None,
) -> tuple[Path, Path, str]:
    """
    Full convenience wrapper: preprocess -> build prompt -> call backend.

    Returns
    -------
    (prepared_input_path, staged_output_path, prompt_used)
    """
    prompt = build_staging_prompt(room_type, style)
    if backend == "seedream":
        prepared = prepare_image_for_staging(image_path)
        staged = call_replicate_seedream(
            prepared, prompt, out_path or "staged_seedream.jpg"
        )
    elif backend == "openai":
        prepared = optimize_image_for_ai(image_path)
        staged = call_openai_fallback(
            prepared, prompt, out_path or "staged_openai.png"
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    return prepared, staged, prompt


__all__ = [
    "USE_LIVE_API",
    "STRUCTURAL_INTEGRITY_RULE",
    "CONSTRAINT_CLAUSE",
    "EXTERIOR_CONSTRAINT_CLAUSE",
    "EXTERIOR_PROMPTS",
    "ROOM_STYLE_PROMPTS",
    "STYLE_DESCRIPTIONS",
    "ROOM_DESCRIPTIONS",
    "build_staging_prompt",
    "prepare_image_for_staging",
    "optimize_image_for_ai",
    "call_replicate_seedream",
    "call_openai_fallback",
    "stage_image",
]


from ultralytics import YOLOWorld

# --- New "Intelligent" Wrapper ---

# --- Add this to the bottom of src/cv_logic.py ---
from ultralytics import YOLOWorld

def run_intelligent_staging(img_filename, style="modern"):
    """
    Orchestration Layer:
    1. Detects architectural anchors using YOLO-World.
    2. Maps anchors to RoomType (Living Room vs Bedroom).
    3. Executes the generative staging pipeline.
    """
    # Initialize Zero-Shot Model
    model = YOLOWorld('yolov8s-world.pt') 
    model.set_classes(["window", "fireplace", "ceiling fan", "refrigerator"])
    
    # Pathing logic consistent with Colab environment
    img_path = f"/content/data/raw/{img_filename}"
    
    # Anchor Detection (using the 0.10 threshold)
    results = model.predict(img_path, conf=0.10, verbose=False)
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
    
    # Heuristic Mapping
    if "fireplace" in labels:
        room_type = "living_room"
    elif "refrigerator" in labels:
        room_type = "kitchen"
    else:
        room_type = "bedroom" # Default fallback
        
    print(f"🔍 Analysis: Detected {labels}. Using: {room_type}")
    
    # Call the original staging engine
    prepared, staged, _ = stage_image(img_path, room_type=room_type, style=style)
    
    return img_path, results[0], staged
