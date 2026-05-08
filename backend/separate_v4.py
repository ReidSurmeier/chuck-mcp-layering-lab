#!/usr/bin/env python3
"""
Woodblock color separation V4 — V2 base + Real-ESRGAN upscaling + Gemini AI QA.

Pipeline:
1. Real-ESRGAN 2x upscale (preprocessing) — gives K-means more pixels for smoother boundaries
2. V2 CIELAB K-means++ separation (core)
3. Downscale plate masks to original resolution
4. Gemini AI quality assessment (post-processing, non-blocking)
"""
import io
import json
import logging
import os
import zipfile

import numpy as np
from PIL import Image

import separate_v2 as v2

logger = logging.getLogger(__name__)

# ── Real-ESRGAN setup ──

_upsampler = None
_upsampler_init_attempted = False

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def _download_weights():
    """Download Real-ESRGAN weights if not present."""
    if os.path.exists(WEIGHTS_PATH):
        return True
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    logger.info("Downloading Real-ESRGAN weights to %s ...", WEIGHTS_PATH)
    try:
        import urllib.request
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        logger.info("Weights downloaded successfully.")
        return True
    except Exception as e:
        logger.error("Failed to download Real-ESRGAN weights: %s", e)
        return False


def _get_upsampler():
    """Lazy-init the Real-ESRGAN upsampler. Returns None if unavailable."""
    global _upsampler, _upsampler_init_attempted
    if _upsampler_init_attempted:
        return _upsampler
    _upsampler_init_attempted = True

    if not _download_weights():
        return None

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )

        use_half = torch.cuda.is_available()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available — Real-ESRGAN will run on CPU (slower)")

        _upsampler = RealESRGANer(
            scale=4,
            model_path=WEIGHTS_PATH,
            model=model,
            half=use_half,
            device=device,
            tile=256,
            tile_pad=10,
        )
        logger.info("Real-ESRGAN initialized on %s (half=%s)", device, use_half)
    except Exception as e:
        logger.error("Failed to initialize Real-ESRGAN: %s", e)
        _upsampler = None

    return _upsampler


def upscale_image(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Upscale an RGB numpy array 2x with Real-ESRGAN.
    Returns (upscaled_array, was_upscaled).
    Falls back to the original if Real-ESRGAN is unavailable.
    """
    upsampler = _get_upsampler()
    if upsampler is None:
        logger.warning("Real-ESRGAN unavailable — skipping upscale")
        return arr, False

    try:
        import cv2
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Real-ESRGAN expects BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(bgr, outscale=4)
        result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        logger.info("Upscaled from %s to %s", arr.shape[:2], result.shape[:2])
        return result, True
    except Exception as e:
        logger.error("Real-ESRGAN upscale failed: %s", e)
        return arr, False


# ── Anthropic AI QA ──

def anthropic_qa(original_arr: np.ndarray, composite_arr: np.ndarray) -> dict | None:
    """
    Send original + composite to Claude Sonnet for quality assessment.
    Returns analysis dict or None if unavailable/failed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping AI QA")
        return None

    try:
        import anthropic
        import base64

        client = anthropic.Anthropic()

        # Encode images as JPEG for smaller payloads
        orig_img = Image.fromarray(original_arr)
        comp_img = Image.fromarray(composite_arr)

        # Resize if too large
        max_dim = 1024
        for img in [orig_img, comp_img]:
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img.thumbnail((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

        orig_buf = io.BytesIO()
        orig_img.save(orig_buf, format="JPEG", quality=85)
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        comp_buf = io.BytesIO()
        comp_img.save(comp_buf, format="JPEG", quality=85)
        comp_b64 = base64.b64encode(comp_buf.getvalue()).decode()

        prompt = """Analyze this woodblock print color separation. Image 1 is the original, Image 2 is the separated composite.

Return ONLY valid JSON (no markdown fences) with this structure:
{
  "quality_score": <0-100>,
  "problem_regions": [
    {"x": <int>, "y": <int>, "width": <int>, "height": <int>, "description": "<string>"}
  ],
  "color_accuracy": "<string assessment>",
  "boundary_quality": "<string assessment>",
  "suggestions": ["<string>"],
  "summary": "<one sentence overall assessment>"
}

Be specific about any color bleeding, missed details, or boundary artifacts."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": orig_b64}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": comp_b64}},
                ],
            }],
        )

        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        analysis = json.loads(text)
        logger.info("Anthropic QA score: %s", analysis.get("quality_score"))
        return analysis

    except Exception as e:
        logger.error("Anthropic QA failed: %s", e)
        return None


# ── Preview result cache ──
_last_result_cache: dict = {}


# ── Main separation pipeline ──

def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=50,
             use_edges=True, edge_sigma=1.5, locked_colors=None, return_data=False,
             upscale=True, median_size=5, chroma_boost=1.3,
             shadow_threshold=8, highlight_threshold=95):
    """
    V4 separation: Real-ESRGAN upscale → V2 separation → Gemini QA.

    Extra args vs V2:
        upscale: whether to 4x upscale before separation (default True)
        median_size: median filter kernel size (odd, 1-11)
        chroma_boost: chroma multiplier for LAB a*/b* channels
        shadow_threshold: L* lower bound for content masking
        highlight_threshold: L* upper bound for content masking
    """
    # Load image
    if isinstance(input_path_or_array, str):
        img = Image.open(input_path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        arr = input_path_or_array

    orig_h, orig_w = arr.shape[:2]

    # Step 1: Upscale
    was_upscaled = False
    if upscale:
        arr_up, was_upscaled = upscale_image(arr)
    else:
        arr_up = arr

    # Step 2: Run V2 separation on (possibly upscaled) image
    result = v2.separate(
        arr_up,
        output_dir=None,
        n_plates=n_plates,
        dust_threshold=dust_threshold * (16 if was_upscaled else 1),  # scale dust threshold with 4x area
        use_edges=use_edges,
        edge_sigma=edge_sigma,
        locked_colors=locked_colors,
        return_data=True,
        median_size=median_size,
        chroma_boost=chroma_boost,
        shadow_threshold=shadow_threshold,
        highlight_threshold=highlight_threshold,
    )

    # Step 2.5: Fix white gaps — dilate all color plates to fill gaps, then recomposite
    # This is the key fix for the white lines / gaps between plate boundaries
    from scipy.ndimage import binary_dilation, binary_fill_holes
    from skimage.morphology import disk
    
    plate_masks = {}
    for name, plate_data in result["plates"].items():
        mask = plate_data["mask"].copy()
        # Fill internal holes
        mask = binary_fill_holes(mask)
        # Dilate to bleed under adjacent plates and edges — reduced to preserve detail
        mask = binary_dilation(mask, disk(2))
        plate_masks[name] = mask
    
    # Rebuild composite with dilated plates (lightest first, darkest on top)
    h_up, w_up = arr_up.shape[:2]
    comp = np.ones((h_up, w_up, 3), dtype=np.uint8) * 255
    plates_by_brightness = sorted(
        result["manifest"]["plates"],
        key=lambda p: sum(p["color"]),
        reverse=True  # lightest first
    )
    for plate_info in plates_by_brightness:
        name = plate_info["name"]
        comp[plate_masks[name]] = plate_info["color"]
    
    result["composite"] = Image.fromarray(comp)
    # Update plate data with dilated masks
    for name, mask in plate_masks.items():
        binary = np.where(mask, 0, 255).astype(np.uint8)
        result["plates"][name]["mask"] = mask
        result["plates"][name]["binary"] = binary
        result["plates"][name]["image"] = Image.fromarray(binary)

    # Step 3: Keep full upscaled resolution — that's the whole point of upscaling
    # Manifest already has the upscaled dimensions from V2

    # Step 4: Anthropic QA (non-blocking — skip on failure)
    composite_arr = np.array(result["composite"])
    ai_analysis = anthropic_qa(arr, composite_arr)
    result["manifest"]["ai_analysis"] = ai_analysis
    result["manifest"]["upscaled"] = was_upscaled

    if return_data:
        return result

    # Save files (same as V2)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result["composite"].save(os.path.join(output_dir, "composite.png"))
        for plate_info in result["manifest"]["plates"]:
            name = plate_info["name"]
            result["plates"][name]["image"].save(os.path.join(output_dir, f"{name}.png"))
            svg_path = os.path.join(output_dir, f"{name}.svg")
            v2.mask_to_svg(result["plates"][name]["mask"], svg_path, orig_w, orig_h)
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(result["manifest"], f, indent=2)

    return result["manifest"]


def build_preview_response(image_bytes, plates=4, dust=50, use_edges=True,
                           edge_sigma=1.5, locked_colors=None, upscale=True,
                           median_size=5, chroma_boost=1.3,
                           shadow_threshold=8, highlight_threshold=95):
    """Process image and return composite PNG bytes + manifest. Caches result for ZIP reuse."""
    import hashlib
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Cap input at 1000px for preview — 4x upscale gives 4000px output
    max_dim = 1000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    arr = np.array(img)
    result = separate(arr, n_plates=plates, dust_threshold=dust,
                      use_edges=use_edges, edge_sigma=edge_sigma,
                      locked_colors=locked_colors, return_data=True,
                      upscale=upscale, median_size=median_size,
                      chroma_boost=chroma_boost,
                      shadow_threshold=shadow_threshold,
                      highlight_threshold=highlight_threshold)

    # Cache the result keyed by image hash for ZIP reuse
    img_hash = hashlib.md5(image_bytes[:8192]).hexdigest()
    _last_result_cache["last"] = result
    _last_result_cache["hash"] = img_hash

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_zip_response(image_bytes, plates=4, dust=50, use_edges=True,
                       edge_sigma=1.5, locked_colors=None, upscale=True,
                       median_size=5, chroma_boost=1.3,
                       shadow_threshold=8, highlight_threshold=95):
    """Process image and return ZIP bytes. Uses cached preview result when available."""
    import hashlib

    # Check if we can reuse cached preview result
    img_hash = hashlib.md5(image_bytes[:8192]).hexdigest()
    if _last_result_cache.get("hash") == img_hash and "last" in _last_result_cache:
        logger.info("Using cached preview result for ZIP — skipping reprocessing")
        result = _last_result_cache["last"]
    else:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Cap at 1500px for zip — reduced from 4x to 2x upscale for speed
        max_dim = 1500
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        arr = np.array(img)
        result = separate(arr, n_plates=plates, dust_threshold=dust,
                          use_edges=use_edges, edge_sigma=edge_sigma,
                          locked_colors=locked_colors, return_data=True,
                          upscale=upscale, median_size=median_size,
                          chroma_boost=chroma_boost,
                          shadow_threshold=shadow_threshold,
                          highlight_threshold=highlight_threshold)

    # Get dimensions from composite
    comp_arr = np.array(result["composite"])
    h, w = comp_arr.shape[:2]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        comp_buf = io.BytesIO()
        result["composite"].save(comp_buf, format="PNG")
        zf.writestr("composite.png", comp_buf.getvalue())

        for plate_info in result["manifest"]["plates"]:
            name = plate_info["name"]
            plate_data = result["plates"][name]

            png_buf = io.BytesIO()
            plate_data["image"].save(png_buf, format="PNG")
            zf.writestr(f"{name}.png", png_buf.getvalue())

            svg_content = v2.mask_to_svg_string(plate_data["mask"], w, h)
            zf.writestr(f"{name}.svg", svg_content)

        zf.writestr("manifest.json", json.dumps(result["manifest"], indent=2))

    return zip_buf.getvalue()
