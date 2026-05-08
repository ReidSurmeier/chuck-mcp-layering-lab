"""
RunPod Serverless handler — wraps the color-separator backend.

Receives jobs via RunPod's queue, processes them through the existing
separation pipeline, and returns results as base64-encoded images.

Job input format:
{
    "image": "<base64 encoded image>",
    "version": "v20",
    "plates": 10,
    "dust": 5,
    "upscale": true,
    "chroma_boost": 1.3,
    "shadow_threshold": 8,
    "highlight_threshold": 95,
    "median_size": 3,
    "action": "preview" | "separate" | "plates"
}
"""
import os
os.environ["GPU_MODE"] = "1"

import runpod
import base64
import io
import json
import time

# Import the actual processing functions
from gpu_config import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def get_processor(version):
    """Get the right separation module for the version."""
    VERSION_MAP = {}
    try:
        import separate_v20 as v20
        import separate_v19 as v19
        import separate_v18 as v18
        import separate_v17 as v17
        import separate_v16 as v16
        import separate_v15 as v15
        VERSION_MAP.update({"v20": v20, "v19": v19, "v18": v18, "v17": v17, "v16": v16, "v15": v15})
    except ImportError:
        pass

    import separate_v14 as v14, separate_v13 as v13, separate_v12 as v12
    import separate_v11 as v11, separate_v10 as v10, separate_v9 as v9
    import separate_v6 as v6, separate_v5 as v5, separate_v4 as v4
    import separate as v3, separate_v2 as v2
    VERSION_MAP.update({
        "v14": v14, "v13": v13, "v12": v12, "v11": v11,
        "v10": v10, "v9": v9, "v6": v6, "v5": v5,
        "v4": v4, "v3": v3, "v2": v2,
    })
    return VERSION_MAP.get(version, VERSION_MAP.get("v20"))


def handler(job):
    """RunPod serverless handler — process one image separation job."""
    job_input = job["input"]

    # Decode image
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided"}

    image_bytes = base64.b64decode(image_b64)

    # Validate size
    if len(image_bytes) > 50 * 1024 * 1024:
        return {"error": "File too large. Max 50MB."}

    # Parameters
    version = job_input.get("version", "v20")
    action = job_input.get("action", "preview")
    plates = int(job_input.get("plates", 4))
    dust = int(job_input.get("dust", 5))
    upscale = job_input.get("upscale", UPSCALE_ENABLED)
    chroma_boost = float(job_input.get("chroma_boost", 1.3))
    shadow_threshold = int(job_input.get("shadow_threshold", 8))
    highlight_threshold = int(job_input.get("highlight_threshold", 95))
    median_size = int(job_input.get("median_size", 3))

    # Clamp inputs
    plates = max(2, min(plates, 60))
    dust = max(0, min(dust, 100))

    module = get_processor(version)
    if module is None:
        return {"error": f"Version {version} not available"}

    start = time.time()

    kwargs = {
        "plates": plates, "dust": dust,
        "chroma_boost": chroma_boost,
        "shadow_threshold": shadow_threshold,
        "highlight_threshold": highlight_threshold,
        "median_size": median_size,
        "upscale": upscale,
    }

    try:
        if action == "preview":
            composite_bytes, manifest = module.build_preview_response(image_bytes, **kwargs)
            result_b64 = base64.b64encode(composite_bytes).decode("utf-8")
            return {
                "composite": result_b64,
                "manifest": manifest,
                "time_seconds": round(time.time() - start, 2),
                "gpu_mode": True,
                "version": version,
            }

        elif action == "separate":
            zip_bytes = module.build_separate_response(image_bytes, **kwargs)
            result_b64 = base64.b64encode(zip_bytes).decode("utf-8")
            return {
                "zip": result_b64,
                "time_seconds": round(time.time() - start, 2),
                "gpu_mode": True,
                "version": version,
            }

        elif action == "plates":
            # Return individual plate thumbnails
            composite_bytes, manifest = module.build_preview_response(image_bytes, **kwargs)
            plates_data = module.build_plates_response(image_bytes, **kwargs) if hasattr(module, "build_plates_response") else None
            result_b64 = base64.b64encode(composite_bytes).decode("utf-8")
            return {
                "composite": result_b64,
                "manifest": manifest,
                "plates": plates_data,
                "time_seconds": round(time.time() - start, 2),
            }

        else:
            return {"error": f"Unknown action: {action}"}

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "time_seconds": round(time.time() - start, 2),
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
