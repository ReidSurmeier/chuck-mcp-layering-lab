"""
Patch script for separate_v20.py — applies GPU optimization changes.
Run inside container: python /tmp/patch_v20.py
"""
import re

FILEPATH = "/app/separate_v20.py"

with open(FILEPATH, "r") as f:
    content = f.read()

original = content

# ============================================================
# PATCH 1: Fix SAM device — use gpu_config instead of hardcoded "cpu"
# Line ~343: device="cpu" -> dynamic device
# ============================================================
old_sam_call = 'results = model(temp_path, points=points_arr, labels=labels_arr, device="cpu")'
new_sam_call = '''from gpu_config import SAM_FORCE_CPU
        _sam_device = "cpu" if SAM_FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
        results = model(temp_path, points=points_arr, labels=labels_arr, device=_sam_device)'''
content = content.replace(old_sam_call, new_sam_call)

# ============================================================
# PATCH 2: Cache Real-ESRGAN model instance (avoid re-instantiation)
# Replace the upscale_2x function with a cached version
# ============================================================
old_upscale_fn = '''def upscale_2x(arr: np.ndarray, scale: int = None) -> tuple[np.ndarray, bool]:
    """Run Real-ESRGAN on an RGB numpy array. Returns (result, success).
    scale: 2 or 4. Defaults to gpu_config.UPSCALE_SCALE."""
    try:
        if scale is None:
            from gpu_config import UPSCALE_SCALE
            scale = UPSCALE_SCALE
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if scale == 4:
            weights_name = "RealESRGAN_x4plus.pth"
        else:
            weights_name = "RealESRGAN_x2plus.pth"
            scale = 2  # normalize

        weights_path = os.path.join(os.path.dirname(__file__), "weights", weights_name)
        if not os.path.exists(weights_path):
            # Fallback to 2x if 4x weights missing
            if scale == 4:
                weights_path = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x2plus.pth")
                scale = 2
                if not os.path.exists(weights_path):
                    return arr, False
            else:
                return arr, False

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(
            scale=scale, model_path=weights_path, model=model,
            half=torch.cuda.is_available(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            tile=256, tile_pad=10
        )
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(bgr, outscale=scale)
        result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        del upsampler, model
        torch.cuda.empty_cache()
        gc.collect()
        return result, True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Upscale failed: {e}")
        return arr, False'''

new_upscale_fn = '''# ── Cached Real-ESRGAN model instances ──
_esrgan_models: dict[int, object] = {}  # scale -> RealESRGANer


def _get_esrgan(scale: int):
    """Return cached RealESRGANer instance for given scale."""
    global _esrgan_models
    if scale in _esrgan_models:
        return _esrgan_models[scale]

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if scale == 4:
        weights_name = "RealESRGAN_x4plus.pth"
    else:
        weights_name = "RealESRGAN_x2plus.pth"
        scale = 2

    weights_path = os.path.join(os.path.dirname(__file__), "weights", weights_name)
    if not os.path.exists(weights_path):
        if scale == 4:
            weights_path = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x2plus.pth")
            scale = 2
            if not os.path.exists(weights_path):
                return None
        else:
            return None

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale, model_path=weights_path, model=model,
        half=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        tile=256, tile_pad=10
    )
    _esrgan_models[scale] = upsampler
    return upsampler


def upscale_2x(arr: np.ndarray, scale: int = None) -> tuple[np.ndarray, bool]:
    """Run Real-ESRGAN on an RGB numpy array. Returns (result, success).
    scale: 2 or 4. Defaults to gpu_config.UPSCALE_SCALE.
    Uses cached model instance to avoid re-loading weights each call."""
    try:
        if scale is None:
            from gpu_config import UPSCALE_SCALE
            scale = UPSCALE_SCALE
        import torch

        upsampler = _get_esrgan(scale)
        if upsampler is None:
            return arr, False

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(bgr, outscale=scale)
        result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return result, True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Upscale failed: {e}")
        return arr, False'''

content = content.replace(old_upscale_fn, new_upscale_fn)

# ============================================================
# PATCH 3: Update release_upscaler to clear cached models
# ============================================================
old_release = '''def release_upscaler():
    """Free upscale cache to reclaim RAM after processing."""
    global _upscale_cache
    _upscale_cache = LRUCache(maxsize=5)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass'''

new_release = '''def release_upscaler():
    """Free upscale result cache (keeps model loaded for reuse)."""
    global _upscale_cache
    _upscale_cache = LRUCache(maxsize=5)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass'''

content = content.replace(old_release, new_release)

# ============================================================
# PATCH 4: Add warmup function for pre-loading models at startup
# ============================================================
warmup_fn = '''

def warmup_models():
    """Pre-load SAM and Real-ESRGAN models into GPU memory.
    Call at container startup to eliminate cold-start latency."""
    import logging
    log = logging.getLogger(__name__)
    log.info("Warming up models...")

    # Load SAM
    try:
        model = get_sam_model()
        log.info(f"SAM loaded: {type(model)}")
    except Exception as e:
        log.warning(f"SAM warmup failed: {e}")

    # Load Real-ESRGAN (2x and 4x if weights exist)
    for scale in (2, 4):
        try:
            esrgan = _get_esrgan(scale)
            if esrgan:
                log.info(f"ESRGAN {scale}x loaded")
        except Exception as e:
            log.warning(f"ESRGAN {scale}x warmup failed: {e}")

    log.info("Model warmup complete")
'''

# Insert warmup before build_preview_response
content = content.replace(
    "def build_preview_response(",
    warmup_fn + "\ndef build_preview_response("
)

# ============================================================
# PATCH 5: Add ETA to progress_callback calls
# ============================================================
# Update the report() inner function in separate() to include ETA
old_report = '''    def report(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)'''

new_report = '''    _start_time = time.time()

    def report(stage, pct):
        if progress_callback:
            elapsed = time.time() - _start_time
            if pct > 0:
                eta = max(0, (elapsed / pct) * (100 - pct))
            else:
                eta = -1  # unknown
            progress_callback(stage, pct, eta)'''

content = content.replace(old_report, new_report)

# Update progress_callback signature in separate() function signature
# The callback now accepts (stage, pct, eta)

# ============================================================
# Verify patches applied
# ============================================================
changes = []
if 'SAM_FORCE_CPU' in content and 'device="cpu"' not in content.split('results = model(')[0].split('results = model(')[-1] if 'results = model(' in content else True:
    changes.append("SAM GPU device fix")
if '_esrgan_models' in content:
    changes.append("ESRGAN model caching")
if 'warmup_models' in content:
    changes.append("Model warmup function")
if '_start_time = time.time()' in content:
    changes.append("ETA in progress callback")

if content == original:
    print("ERROR: No patches applied!")
    exit(1)

print(f"Patches applied: {', '.join(changes)}")

with open(FILEPATH, "w") as f:
    f.write(content)

print(f"Written {len(content)} bytes to {FILEPATH}")
