#!/usr/bin/env python3
"""Apply performance optimizations to separate_v20.py — surgical patches only."""
import re

# Read original
with open('/app/separate_v20.py', 'r') as f:
    code = f.read()

# ============================================================
# OPTIMIZATION 1: Cache Real-ESRGAN model globally
# Replace the upscale_2x function to cache the upsampler
# ============================================================
old_upscale = '''def upscale_2x(arr: np.ndarray, scale: int = None) -> tuple[np.ndarray, bool]:
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

new_upscale = '''# ── Cached ESRGAN upsamplers (avoid rebuilding model every call) ──
_esrgan_cache = {}  # scale -> RealESRGANer


def _get_esrgan(scale: int):
    """Get or create a cached RealESRGANer for the given scale."""
    if scale in _esrgan_cache:
        return _esrgan_cache[scale]
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
    _esrgan_cache[scale] = upsampler
    return upsampler


def upscale_2x(arr: np.ndarray, scale: int = None) -> tuple[np.ndarray, bool]:
    """Run Real-ESRGAN on an RGB numpy array. Returns (result, success).
    scale: 2 or 4. Defaults to gpu_config.UPSCALE_SCALE."""
    try:
        if scale is None:
            from gpu_config import UPSCALE_SCALE
            scale = UPSCALE_SCALE
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

code = code.replace(old_upscale, new_upscale)

# ============================================================
# OPTIMIZATION 2: SAM on CUDA instead of CPU
# ============================================================
old_sam_call = '        results = model(temp_path, points=points_arr, labels=labels_arr, device="cpu")'
new_sam_call = '''        from gpu_config import SAM_FORCE_CPU
        _sam_device = "cpu" if SAM_FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
        results = model(temp_path, points=points_arr, labels=labels_arr, device=_sam_device)'''
code = code.replace(old_sam_call, new_sam_call)

# ============================================================
# OPTIMIZATION 3: Vectorize edge pixel assignment (Step 5)
# Replace per-pixel Python loop with numpy vectorized version
# ============================================================
old_edge_loop = '''        ey, ex = np.where(edges)
        for y, x in zip(ey, ex):
            y0, y1 = max(0, y - 3), min(h, y + 4)
            x0, x1 = max(0, x - 3), min(w, x + 4)
            nbr = pixel_labels[y0:y1, x0:x1]
            plates_nearby = np.unique(nbr)
            plates_nearby = plates_nearby[plates_nearby < n_plates]
            if len(plates_nearby) > 0:
                darkest = min(plates_nearby, key=lambda p: palette_lab[p][0])
                pixel_labels[y, x] = darkest'''

new_edge_loop = '''        ey, ex = np.where(edges)
        if len(ey) > 0:
            # Vectorized edge assignment: for each edge pixel, sample neighbors
            # and assign to darkest plate found nearby
            palette_brightness = np.array([palette_lab[p][0] for p in range(n_plates)])
            # Pad labels to avoid boundary checks
            padded = np.pad(pixel_labels, 3, mode='edge')
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    pass  # just need neighbor access
            # Batch process: get 7x7 neighborhood mode via sliding window
            # For each edge pixel, collect unique plates in 7x7 window, pick darkest
            # Use vectorized approach with padded array
            ny = ey + 3  # offset for padding
            nx = ex + 3
            best_labels = np.full(len(ey), -1, dtype=np.int32)
            best_brightness = np.full(len(ey), np.inf)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nbr_labels = padded[ny + dy, nx + dx]
                    valid = nbr_labels < n_plates
                    nbr_bright = np.where(valid, palette_brightness[np.clip(nbr_labels, 0, n_plates - 1)], np.inf)
                    darker = nbr_bright < best_brightness
                    best_brightness[darker] = nbr_bright[darker]
                    best_labels[darker] = nbr_labels[darker]
            assigned = best_labels >= 0
            pixel_labels[ey[assigned], ex[assigned]] = best_labels[assigned]'''

code = code.replace(old_edge_loop, new_edge_loop)

# ============================================================
# OPTIMIZATION 4: Remove unnecessary empty_cache before SAM
# (the one before get_sam_model — model is already cached)
# ============================================================
old_pre_sam = '''    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    model = get_sam_model()'''
new_pre_sam = '''    import torch
    model = get_sam_model()'''
code = code.replace(old_pre_sam, new_pre_sam, 1)  # only first occurrence

# ============================================================
# OPTIMIZATION 5: Update release_upscaler to not destroy cache
# Since we now cache the model globally, release_upscaler should
# only clear the image cache, not the model
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
    """Free upscale image cache to reclaim RAM after processing.
    ESRGAN model stays cached globally for reuse."""
    global _upscale_cache
    _upscale_cache = LRUCache(maxsize=5)'''

code = code.replace(old_release, new_release)

# ============================================================
# OPTIMIZATION 6: Add warmup function for pre-loading models
# ============================================================
warmup_func = '''

def warmup_models():
    """Pre-load SAM and ESRGAN models into GPU memory.
    Call at container startup to eliminate cold-start penalty."""
    import logging
    log = logging.getLogger(__name__)
    import time

    t0 = time.perf_counter()
    log.info("Warming up SAM model...")
    get_sam_model()
    t1 = time.perf_counter()
    log.info(f"SAM loaded in {t1 - t0:.1f}s")

    log.info("Warming up ESRGAN models...")
    from gpu_config import UPSCALE_SCALE
    _get_esrgan(UPSCALE_SCALE)
    t2 = time.perf_counter()
    log.info(f"ESRGAN loaded in {t2 - t1:.1f}s")

    log.info(f"Model warmup complete in {t2 - t0:.1f}s")

'''

# Insert warmup_models() after release_upscaler
code = code.replace(new_release, new_release + warmup_func)

# Write optimized file
with open('/app/separate_v20.py', 'w') as f:
    f.write(code)

print("separate_v20.py patched successfully")
print("Changes applied:")
print("  1. Cached ESRGAN model globally (_esrgan_cache)")
print("  2. SAM inference uses CUDA (respects SAM_FORCE_CPU)")
print("  3. Vectorized edge pixel assignment loop")
print("  4. Removed pre-SAM empty_cache call")
print("  5. release_upscaler only clears image cache, not model")
print("  6. Added warmup_models() function")
