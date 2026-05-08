#!/usr/bin/env python3
"""Patch main.py to add model warmup on startup and update health endpoint."""

with open('/app/main.py', 'r') as f:
    code = f.read()

# ============================================================
# PATCH 1: Add lifespan with model warmup
# ============================================================
old_app_line = 'app = FastAPI(title="Woodblock Color Separation API")'
new_app_block = '''from contextlib import asynccontextmanager
import logging as _logging

_log = _logging.getLogger("colorsep")


@asynccontextmanager
async def lifespan(app):
    """Pre-warm GPU models on startup to eliminate cold-start penalty."""
    _log.info("Starting model warmup...")
    try:
        from separate_v20 import warmup_models
        import asyncio
        await asyncio.get_event_loop().run_in_executor(None, warmup_models)
        _log.info("Model warmup complete")
    except Exception as e:
        _log.warning(f"Model warmup failed (non-fatal): {e}")
    yield


app = FastAPI(title="Woodblock Color Separation API", lifespan=lifespan)'''

code = code.replace(old_app_line, new_app_block)

# ============================================================
# PATCH 2: Update health endpoint to include model_v20_loaded
# ============================================================
old_health_return = '''    return {
        "status": "ok" if ok else "degraded",
        "sam_cached": sam_cached,
        "sam_ready": ok,
        "sam_memory_note": mem_msg if not ok else None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_mem_mb,
        "system_memory_available_gb": round(mem.available / (1024**3), 1),
        "sam_plates_max": SAM_PLATES_MAX,
        "plates_60_supported": SAM_PLATES_MAX >= 60,
        "upscale_enabled": UPSCALE_ENABLED,
        "upscale_scale": UPSCALE_SCALE if UPSCALE_ENABLED else None,
    }'''

new_health_return = '''    # Check if ESRGAN is also cached
    esrgan_cached = False
    try:
        from separate_v20 import _esrgan_cache
        esrgan_cached = len(_esrgan_cache) > 0
    except ImportError:
        pass

    model_v20_loaded = sam_cached and (esrgan_cached or not UPSCALE_ENABLED)

    return {
        "status": "ok" if ok else "degraded",
        "model_v20_loaded": model_v20_loaded,
        "sam_cached": sam_cached,
        "esrgan_cached": esrgan_cached,
        "sam_ready": ok,
        "sam_memory_note": mem_msg if not ok else None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_mem_mb,
        "system_memory_available_gb": round(mem.available / (1024**3), 1),
        "sam_plates_max": SAM_PLATES_MAX,
        "plates_60_supported": SAM_PLATES_MAX >= 60,
        "upscale_enabled": UPSCALE_ENABLED,
        "upscale_scale": UPSCALE_SCALE if UPSCALE_ENABLED else None,
    }'''

code = code.replace(old_health_return, new_health_return)

with open('/app/main.py', 'w') as f:
    f.write(code)

print("main.py patched successfully")
print("Changes applied:")
print("  1. Added lifespan with model warmup on startup")
print("  2. Updated health endpoint with model_v20_loaded field")
