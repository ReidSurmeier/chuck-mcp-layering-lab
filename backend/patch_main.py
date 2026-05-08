"""
Patch script for main.py — adds GPU status endpoint and model warmup on startup.
Run inside container: python /tmp/patch_main.py
"""

FILEPATH = "/app/main.py"

with open(FILEPATH, "r") as f:
    content = f.read()

original = content

# ============================================================
# PATCH 1: Add /api/gpu-status diagnostic endpoint
# Insert before the auto-optimize endpoint
# ============================================================
gpu_status_endpoint = '''
@app.get("/api/gpu-status")
async def gpu_status():
    """Diagnostic endpoint: report GPU state, model locations, VRAM usage."""
    import torch
    status = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2) if torch.cuda.is_available() else 0,
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0,
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2) if torch.cuda.is_available() else 0,
    }

    # Check SAM device
    try:
        if v20 and v20._sam_model is not None:
            sam_params = list(v20._sam_model.model.parameters())
            if sam_params:
                status["sam_device"] = str(sam_params[0].device)
                status["sam_loaded"] = True
            else:
                status["sam_device"] = "unknown"
                status["sam_loaded"] = True
        else:
            status["sam_device"] = "not_loaded"
            status["sam_loaded"] = False
    except Exception as e:
        status["sam_device"] = f"error: {e}"
        status["sam_loaded"] = False

    # Check ESRGAN device
    try:
        if v20 and hasattr(v20, '_esrgan_models') and v20._esrgan_models:
            for scale, upsampler in v20._esrgan_models.items():
                params = list(upsampler.model.parameters())
                if params:
                    status[f"esrgan_{scale}x_device"] = str(params[0].device)
            status["esrgan_loaded"] = True
        else:
            status["esrgan_loaded"] = False
    except Exception as e:
        status["esrgan_loaded"] = False
        status["esrgan_error"] = str(e)

    from gpu_config import GPU_MODE, SAM_FORCE_CPU, UPSCALE_ENABLED, UPSCALE_SCALE
    status["gpu_mode"] = GPU_MODE
    status["sam_force_cpu"] = SAM_FORCE_CPU
    status["upscale_enabled"] = UPSCALE_ENABLED
    status["upscale_scale"] = UPSCALE_SCALE

    return Response(content=json.dumps(status), media_type="application/json")


'''

# Insert before @app.post("/api/auto-optimize")
content = content.replace(
    '@app.post("/api/auto-optimize")',
    gpu_status_endpoint + '@app.post("/api/auto-optimize")'
)

# ============================================================
# PATCH 2: Add model warmup on startup event
# ============================================================
startup_event = '''

@app.on_event("startup")
async def startup_warmup():
    """Pre-load SAM and ESRGAN models into GPU memory at container start."""
    import asyncio
    loop = asyncio.get_event_loop()
    if v20 is not None:
        await loop.run_in_executor(None, v20.warmup_models)


'''

# Insert after app middleware setup, before the first endpoint
# Find a good insertion point — after the API key middleware class
content = content.replace(
    'class APIKeyMiddleware(BaseHTTPMiddleware):',
    startup_event + 'class APIKeyMiddleware(BaseHTTPMiddleware):'
)

# ============================================================
# PATCH 3: Update SSE progress handler to forward ETA
# The progress_callback now receives (stage, pct, eta)
# ============================================================
old_on_progress = '''    def on_progress(stage: str, pct: int):
        progress_events.append({"stage": stage, "pct": pct})'''

new_on_progress = '''    def on_progress(stage: str, pct: int, eta: float = -1):
        progress_events.append({"stage": stage, "pct": pct, "eta": round(eta, 1)})'''

content = content.replace(old_on_progress, new_on_progress)

# ============================================================
# Verify
# ============================================================
changes = []
if '/api/gpu-status' in content:
    changes.append("GPU status endpoint")
if 'startup_warmup' in content:
    changes.append("Model warmup on startup")
if 'eta: float = -1' in content:
    changes.append("ETA in SSE progress")

if content == original:
    print("ERROR: No patches applied!")
    exit(1)

print(f"Patches applied: {', '.join(changes)}")

with open(FILEPATH, "w") as f:
    f.write(content)

print(f"Written {len(content)} bytes to {FILEPATH}")
