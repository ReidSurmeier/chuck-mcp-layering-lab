"""
GPU configuration for cloud deployment (RunPod 5090 / any CUDA GPU).
Import this at the top of main.py to override CPU-bound defaults.

Usage: set env var GPU_MODE=1 to activate.
Auth:  set BACKEND_API_KEY env var to require X-API-Key header.
       set GPU_AUTH_PASSWORD for the frontend password gate.
"""
import os

GPU_MODE = os.environ.get("GPU_MODE", "0") == "1"

# ── Authentication ──
# When set, all /api/* requests must include X-API-Key header
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")
# Password users must enter in the frontend to unlock GPU processing
GPU_AUTH_PASSWORD = os.environ.get("GPU_AUTH_PASSWORD", "")
# Rate limiting (requests per minute per IP)
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10" if GPU_MODE else "30"))

if GPU_MODE:
    # ── Tuning for RTX 5090 (32GB VRAM, ~128GB system RAM on RunPod) ──

    # Allow 4 concurrent SAM requests (5090 has headroom)
    HEAVY_SEMAPHORE_LIMIT = 4

    # Memory check thresholds — relaxed for cloud
    MEMORY_REQUIRED_CACHED = 1.0      # GB (was 4.0)
    MEMORY_REQUIRED_UNCACHED = 2.0    # GB (was 11.0)

    # SAM model — env var SAM_WEIGHTS overrides default (for Docker pre-downloaded weights)
    SAM_MODEL = os.environ.get("SAM_WEIGHTS", "sam2.1_l.pt")

    # Force GPU for SAM (don't call .cpu())
    SAM_FORCE_CPU = False

    # Image dimension limits — unlocked for 8K+
    PREVIEW_MAX_DIM = 4000           # was 1000/1500
    MERGE_MAX_DIM = 4000             # was 1000/1500
    SEPARATE_MAX_DIM = 8192          # was 4000
    UPSCALE_PRE_MAX_DIM = 4000       # was 1200 (pre-SAM resize limit)
    UPSCALE_CACHE_MAX_DIM = 4000     # was 1000 (upscale_and_cache input limit)

    # Enable upscaling for v20 (safe on 32GB VRAM)
    UPSCALE_ENABLED = True

    # Upscale scale factor: 4 for GPU (RTX 4070 SUPER 12GB handles it fine)
    UPSCALE_SCALE = int(os.environ.get("UPSCALE_SCALE", "4"))

    # PIL pixel limit — match 8K (8192² = 67M pixels)
    MAX_IMAGE_PIXELS = 70_000_000

    # Uvicorn workers
    WORKERS = 4

else:
    # ── Local/CPU defaults (your 16GB desktop) ──
    HEAVY_SEMAPHORE_LIMIT = 1
    MEMORY_REQUIRED_CACHED = 4.0
    MEMORY_REQUIRED_UNCACHED = 11.0
    SAM_MODEL = os.environ.get("SAM_WEIGHTS", "sam2.1_t.pt")
    SAM_FORCE_CPU = True
    PREVIEW_MAX_DIM = 1500
    MERGE_MAX_DIM = 1500
    SEPARATE_MAX_DIM = 4000
    UPSCALE_PRE_MAX_DIM = 1200
    UPSCALE_CACHE_MAX_DIM = 1000
    UPSCALE_ENABLED = False  # v20 upscale off on 16GB
    UPSCALE_SCALE = int(os.environ.get("UPSCALE_SCALE", "2"))
    MAX_IMAGE_PIXELS = 50_000_000
    WORKERS = 2
