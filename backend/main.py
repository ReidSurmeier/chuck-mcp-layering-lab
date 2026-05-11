"""
FastAPI backend for woodblock color separation.
Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""
import asyncio
import re as _re
import base64
import gc
import hashlib
import hmac
import io
import json
import os
import secrets
import tempfile
import time
from pathlib import Path

# Temp directory for streaming results — cleaned up after 30 minutes
_RESULT_DIR = Path(tempfile.gettempdir()) / "colorsep_results"
_RESULT_DIR.mkdir(exist_ok=True)
_RESULT_MAX_AGE_S = 1800  # 30 minutes

import numpy as np
import psutil
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse

from pydantic import BaseModel

from gpu_config import (
    BACKEND_API_KEY,
    GPU_AUTH_PASSWORD,
    HEAVY_SEMAPHORE_LIMIT,
    MAX_IMAGE_PIXELS,
    MEMORY_REQUIRED_CACHED,
    MEMORY_REQUIRED_UNCACHED,
    RATE_LIMIT_PER_MINUTE,
    UPSCALE_ENABLED,
    UPSCALE_SCALE,
)

# HEIF/HEIC image support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Concurrency limit: controlled by gpu_config (1 on CPU, 4 on GPU)
_heavy_semaphore = asyncio.Semaphore(HEAVY_SEMAPHORE_LIMIT)

# Historic versions v2..v19 pruned 2026-05-10 (woodblock fork).
# Only v20 retained; v21_mokuhanga added in upcoming commits.
try:
    import separate_v20 as v20  # noqa: E402
except ImportError:
    v20 = None
try:
    import auto_optimize
except ImportError:
    auto_optimize = None
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # From gpu_config  # noqa: F405

# Magic-byte signatures for allowed image formats
_IMAGE_MAGIC: list[tuple[bytes, bytes | None]] = [
    (b"\x89PNG", None),       # PNG
    (b"\xff\xd8\xff", None),  # JPEG
    (b"GIF87a", None),        # GIF87
    (b"GIF89a", None),        # GIF89
    (b"RIFF", b"WEBP"),      # WebP  (bytes 8-11 = "WEBP")
    (b"II*\x00", None),      # TIFF LE
    (b"MM\x00*", None),      # TIFF BE
    # HEIF/HEIC handled separately below via ftyp brand check
    (b"BM", None),           # BMP
]


def _check_magic(data: bytes) -> bool:
    """Return True if data starts with a recognised image magic sequence."""
    for magic, extra in _IMAGE_MAGIC:
        if data[:len(magic)] == magic:
            if extra is None:
                return True
            # For WebP: bytes 8-11 must equal the extra tag
            if data[8:12] == extra:
                return True
    # HEIF/HEIC: ftyp box at offset 4 with validated brand
    _HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS:
        return True
    return False


async def validate_upload(image_bytes: bytes):
    """Validate uploaded image. Returns error response or None if valid.

    Checks performed:
    1. File-size limit (50 MB).
    2. Magic-byte signature matches a known image format.
    3. PIL can fully decode the image (catches truncated / corrupt files).
    4. Image dimensions are within MAX_IMAGE_PIXELS (set on PIL globally).
    EXIF data is stripped from the returned bytes via a re-save; callers
    that need clean bytes should call strip_exif() themselves.
    """
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"error": "File too large. Max 50MB."})

    if not _check_magic(image_bytes):
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # Force full decode; catches truncated data
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            return JSONResponse(
                status_code=413,
                content={"error": f"Image too large ({w}x{h}). Max {MAX_IMAGE_PIXELS} pixels."},
            )
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})
    return None


def strip_exif(image_bytes: bytes) -> bytes:
    """Return image bytes with EXIF/metadata stripped (re-saved as PNG)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes  # Best-effort; original bytes if strip fails


app = FastAPI(title="Woodblock Color Separation API")

from analytics import RequestLog  # noqa: E402
from analytics_api import router as _analytics_router  # noqa: E402
from job_routes import router as _job_router  # noqa: E402
from routes.sam_v23 import router as _sam_v23_router  # noqa: E402
app.include_router(_analytics_router)
app.include_router(_job_router)
app.include_router(_sam_v23_router, prefix="/api")

from job_queue import cleanup_expired_jobs  # noqa: E402

@app.on_event("startup")
async def _startup():
    asyncio.create_task(cleanup_expired_jobs())

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tools.reidsurmeier.wtf",
        "https://colorseparator.reidsurmeier.wtf",
        "https://colorseperator.reidsurmeier.wtf",
        "http://localhost:3008",
        "http://localhost:3003",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Accept"],
)

# ── Security headers middleware ──


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS: tell browsers to use HTTPS for 1 year (backend is API-only, proxied via Cloudflare)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ── API Key authentication middleware ──
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require X-API-Key header when BACKEND_API_KEY is configured."""
    async def dispatch(self, request: Request, call_next):
        if not BACKEND_API_KEY:  # noqa: F405
            return await call_next(request)  # No key set = no auth (local dev)
        if request.url.path == "/api/health":
            return await call_next(request)  # Health check always open
        key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(key, BACKEND_API_KEY):  # noqa: F405
            return JSONResponse(status_code=401, content={"error": "Invalid or missing API key."})
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)

# ── Rate limiting ──
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"])  # noqa: F405
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError as _slowapi_err:
    raise ImportError(
        "slowapi is required for rate limiting. "
        "Install it with: pip install slowapi"
    ) from _slowapi_err

# ── GPU auth session token store ──
# Maps token -> expiry timestamp (unix seconds). Module-level; survives request lifetime.
_GPU_TOKENS: dict[str, float] = {}
_GPU_TOKEN_TTL = 86400  # 24 hours


def _purge_expired_tokens() -> None:
    """Remove expired tokens from the in-memory store."""
    now = time.time()
    expired = [tok for tok, exp in _GPU_TOKENS.items() if exp < now]
    for tok in expired:
        del _GPU_TOKENS[tok]


# ── GPU auth password verification endpoint ──


class AuthRequest(BaseModel):
    password: str


@app.post("/api/auth/verify")
async def verify_gpu_password(body: AuthRequest):
    """Verify password to unlock GPU features on the frontend."""
    if not GPU_AUTH_PASSWORD:  # noqa: F405
        return {"authorized": True}  # No password set = always authorized
    if hmac.compare_digest(body.password, GPU_AUTH_PASSWORD):  # noqa: F405
        _purge_expired_tokens()
        token = secrets.token_urlsafe(32)
        _GPU_TOKENS[token] = time.time() + _GPU_TOKEN_TTL
        return {"authorized": True, "token": token}
    return JSONResponse(status_code=403, content={"authorized": False, "error": "Invalid password."})


@app.post("/api/auth/check")
async def check_gpu_token(body: dict):
    """Check if a GPU auth token is still valid."""
    if not GPU_AUTH_PASSWORD:
        return {"valid": True}
    _purge_expired_tokens()
    token = body.get("token", "")
    if not token:
        return {"valid": False}
    expiry = _GPU_TOKENS.get(token)
    valid = expiry is not None and time.time() < expiry
    return {"valid": valid}


def parse_locked_colors(raw: str | None) -> list[list[int]] | None:
    if not raw:
        return None
    try:
        colors = json.loads(raw)
        if isinstance(colors, list) and len(colors) > 0:
            return colors
    except (json.JSONDecodeError, TypeError):
        pass
    return None


VERSION_MAP = {
    k: mod for k, mod in {
        "v20": v20,
    }.items() if mod is not None
}

PLATES_MIN = 2
PLATES_MAX = 60
SAM_PLATES_MAX = 60  # SAM versions need O(n_plates * h * w) memory — cap to prevent OOM
DUST_MIN = 5
DUST_MAX = 100
SAM_TIMEOUT_SECONDS = 600  # Request-level timeout for SAM processing (600s for 60 plates)


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def _clamp_float_params(params: dict) -> dict:
    """Clamp all numeric processing params to safe ranges to prevent resource exhaustion."""
    clamps = {
        "edge_sigma": (0.1, 10.0),
        "sigma_s": (1.0, 300.0),
        "sigma_r": (0.01, 1.0),
        "meanshift_sp": (1, 60),
        "meanshift_sr": (1, 80),
        "chroma_boost": (0.5, 3.0),
        "median_size": (1, 15),
        "shadow_threshold": (0, 50),
        "highlight_threshold": (50, 100),
        "detail_strength": (0.0, 2.0),
        "n_segments": (100, 10000),
        "compactness": (1, 100),
        "crf_spatial": (1, 20),
        "crf_color": (1, 50),
        "crf_compat": (1, 50),
    }
    for key, (lo, hi) in clamps.items():
        if key in params:
            params[key] = type(lo)(max(lo, min(params[key], hi)))
    return params


VALID_VERSIONS = set(VERSION_MAP.keys())


def _cleanup_gpu():
    """Force garbage collection and free GPU memory. Call after every request."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_module(version: str):
    if version not in VALID_VERSIONS:
        return None  # caller must handle
    return VERSION_MAP.get(version)


SAM_VERSIONS = ("v20",)


def check_memory_for_sam(n_colors: int = 20):
    """Check if enough memory is available for SAM processing.
    Returns (ok: bool, message: str)

    For n_colors > 20, requires 4GB extra headroom to cover the larger
    plate-score tensor (n_plates * h * w * float32).
    """
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    available_gb = (mem.available + swap.free) / (1024**3)

    # SAM needs ~10GB peak, but if model is already cached, needs much less
    sam_cached = False
    try:
        from separate_v20 import _sam_model
        sam_cached = _sam_model is not None
    except ImportError:
        pass

    required_gb = MEMORY_REQUIRED_CACHED if sam_cached else MEMORY_REQUIRED_UNCACHED

    # High plate counts require extra headroom for plate_scores tensor
    if n_colors > 20:
        required_gb = required_gb + 4.0

    if available_gb < required_gb:
        return False, f"Insufficient memory: {available_gb:.1f}GB available, need {required_gb:.1f}GB."
    return True, "OK"


@app.get("/api/health")
async def health():
    mem = psutil.virtual_memory()
    sam_cached = False
    try:
        from separate_v20 import _sam_model
        sam_cached = _sam_model is not None
    except ImportError:
        pass

    # GPU availability check
    gpu_available = False
    gpu_name = None
    gpu_mem_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
    except Exception:
        pass

    ok, mem_msg = check_memory_for_sam()

    return {
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
    }


@app.post("/api/preview")
async def preview(
    request: Request,
    image: UploadFile = File(...),
    plates: int = Form(3),
    dust: int = Form(20),
    use_edges: bool = Form(True),
    edge_sigma: float = Form(1.5),
    locked_colors: str | None = Form(None),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    median_size: int = Form(5),
    chroma_boost: float = Form(1.3),
    shadow_threshold: int = Form(8),
    highlight_threshold: int = Form(95),
    n_segments: int = Form(3000),
    compactness: int = Form(15),
    crf_spatial: int = Form(3),
    crf_color: int = Form(13),
    crf_compat: int = Form(10),
    sigma_s: float = Form(100),
    sigma_r: float = Form(0.5),
    meanshift_sp: int = Form(15),
    meanshift_sr: int = Form(30),
    detail_strength: float = Form(0.5),
):
    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        return err
    image_bytes = strip_exif(image_bytes)
    # ── Analytics setup ──────────────────────────────────────────────────
    _rlog_pv = RequestLog("/api/preview")
    _rlog_pv.set_client(
        request.headers.get("user-agent", ""),
        request.client.host if request.client else "",
    )
    try:
        _img_probe_pv = Image.open(io.BytesIO(image_bytes))
        _rlog_pv.set_input(
            w=_img_probe_pv.size[0], h=_img_probe_pv.size[1],
            kb=len(image_bytes) / 1024, fmt=_img_probe_pv.format or "PNG",
        )
    except Exception:
        pass
    _rlog_pv.set_params(plates=plates, dust=dust, version=version,
                        upscale=upscale, upscale_scale=upscale_scale)
    # ─────────────────────────────────────────────────────────────────────
    locked = parse_locked_colors(locked_colors)
    plates = _clamp(plates, PLATES_MIN, PLATES_MAX)
    dust = _clamp(dust, DUST_MIN, DUST_MAX)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    _clamped = _clamp_float_params(dict(
        edge_sigma=edge_sigma, sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        chroma_boost=chroma_boost, median_size=median_size,
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        detail_strength=detail_strength, n_segments=n_segments, compactness=compactness,
        crf_spatial=crf_spatial, crf_color=crf_color, crf_compat=crf_compat,
    ))
    edge_sigma, sigma_s, sigma_r = _clamped["edge_sigma"], _clamped["sigma_s"], _clamped["sigma_r"]
    meanshift_sp, meanshift_sr = _clamped["meanshift_sp"], _clamped["meanshift_sr"]
    chroma_boost, median_size = _clamped["chroma_boost"], _clamped["median_size"]
    shadow_threshold, highlight_threshold = _clamped["shadow_threshold"], _clamped["highlight_threshold"]
    detail_strength = _clamped["detail_strength"]
    n_segments, compactness = _clamped["n_segments"], _clamped["compactness"]
    crf_spatial, crf_color, crf_compat = _clamped["crf_spatial"], _clamped["crf_color"], _clamped["crf_compat"]
    mod = get_module(version)
    if mod is None:
        _rlog_pv.set_error(f"Unknown version: {version}", "ValueError")
        _rlog_pv.finish(status=400)
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown version: {version}. Valid: {sorted(VALID_VERSIONS)}"},
        )

    _rlog_pv.set_image_hash(hashlib.sha256(image_bytes).hexdigest()[:16])

    kwargs: dict = dict(
        image_bytes=image_bytes, plates=plates, dust=dust,
        use_edges=use_edges, edge_sigma=edge_sigma, locked_colors=locked,
    )
    # v20 is the only version; all requests go through the SAM path
    kwargs["shadow_threshold"] = shadow_threshold
    kwargs["highlight_threshold"] = highlight_threshold
    kwargs["median_size"] = median_size
    kwargs["chroma_boost"] = chroma_boost
    kwargs["upscale"] = upscale if UPSCALE_ENABLED else False
    kwargs["upscale_scale"] = upscale_scale

    if version in SAM_VERSIONS:
        plates = _clamp(plates, PLATES_MIN, SAM_PLATES_MAX)
        kwargs["plates"] = plates
        ok, msg = check_memory_for_sam(n_colors=plates)
        if not ok:
            _rlog_pv.set_error(msg, "MemoryError")
            _rlog_pv.finish(status=503)
            return JSONResponse(
                status_code=503,
                content={"error": msg, "code": "MEMORY_LOW", "retry_after_seconds": 30, "plates_completed": 0},
                headers={"Retry-After": "30"},
            )
        try:
            async with _heavy_semaphore:
                with _rlog_pv.stage("separation"):
                    composite_bytes, manifest = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: mod.build_preview_response(**kwargs)
                        ),
                        timeout=SAM_TIMEOUT_SECONDS,
                    )
        except asyncio.TimeoutError:
            _cleanup_gpu()
            _rlog_pv.set_error(f"Timed out after {SAM_TIMEOUT_SECONDS}s", "TimeoutError")
            _rlog_pv.finish(status=504)
            return JSONResponse(status_code=504, content={
                "error": f"Processing timed out after {SAM_TIMEOUT_SECONDS}s. Try fewer plates or a smaller image.",
                "code": "TIMEOUT",
            })
        except Exception as e:
            _cleanup_gpu()
            _rlog_pv.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            _rlog_pv.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Separation failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })
        finally:
            _cleanup_gpu()
    else:
        try:
            with _rlog_pv.stage("separation"):
                composite_bytes, manifest = mod.build_preview_response(**kwargs)
        except Exception as e:
            _rlog_pv.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            _rlog_pv.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Separation failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })

    try:
        _rlog_pv.set_output(
            w=manifest.get("width"), h=manifest.get("height"),
            plates_returned=len(manifest.get("plates", [])),
        )
    except Exception:
        pass
    _rlog_pv.finish(status=200)

    return Response(
        content=composite_bytes,
        media_type="image/png",
        headers={"X-Manifest": json.dumps(manifest)},
    )


@app.post("/api/preview-stream")
async def preview_stream(
    image: UploadFile = File(...),
    plates: int = Form(3),
    dust: int = Form(20),
    use_edges: bool = Form(True),
    edge_sigma: float = Form(1.5),
    locked_colors: str | None = Form(None),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    median_size: int = Form(5),
    chroma_boost: float = Form(1.3),
    shadow_threshold: int = Form(8),
    highlight_threshold: int = Form(95),
    n_segments: int = Form(3000),
    compactness: int = Form(15),
    crf_spatial: int = Form(3),
    crf_color: int = Form(13),
    crf_compat: int = Form(10),
    sigma_s: float = Form(100),
    sigma_r: float = Form(0.5),
    meanshift_sp: int = Form(15),
    meanshift_sr: int = Form(30),
    detail_strength: float = Form(0.5),
):
    """Stream progress events via SSE for all versions, then send final result.

    Routing all versions through this endpoint ensures Cloudflare tunnel (100s
    timeout) never kills a slow request — the SSE heartbeats keep the connection
    alive regardless of processing time.
    """
    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        return err
    image_bytes = strip_exif(image_bytes)
    locked = parse_locked_colors(locked_colors)
    plates = _clamp(plates, PLATES_MIN, PLATES_MAX)
    dust = _clamp(dust, DUST_MIN, DUST_MAX)

    # ── Analytics setup ──────────────────────────────────────────────────
    _rlog_ps = RequestLog("/api/preview-stream")
    try:
        _img_probe_ps = Image.open(io.BytesIO(image_bytes))
        _rlog_ps.set_input(
            w=_img_probe_ps.size[0], h=_img_probe_ps.size[1],
            kb=len(image_bytes) / 1024, fmt=_img_probe_ps.format or "PNG",
        )
    except Exception:
        pass
    _rlog_ps.set_params(plates=plates, dust=dust, version=version,
                        upscale=upscale, upscale_scale=upscale_scale)
    # ─────────────────────────────────────────────────────────────────────
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    _clamped = _clamp_float_params(dict(
        edge_sigma=edge_sigma, sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        chroma_boost=chroma_boost, median_size=median_size,
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        detail_strength=detail_strength, n_segments=n_segments, compactness=compactness,
        crf_spatial=crf_spatial, crf_color=crf_color, crf_compat=crf_compat,
    ))
    edge_sigma, sigma_s, sigma_r = _clamped["edge_sigma"], _clamped["sigma_s"], _clamped["sigma_r"]
    meanshift_sp, meanshift_sr = _clamped["meanshift_sp"], _clamped["meanshift_sr"]
    chroma_boost, median_size = _clamped["chroma_boost"], _clamped["median_size"]
    shadow_threshold, highlight_threshold = _clamped["shadow_threshold"], _clamped["highlight_threshold"]
    detail_strength = _clamped["detail_strength"]
    n_segments, compactness = _clamped["n_segments"], _clamped["compactness"]
    crf_spatial, crf_color, crf_compat = _clamped["crf_spatial"], _clamped["crf_color"], _clamped["crf_compat"]
    mod = get_module(version)
    if mod is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown version: {version}. Valid: {sorted(VALID_VERSIONS)}"},
        )

    kwargs: dict = dict(
        image_bytes=image_bytes, plates=plates, dust=dust,
        use_edges=use_edges, edge_sigma=edge_sigma, locked_colors=locked,
    )
    # v20 is the only version; all requests go through the SAM path
    kwargs["shadow_threshold"] = shadow_threshold
    kwargs["highlight_threshold"] = highlight_threshold
    kwargs["median_size"] = median_size
    kwargs["chroma_boost"] = chroma_boost
    kwargs["upscale"] = upscale if UPSCALE_ENABLED else False
    kwargs["upscale_scale"] = upscale_scale

    if version in SAM_VERSIONS:
        plates = _clamp(plates, PLATES_MIN, SAM_PLATES_MAX)
        kwargs["plates"] = plates
        ok, msg = check_memory_for_sam(n_colors=plates)
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"error": msg, "code": "MEMORY_LOW", "retry_after_seconds": 30, "plates_completed": 0},
                headers={"Retry-After": "30"},
            )

    # Only SAM versions have progress_callback support in their modules
    is_sam = version in SAM_VERSIONS
    progress_events: list[dict] = []

    def on_progress(stage: str, pct: int):
        progress_events.append({"stage": stage, "pct": pct})

    if is_sam:
        kwargs["progress_callback"] = on_progress

    async def generate():
        loop = asyncio.get_event_loop()

        # SAM versions need the heavy semaphore; non-SAM versions run freely
        if is_sam:
            ctx = _heavy_semaphore
        else:
            import contextlib
            ctx = contextlib.nullcontext()

        try:
            async with ctx:
                future = loop.run_in_executor(
                    None, lambda: mod.build_preview_response(**kwargs)
                )

                sent = 0
                heartbeat_counter = 0
                elapsed = 0.0
                last_heartbeat_ts = time.time()
                plates_seen = 0
                total_plates = kwargs.get("plates", 3)
                while not future.done():
                    await asyncio.sleep(0.3)
                    elapsed += 0.3
                    if elapsed > SAM_TIMEOUT_SECONDS:
                        future.cancel()
                        _t_err = (
                            f'Processing timed out after {SAM_TIMEOUT_SECONDS}s.'
                            ' Try fewer plates or a smaller image.'
                        )
                        yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'error': _t_err})}\n\n"
                        _rlog_ps.set_error(_t_err, "TimeoutError")
                        _rlog_ps.finish(status=504)
                        return
                    # Flush any real progress events from SAM versions
                    while sent < len(progress_events):
                        evt = progress_events[sent]
                        pct = evt.get("pct", 0)
                        yield f"data: {json.dumps(evt)}\n\n"
                        sent += 1
                        # Per-plate SSE: emit plate_complete when a plate stage is seen
                        if "plate" in evt.get("stage", "").lower():
                            plates_seen += 1
                            _pc = {'stage': 'plate_complete', 'pct': pct,
                                   'plate_index': plates_seen, 'total_plates': total_plates}
                            yield f"data: {json.dumps(_pc)}\n\n"
                    # For non-SAM versions (no progress_callback), send a heartbeat
                    # every ~5s so Cloudflare's 100s tunnel timeout never fires.
                    heartbeat_counter += 1
                    if not is_sam and heartbeat_counter % 17 == 0:
                        estimated_pct = min(90, int((heartbeat_counter / 17) * 10))
                        yield f"data: {json.dumps({'stage': 'Separating colors', 'pct': estimated_pct})}\n\n"
                    # Universal heartbeat: every ~10 seconds regardless of SAM or non-SAM
                    now = time.time()
                    if now - last_heartbeat_ts >= 10.0:
                        yield f"data: {json.dumps({'stage': 'heartbeat', 'pct': -1, 'ts': now})}\n\n"
                        last_heartbeat_ts = now

                # Drain remaining real progress events
                while sent < len(progress_events):
                    evt = progress_events[sent]
                    pct = evt.get("pct", 0)
                    yield f"data: {json.dumps(evt)}\n\n"
                    sent += 1
                    if "plate" in evt.get("stage", "").lower():
                        plates_seen += 1
                        _pc2 = {'stage': 'plate_complete', 'pct': pct,
                                'plate_index': plates_seen, 'total_plates': total_plates}
                        yield f"data: {json.dumps(_pc2)}\n\n"

                composite_bytes, manifest = future.result()

            # Save result to temp file instead of base64-encoding inline
            # This prevents 30+MB SSE events that crash the container
            _cleanup_old_results()
            result_id = hashlib.sha256(composite_bytes[:4096] + str(time.time()).encode()).hexdigest()[:16]
            result_path = _RESULT_DIR / f"{result_id}.png"
            result_path.write_bytes(composite_bytes)
            del composite_bytes  # Free memory immediately

            pct = 100
            if manifest.get("partial_results"):
                _partial = {
                    'stage': 'partial_complete', 'pct': pct,
                    'plates_completed': len(manifest.get('plates', [])),
                    'error': 'memory_pressure', 'manifest': manifest, 'result_id': result_id,
                }
                yield f"data: {json.dumps(_partial)}\n\n"
                _rlog_ps.set_warning("partial_results: memory_pressure")
            else:
                _done = {'stage': 'complete', 'pct': 100,
                         'manifest': manifest, 'result_id': result_id}
                yield f"data: {json.dumps(_done)}\n\n"
            try:
                _m = manifest
                _rlog_ps.set_output(
                    plates_returned=len(_m.get("plates", [])),
                    w=_m.get("width"), h=_m.get("height"),
                )
            except Exception:
                pass
            _rlog_ps.finish(status=200)
        except (MemoryError, Exception) as e:
            err_str = str(e)
            if isinstance(e, MemoryError) or "CUDA out of memory" in err_str:
                _oom = {'stage': 'partial_complete', 'pct': 0,
                        'plates_completed': 0, 'error': 'memory_pressure'}
                yield f"data: {json.dumps(_oom)}\n\n"
                _rlog_ps.set_error("OOM / CUDA out of memory", type(e).__name__)
                _rlog_ps.finish(status=500)
                return
            _rlog_ps.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            _rlog_ps.finish(status=500)
            yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'error': f'{type(e).__name__}: {e}'})}\n\n"
        finally:
            _cleanup_gpu()

    return StreamingResponse(generate(), media_type="text/event-stream")


def _cleanup_old_results():
    """Remove result files older than _RESULT_MAX_AGE_S."""
    now = time.time()
    try:
        for f in _RESULT_DIR.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > _RESULT_MAX_AGE_S:
                f.unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/api/result/{result_id}")
async def get_result(result_id: str):
    """Serve a previously-saved preview result image."""
    if not _re.match(r'^[a-zA-Z0-9\-]+$', result_id):
        return JSONResponse(status_code=400, content={"error": "Invalid result ID"})

    path = _RESULT_DIR / f"{result_id}.png"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Result not found or expired"})

    return Response(
        content=path.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=1800"},
    )


@app.post("/api/separate")
async def separate_endpoint(
    request: Request,
    image: UploadFile = File(...),
    plates: int = Form(3),
    dust: int = Form(20),
    use_edges: bool = Form(True),
    edge_sigma: float = Form(1.5),
    locked_colors: str | None = Form(None),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    median_size: int = Form(5),
    chroma_boost: float = Form(1.3),
    shadow_threshold: int = Form(8),
    highlight_threshold: int = Form(95),
    n_segments: int = Form(3000),
    compactness: int = Form(15),
    crf_spatial: int = Form(3),
    crf_color: int = Form(13),
    crf_compat: int = Form(10),
    sigma_s: float = Form(100),
    sigma_r: float = Form(0.5),
    meanshift_sp: int = Form(15),
    meanshift_sr: int = Form(30),
    detail_strength: float = Form(0.5),
):
    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        return err
    image_bytes = strip_exif(image_bytes)
    # ── Analytics setup ──────────────────────────────────────────────────
    _rlog_sep = RequestLog("/api/separate")
    _rlog_sep.set_client(
        request.headers.get("user-agent", ""),
        request.client.host if request.client else "",
    )
    try:
        _img_probe_sep = Image.open(io.BytesIO(image_bytes))
        _rlog_sep.set_input(
            w=_img_probe_sep.size[0], h=_img_probe_sep.size[1],
            kb=len(image_bytes) / 1024, fmt=_img_probe_sep.format or "PNG",
        )
    except Exception:
        pass
    _rlog_sep.set_image_hash(hashlib.sha256(image_bytes).hexdigest()[:16])
    _rlog_sep.set_params(plates=plates, dust=dust, version=version,
                         upscale=upscale, upscale_scale=upscale_scale)
    # ─────────────────────────────────────────────────────────────────────
    locked = parse_locked_colors(locked_colors)
    plates = _clamp(plates, PLATES_MIN, PLATES_MAX)
    dust = _clamp(dust, DUST_MIN, DUST_MAX)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    _clamped = _clamp_float_params(dict(
        edge_sigma=edge_sigma, sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        chroma_boost=chroma_boost, median_size=median_size,
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        detail_strength=detail_strength, n_segments=n_segments, compactness=compactness,
        crf_spatial=crf_spatial, crf_color=crf_color, crf_compat=crf_compat,
    ))
    edge_sigma, sigma_s, sigma_r = _clamped["edge_sigma"], _clamped["sigma_s"], _clamped["sigma_r"]
    meanshift_sp, meanshift_sr = _clamped["meanshift_sp"], _clamped["meanshift_sr"]
    chroma_boost, median_size = _clamped["chroma_boost"], _clamped["median_size"]
    shadow_threshold, highlight_threshold = _clamped["shadow_threshold"], _clamped["highlight_threshold"]
    detail_strength = _clamped["detail_strength"]
    n_segments, compactness = _clamped["n_segments"], _clamped["compactness"]
    crf_spatial, crf_color, crf_compat = _clamped["crf_spatial"], _clamped["crf_color"], _clamped["crf_compat"]
    mod = get_module(version)
    if mod is None:
        _rlog_sep.set_error(f"Unknown version: {version}", "ValueError")
        _rlog_sep.finish(status=400)
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown version: {version}. Valid: {sorted(VALID_VERSIONS)}"},
        )

    kwargs: dict = dict(
        image_bytes=image_bytes, plates=plates, dust=dust,
        use_edges=use_edges, edge_sigma=edge_sigma, locked_colors=locked,
    )
    # v20 is the only version; all requests go through the SAM path
    kwargs["shadow_threshold"] = shadow_threshold
    kwargs["highlight_threshold"] = highlight_threshold
    kwargs["median_size"] = median_size
    kwargs["chroma_boost"] = chroma_boost
    kwargs["upscale"] = upscale if UPSCALE_ENABLED else False
    kwargs["upscale_scale"] = upscale_scale

    if version in SAM_VERSIONS:
        plates = _clamp(plates, PLATES_MIN, SAM_PLATES_MAX)
        kwargs["plates"] = plates
        ok, msg = check_memory_for_sam(n_colors=plates)
        if not ok:
            _rlog_sep.set_error(msg, "MemoryError")
            _rlog_sep.finish(status=503)
            return JSONResponse(
                status_code=503,
                content={"error": msg, "code": "MEMORY_LOW", "retry_after_seconds": 30, "plates_completed": 0},
                headers={"Retry-After": "30"},
            )
        try:
            async with _heavy_semaphore:
                with _rlog_sep.stage("separation"):
                    zip_bytes = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: mod.build_zip_response(**kwargs)
                        ),
                        timeout=SAM_TIMEOUT_SECONDS,
                    )
        except asyncio.TimeoutError:
            _cleanup_gpu()
            _rlog_sep.set_error(f"Timed out after {SAM_TIMEOUT_SECONDS}s", "TimeoutError")
            _rlog_sep.finish(status=504)
            return JSONResponse(status_code=504, content={
                "error": f"Processing timed out after {SAM_TIMEOUT_SECONDS}s. Try fewer plates or a smaller image.",
                "code": "TIMEOUT",
            })
        except Exception as e:
            _cleanup_gpu()
            _rlog_sep.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            _rlog_sep.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Separation failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })
        finally:
            _cleanup_gpu()
    else:
        try:
            with _rlog_sep.stage("separation"):
                zip_bytes = mod.build_zip_response(**kwargs)
        except Exception as e:
            _rlog_sep.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            _rlog_sep.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Separation failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })

    _rlog_sep.set_output(plates_returned=plates)
    _rlog_sep.finish(status=200)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=woodblock-plates.zip"},
    )


@app.post("/api/upscale")
async def upscale_endpoint(
    image: UploadFile = File(...),
    upscale_scale: int = Form(2),
):
    """Pre-upscale an image and cache it for later processing."""
    rlog = RequestLog("/api/upscale")
    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        rlog.set_error("validation failed", "ValidationError")
        rlog.finish(status=err.status_code)
        return err
    image_bytes = strip_exif(image_bytes)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    try:
        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        rlog.set_input(w=orig_w, h=orig_h, kb=len(image_bytes) / 1024, fmt=img.format or "PNG")
    except Exception:
        pass
    rlog.set_params(upscale_scale=upscale_scale)
    with rlog.stage("upscale"):
        if v20 is None:
            return JSONResponse(status_code=503, content={"error": "v20 module not available"})
        img_hash, cached, success = v20.upscale_and_cache(image_bytes, scale=upscale_scale)
    rlog.set_cache_hit(hit=bool(cached))
    rlog.finish(status=200)
    return Response(
        content=json.dumps({"hash": img_hash, "cached": cached, "upscaled": success}),
        media_type="application/json",
    )


@app.post("/api/merge")
async def merge_endpoint(
    image: UploadFile = File(...),
    merge_pairs: str = Form(...),
    plates: int = Form(3),
    dust: int = Form(20),
    locked_colors: str | None = Form(None),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    chroma_boost: float = Form(1.3),
    sigma_s: float = Form(100),
    sigma_r: float = Form(0.5),
    meanshift_sp: int = Form(15),
    meanshift_sr: int = Form(30),
    img_hash: str | None = Form(None),
):
    """Run separation then merge specified plate pairs."""
    rlog = RequestLog("/api/merge")
    image_bytes = await image.read()
    rlog.set_input(w=0, h=0, kb=len(image_bytes) / 1024)
    err = await validate_upload(image_bytes)
    if err is not None:
        rlog.set_error("validation failed", "ValidationError")
        rlog.finish(status=400)
        return err
    image_bytes = strip_exif(image_bytes)
    locked = parse_locked_colors(locked_colors)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    try:
        pairs = json.loads(merge_pairs)
    except (json.JSONDecodeError, TypeError):
        rlog.set_error("Invalid merge_pairs JSON", "JSONDecodeError")
        rlog.finish(status=400)
        return JSONResponse(status_code=400, content={"error": "Invalid merge_pairs JSON."})

    rlog.set_params(plates=plates, dust=dust, version=version, upscale=upscale,
                    upscale_scale=upscale_scale, merge_pairs=pairs)

    merge_mod = VERSION_MAP.get(version)
    if merge_mod is None:
        rlog.set_error(f"Unknown version: {version}", "ValueError")
        rlog.finish(status=400)
        return JSONResponse(status_code=400, content={"error": f"Unknown version: {version}"})

    if version == "v20" and not UPSCALE_ENABLED:
        upscale = False  # Disabled in gpu_config

    merge_kwargs = dict(
        image_bytes=image_bytes,
        merge_pairs=pairs,
        plates=plates,
        dust=dust,
        locked_colors=locked,
        chroma_boost=chroma_boost,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        meanshift_sp=meanshift_sp,
        meanshift_sr=meanshift_sr,
        upscale=upscale,
        img_hash=img_hash,
    )
    if version in SAM_VERSIONS:
        merge_kwargs["upscale_scale"] = upscale_scale

    if version in SAM_VERSIONS:
        merge_kwargs["plates"] = _clamp(plates, PLATES_MIN, SAM_PLATES_MAX)
        ok, msg = check_memory_for_sam(n_colors=_clamp(plates, PLATES_MIN, SAM_PLATES_MAX))
        if not ok:
            rlog.set_error(msg, "MemoryError")
            rlog.finish(status=503)
            return JSONResponse(
                status_code=503,
                content={"error": msg, "code": "MEMORY_LOW", "retry_after_seconds": 30, "plates_completed": 0},
                headers={"Retry-After": "30"},
            )
        try:
            async with _heavy_semaphore:
                with rlog.stage("merge"):
                    composite_bytes, manifest = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: merge_mod.build_merge_response(**merge_kwargs)
                        ),
                        timeout=SAM_TIMEOUT_SECONDS,
                    )
        except asyncio.TimeoutError:
            _cleanup_gpu()
            rlog.set_error(f"Merge timed out after {SAM_TIMEOUT_SECONDS}s", "TimeoutError")
            rlog.finish(status=504)
            return JSONResponse(status_code=504, content={
                "error": f"Merge timed out after {SAM_TIMEOUT_SECONDS}s. Try fewer plates or a smaller image.",
                "code": "TIMEOUT",
            })
        except Exception as e:
            _cleanup_gpu()
            rlog.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            rlog.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Merge failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })
        finally:
            _cleanup_gpu()
    else:
        try:
            with rlog.stage("merge"):
                composite_bytes, manifest = merge_mod.build_merge_response(**merge_kwargs)
        except Exception as e:
            rlog.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            rlog.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Merge failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })

    rlog.set_output(plates_returned=len(manifest.get("plates", [])))
    rlog.finish(status=200)
    return Response(
        content=composite_bytes,
        media_type="image/png",
        headers={"X-Manifest": json.dumps(manifest)},
    )


@app.post("/api/plates")
async def plates_endpoint(
    image: UploadFile = File(...),
    plates: int = Form(3),
    dust: int = Form(20),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    chroma_boost: float = Form(1.3),
    sigma_s: float = Form(100),
    sigma_r: float = Form(0.5),
    meanshift_sp: int = Form(15),
    meanshift_sr: int = Form(30),
    locked_colors: str | None = Form(None),
):
    """Return JSON with base64-encoded plate thumbnail images (400px max)."""
    rlog = RequestLog("/api/plates")
    image_bytes = await image.read()
    rlog.set_input(w=0, h=0, kb=len(image_bytes) / 1024)
    err = await validate_upload(image_bytes)
    if err is not None:
        rlog.set_error("validation failed", "ValidationError")
        rlog.finish(status=400)
        return err
    image_bytes = strip_exif(image_bytes)
    locked = parse_locked_colors(locked_colors)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.load()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    max_dim = 800
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    plates = _clamp(int(plates), PLATES_MIN, PLATES_MAX)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    rlog.set_input(w=img.size[0], h=img.size[1], kb=len(image_bytes) / 1024)
    rlog.set_params(plates=plates, dust=dust, version=version, upscale_scale=upscale_scale)
    arr = np.array(img)
    mod = get_module(version)
    kwargs: dict = dict(
        n_plates=plates, dust_threshold=dust,
        locked_colors=locked, return_data=True,
    )
    # v20 (SAM) params
    kwargs["upscale"] = False  # plates endpoint always uses small thumbnails
    kwargs["chroma_boost"] = chroma_boost
    kwargs.update(use_edges=True, edge_sigma=1.5, shadow_threshold=8, highlight_threshold=95, median_size=3)
    kwargs["n_plates"] = _clamp(plates, PLATES_MIN, SAM_PLATES_MAX)
    kwargs["upscale_scale"] = upscale_scale

    if version in SAM_VERSIONS:
        ok, msg = check_memory_for_sam(n_colors=kwargs.get("n_plates", plates))
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"error": msg, "code": "MEMORY_LOW", "retry_after_seconds": 30, "plates_completed": 0},
                headers={"Retry-After": "30"},
            )
        try:
            async with _heavy_semaphore:
                with rlog.stage("separation"):
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: mod.separate(arr, **kwargs)
                        ),
                        timeout=SAM_TIMEOUT_SECONDS,
                    )
        except asyncio.TimeoutError:
            _cleanup_gpu()
            rlog.set_error(f"Timed out after {SAM_TIMEOUT_SECONDS}s", "TimeoutError")
            rlog.finish(status=504)
            return JSONResponse(status_code=504, content={
                "error": f"Processing timed out after {SAM_TIMEOUT_SECONDS}s.",
                "code": "TIMEOUT",
            })
        except Exception as e:
            _cleanup_gpu()
            rlog.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            rlog.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Separation failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })
        finally:
            _cleanup_gpu()
    else:
        try:
            with rlog.stage("separation"):
                result = mod.separate(arr, **kwargs)
        except Exception as e:
            rlog.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            rlog.finish(status=500)
            return JSONResponse(status_code=500, content={
                "error": f"Separation failed: {type(e).__name__}: {e}",
                "code": "PROCESSING_ERROR",
            })

    plate_images = []
    for plate_info in result["manifest"]["plates"]:
        name = plate_info["name"]
        plate_data = result["plates"][name]
        buf = io.BytesIO()
        plate_data["image"].save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        plate_images.append({
            "name": name,
            "color": plate_info["color"],
            "coverage": plate_info.get("coverage_pct", 0),
            "image": f"data:image/png;base64,{b64}",
        })

    rlog.set_output(plates_returned=len(plate_images))
    rlog.finish(status=200)
    return Response(
        content=json.dumps({"plates": plate_images}),
        media_type="application/json",
    )


@app.post("/api/plates-stream")
async def plates_stream_endpoint(
    image: UploadFile = File(...),
    plates: int = Form(3),
    dust: int = Form(20),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    chroma_boost: float = Form(1.3),
    sigma_s: float = Form(100),
    sigma_r: float = Form(0.5),
    meanshift_sp: int = Form(15),
    meanshift_sr: int = Form(30),
    locked_colors: str | None = Form(None),
):
    """Stream individual plate thumbnails via SSE as they become available.

    Sends one SSE event per plate as soon as computation is complete, so the
    frontend can show plates progressively instead of waiting for all of them.
    Also sends {"type":"count","total":N} as the first event so the UI can
    show "Loading plate X of N".
    """
    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        return err
    image_bytes = strip_exif(image_bytes)
    locked = parse_locked_colors(locked_colors)

    # ── Analytics setup ──────────────────────────────────────────────────
    _rlog_plts = RequestLog("/api/plates-stream")
    try:
        _img_probe_plts = Image.open(io.BytesIO(image_bytes))
        _rlog_plts.set_input(
            w=_img_probe_plts.size[0], h=_img_probe_plts.size[1],
            kb=len(image_bytes) / 1024, fmt=_img_probe_plts.format or "PNG",
        )
    except Exception:
        pass
    _rlog_plts.set_params(plates=plates, dust=dust, version=version,
                          upscale=upscale, upscale_scale=upscale_scale)
    # ─────────────────────────────────────────────────────────────────────

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.load()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    max_dim = 800
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    plates = _clamp(int(plates), PLATES_MIN, PLATES_MAX)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    arr = np.array(img)
    mod = get_module(version)
    if mod is None:
        return JSONResponse(status_code=400, content={"error": f"Unknown version: {version}"})

    kwargs: dict = dict(
        n_plates=plates, dust_threshold=dust,
        locked_colors=locked, return_data=True,
    )
    # v20 (SAM) params
    kwargs["upscale"] = False
    kwargs["chroma_boost"] = chroma_boost
    kwargs.update(use_edges=True, edge_sigma=1.5, shadow_threshold=8, highlight_threshold=95, median_size=3)
    kwargs["n_plates"] = _clamp(plates, PLATES_MIN, SAM_PLATES_MAX)
    kwargs["upscale_scale"] = upscale_scale

    if version in SAM_VERSIONS:
        ok, msg = check_memory_for_sam(n_colors=kwargs.get("n_plates", plates))
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"error": msg, "code": "MEMORY_LOW", "retry_after_seconds": 30, "plates_completed": 0},
                headers={"Retry-After": "30"},
            )

    async def generate():
        loop = asyncio.get_event_loop()

        try:
            if version in SAM_VERSIONS:
                async with _heavy_semaphore:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: mod.separate(arr, **kwargs)),
                        timeout=SAM_TIMEOUT_SECONDS,
                    )
            else:
                result = await loop.run_in_executor(None, lambda: mod.separate(arr, **kwargs))

            plate_infos = result["manifest"]["plates"]
            total = len(plate_infos)

            # First event: announce total so frontend can show "Loading plate X of N"
            yield f"data: {json.dumps({'type': 'count', 'total': total})}\n\n"

            for idx, plate_info in enumerate(plate_infos):
                name = plate_info["name"]
                plate_data = result["plates"][name]
                buf = io.BytesIO()
                plate_data["image"].save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                evt = {
                    "type": "plate",
                    "index": plate_info.get("index", idx),  # K-means centroid index, not enumerate position
                    "total": total,
                    "name": name,
                    "color": plate_info["color"],
                    "coverage": plate_info.get("coverage_pct", 0),
                    "image": f"data:image/png;base64,{b64}",
                }
                # Potrace SVG for stream fallback (same quality as /api/plates-svg)
                try:
                    from svg_generator import mask_to_svg
                    if "mask" in plate_data:
                        evt["svg"] = mask_to_svg(plate_data["mask"], arr.shape[1], arr.shape[0])
                except Exception:
                    pass  # SVG generation is best-effort in stream
                yield f"data: {json.dumps(evt)}\n\n"
                # Yield control briefly so the event loop can flush the SSE chunk
                await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'done', 'total': total})}\n\n"
            _rlog_plts.set_output(plates_returned=total)
            _rlog_plts.finish(status=200)
        except asyncio.TimeoutError:
            _to_msg = f'Processing timed out after {SAM_TIMEOUT_SECONDS}s.'
            yield f"data: {json.dumps({'type': 'error', 'error': _to_msg})}\n\n"
            _rlog_plts.set_error(_to_msg, "TimeoutError")
            _rlog_plts.finish(status=504)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': f'{type(e).__name__}: {e}'})}\n\n"
            _rlog_plts.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            _rlog_plts.finish(status=500)
        finally:
            _cleanup_gpu()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/plates-svg")
async def plates_svg_endpoint(
    image: UploadFile = File(...),
    plates: int = Form(3),
    dust: int = Form(20),
    version: str = Form("v20"),
    upscale: bool = Form(True),
    upscale_scale: int = Form(2),
    chroma_boost: float = Form(1.3),
    shadow_threshold: int = Form(8),
    highlight_threshold: int = Form(95),
    median_size: int = Form(3),
    locked_colors: str | None = Form(None),
):
    """Enqueue SVG generation and return job_id for polling.

    Always async — returns {job_id, status} immediately. Client polls
    GET /api/job/{job_id} for the result. Eliminates Cloudflare 524 timeout.

    If SVG cache is hot, returns result inline (no job needed).
    """
    from job_queue import create_job, update_job, JobStatus

    if v20 is None:
        return JSONResponse(status_code=400, content={"error": "v20 not available"})

    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        return err
    image_bytes = strip_exif(image_bytes)

    locked = parse_locked_colors(locked_colors)
    plates = _clamp(int(plates), PLATES_MIN, SAM_PLATES_MAX)
    dust = _clamp(int(dust), DUST_MIN, DUST_MAX)
    upscale_scale = upscale_scale if upscale_scale in (2, 4) else 2
    if not UPSCALE_ENABLED:
        upscale = False

    raw_hash = hashlib.sha256(image_bytes).hexdigest()
    cache_key = v20._make_cache_key(raw_hash, plates, dust)

    # Fast path: return cached SVG immediately (no job needed)
    if cache_key in v20._svg_cache:
        rlog = RequestLog("/api/plates-svg", request_id=raw_hash[:16])
        rlog.set_cache_hit(hit=True, svg=True)
        rlog.finish(status=200)
        return Response(
            content=v20._svg_cache[cache_key],
            media_type="application/json",
        )

    # Slow path: enqueue background job, return job_id
    job_id = create_job()

    async def _run_job():
        rlog = RequestLog("/api/plates-svg", request_id=raw_hash[:16])
        try:
            _img_probe = Image.open(io.BytesIO(image_bytes))
            rlog.set_input(
                w=_img_probe.size[0], h=_img_probe.size[1],
                kb=len(image_bytes) / 1024, fmt=_img_probe.format or "PNG",
            )
        except Exception:
            pass
        rlog.set_params(
            plates=plates, dust=dust, version=version,
            upscale=upscale, upscale_scale=upscale_scale,
        )

        loop = asyncio.get_event_loop()
        try:
            update_job(job_id, JobStatus.RUNNING, progress="separation")

            if cache_key not in v20._separation_cache:
                # Cache miss: run at 800px, scale masks up
                _FALLBACK_DIM = 800

                def _run_fallback():
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    orig_w, orig_h = img.size
                    target_w = orig_w * upscale_scale if upscale else orig_w
                    target_h = orig_h * upscale_scale if upscale else orig_h
                    if max(img.size) > _FALLBACK_DIM:
                        ratio = _FALLBACK_DIM / max(img.size)
                        img = img.resize(
                            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                            Image.LANCZOS,
                        )
                    arr = np.array(img)
                    thumb_w, thumb_h = img.size
                    result = v11.separate(
                        arr, n_plates=plates, dust_threshold=dust,
                        locked_colors=locked, return_data=True,
                        chroma_boost=chroma_boost, upscale=False,
                    )
                    manifest = result.get("manifest", {})
                    plates_data = result.get("plates", {})
                    manifest["width"] = target_w
                    manifest["height"] = target_h
                    for plate_info in manifest.get("plates", []):
                        name = plate_info["name"]
                        pd = plates_data.get(name, {})
                        if "mask" in pd and (thumb_w != target_w or thumb_h != target_h):
                            mask_img = Image.fromarray(pd["mask"].astype(np.uint8) * 255, mode="L")
                            mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
                            pd["mask"] = np.array(mask_img) > 127
                            pd["binary"] = np.where(pd["mask"], 0, 255).astype(np.uint8)
                            pd["image"] = Image.fromarray(pd["binary"])
                    v20._separation_cache[cache_key] = {
                        "manifest": manifest,
                        "plates": plates_data,
                    }
                    while len(v20._separation_cache) > v20._CACHE_MAX_ENTRIES:
                        oldest = next(iter(v20._separation_cache))
                        del v20._separation_cache[oldest]

                rlog.set_fallback(used=True)
                with rlog.stage("separation"):
                    await loop.run_in_executor(None, _run_fallback)
            else:
                rlog.set_cache_hit(hit=True)

            cached = v20._separation_cache.get(cache_key)
            if cached is None:
                raise RuntimeError("Cache write failed unexpectedly.")

            manifest = cached["manifest"]
            h, w = manifest["height"], manifest["width"]

            update_job(job_id, JobStatus.RUNNING, progress="potrace")

            def _build_svgs():
                import base64 as b64mod
                svgs = []
                for plate_info in manifest["plates"]:
                    name = plate_info["name"]
                    plate_data = cached["plates"].get(name, {})
                    mask = plate_data.get("mask")
                    svg = v20.mask_to_svg_string(mask, w, h) if mask is not None else ""
                    png_b64 = ""
                    plate_image = plate_data.get("image")
                    if plate_image is not None:
                        buf = io.BytesIO()
                        plate_image.save(buf, format="PNG", compress_level=1)
                        png_b64 = b64mod.b64encode(buf.getvalue()).decode("ascii")
                    svgs.append({
                        "name": name,
                        "color": plate_info["color"],
                        "svg": svg,
                        "png_b64": png_b64,
                    })
                return svgs

            with rlog.stage("potrace"):
                svgs = await loop.run_in_executor(None, _build_svgs)

            response_json = json.dumps(svgs)
            v20._svg_cache[cache_key] = response_json
            while len(v20._svg_cache) > v20._SVG_CACHE_MAX_ENTRIES:
                oldest = next(iter(v20._svg_cache))
                del v20._svg_cache[oldest]

            try:
                svg_sizes = [len(s.get("svg", "")) / 1024 for s in svgs]
                png_sizes = [len(s.get("png_b64", "")) * 3 / 4 / 1024 for s in svgs]
                rlog.set_output(w=w, h=h, plates_returned=len(svgs),
                                svg_sizes_kb=svg_sizes, png_sizes_kb=png_sizes)
            except Exception:
                pass
            rlog.finish(status=200)

            update_job(job_id, JobStatus.DONE, result=response_json.encode())

        except Exception as e:
            rlog.set_error(f"{type(e).__name__}: {e}", type(e).__name__)
            rlog.finish(status=500)
            update_job(job_id, JobStatus.ERROR, error=f"{type(e).__name__}: {e}")

    asyncio.create_task(_run_job())
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.post("/api/auto-optimize")
async def auto_optimize_endpoint(
    image: UploadFile = File(...),
    plates: int = Form(8),
):
    """Trigger auto-optimization via OpenClaw. Returns job ID for polling."""
    image_bytes = await image.read()
    err = await validate_upload(image_bytes)
    if err is not None:
        return err
    image_bytes = strip_exif(image_bytes)
    status = auto_optimize.trigger_optimization(image_bytes, initial_plates=plates)
    return Response(
        content=json.dumps(status),
        media_type="application/json",
    )
# Allowlist pattern for job IDs: exactly 12 lowercase hex chars (matches what
# auto_optimize.trigger_optimization() generates via md5[:12]).
_JOB_ID_RE = _re.compile(r"^[a-f0-9]{12}$")


@app.get("/api/auto-optimize/{job_id}")
async def auto_optimize_status(job_id: str):
    """Poll auto-optimization status."""
    if not _JOB_ID_RE.match(job_id):
        return JSONResponse(status_code=400, content={"error": "Invalid job ID."})
    status = auto_optimize.get_status(job_id)
    return Response(
        content=json.dumps(status),
        media_type="application/json",
    )
