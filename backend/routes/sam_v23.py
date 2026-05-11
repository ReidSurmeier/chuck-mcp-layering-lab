"""v23-MCP SAM HTTP endpoint — ``POST /api/sam``.

Region-prior gateway for the v23 inverse-stack solver. Wraps the cached
v20 ``get_sam_model()`` singleton; emits SAM2.1 auto-mask proposals as
base64-encoded PNG masks + bbox + ``mean_oklab`` per region. Cached by
``(image_sha256, min_region_area_px)``, LRU-bounded at 16 entries.

Authority: ``/tmp/research-v23-mcp-interfaces.md`` §1.1.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import logging
import time
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

log = logging.getLogger("woodblock.routes.sam_v23")
router = APIRouter(tags=["sam-v23"])

# Variable (not Final) so tests can monkeypatch a tiny value.
SAM_TIMEOUT_SECONDS: float = 600.0
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # mirrors backend/main.py
_MAX_REGIONS_CEILING: int = 256

# In-process LRU response cache: (sha256, min_area_px) -> _CachedEntry.
_SAM_CACHE: dict[tuple[str, int], "_CachedEntry"] = {}
_SAM_CACHE_ORDER: list[tuple[str, int]] = []
_SAM_CACHE_MAX: int = 16


class SAMRegion(BaseModel):
    region_id: str
    bbox: tuple[int, int, int, int]
    area_px: int = Field(ge=1)
    mask_png_b64: str
    mean_oklab: tuple[float, float, float]


class S2Result(BaseModel):
    image_sha256: str
    regions: list[SAMRegion]
    sam_wall_s: float
    cached: bool = False
    model_version: str = "sam2.1_b"


class _CachedEntry(BaseModel):
    image_sha256: str
    regions: list[SAMRegion]
    sam_wall_s: float
    model_version: str


def _load_sam_model() -> Any:
    """Indirection over ``separate_v20.get_sam_model`` for monkeypatching."""
    from separate_v20 import get_sam_model  # type: ignore[import-not-found]
    return get_sam_model()


def _rgb_to_oklab(arr: np.ndarray) -> np.ndarray:
    """Vectorised sRGB-uint8 → Oklab. Returns ``(..., 3) float64``."""
    flat = arr.astype(np.float64) / 255.0
    lin = np.where(flat <= 0.04045, flat / 12.92, ((flat + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_c, m_c, s_c = np.cbrt(l_), np.cbrt(m_), np.cbrt(s_)
    L = 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c
    a = 1.9779984951 * l_c - 2.4285922050 * m_c + 0.4505937099 * s_c
    bb = 0.0259040371 * l_c + 0.7827717662 * m_c - 0.8086757660 * s_c
    return np.stack([L, a, bb], axis=-1)


def _mean_oklab(img_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    if not np.any(mask):
        return (0.0, 0.0, 0.0)
    m = np.mean(_rgb_to_oklab(img_rgb[mask]), axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def _encode_mask_png_b64(mask: np.ndarray) -> str:
    """Encode a boolean mask as a 1-channel PNG (mode 'L', 0/255), base64."""
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` AABB of the ``True`` region. ``(0,0,0,0)`` if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize. SAM2.1 may return lower-res masks than input."""
    if mask.shape == target_hw:
        return mask.astype(bool)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8) > 127


def _cache_put(key: tuple[str, int], entry: _CachedEntry) -> None:
    if key in _SAM_CACHE:
        _SAM_CACHE_ORDER.remove(key)
    _SAM_CACHE[key] = entry
    _SAM_CACHE_ORDER.append(key)
    while len(_SAM_CACHE_ORDER) > _SAM_CACHE_MAX:
        oldest = _SAM_CACHE_ORDER.pop(0)
        _SAM_CACHE.pop(oldest, None)


def _build_regions(
    masks: np.ndarray,
    img_rgb: np.ndarray,
    min_region_area_px: int,
    max_regions: int,
) -> list[SAMRegion]:
    """Filter by min_area, sort by area desc, cap, encode."""
    h, w = img_rgb.shape[:2]
    survivors: list[tuple[int, np.ndarray]] = []
    for k in range(masks.shape[0]):
        m = _resize_mask(masks[k], (h, w))
        area = int(np.count_nonzero(m))
        if area < min_region_area_px:
            continue
        survivors.append((area, m))
    survivors.sort(key=lambda t: -t[0])
    survivors = survivors[: max(1, min(max_regions, _MAX_REGIONS_CEILING))]

    regions: list[SAMRegion] = []
    for idx, (area, m) in enumerate(survivors):
        regions.append(SAMRegion(
            region_id=f"rgn_{idx:03d}",
            bbox=_bbox_of_mask(m),
            area_px=area,
            mask_png_b64=_encode_mask_png_b64(m),
            mean_oklab=_mean_oklab(img_rgb, m),
        ))
    return regions


def _coerce_masks(raw: Any) -> np.ndarray | None:
    """Pull ``(N,H,W) bool`` numpy array from ultralytics-style results."""
    if raw is None:
        return None
    data = getattr(raw, "data", raw)
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return None
    return arr.astype(bool)


def _run_sam(img_rgb: np.ndarray) -> np.ndarray | None:
    model = _load_sam_model()
    results = model.predict(img_rgb, verbose=False)
    if not results:
        return None
    return _coerce_masks(results[0].masks)


@router.post("/sam", response_model=S2Result)
async def sam_endpoint(
    file: Annotated[UploadFile, File(description="Source image (PNG/JPEG)")],
    min_region_area_px: Annotated[int, Form(ge=1, le=10_000_000)] = 500,
    max_regions: Annotated[int, Form(ge=1, le=256)] = 64,
    image_sha256: Annotated[str | None, Form()] = None,
) -> Any:
    """SAM2.1 auto-mask generation on the uploaded image.

    Cached by ``(sha256, min_region_area_px)``. The optional ``image_sha256``
    form field lets the client assert what sha it expects — mismatch → 400.
    """
    raw = await file.read()
    if len(raw) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty upload."})
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Max {MAX_UPLOAD_BYTES} bytes."},
        )

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.load()
        img_rgb = np.asarray(img)
    except Exception as exc:  # noqa: BLE001
        log.warning("invalid upload: %s", exc)
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    sha = hashlib.sha256(raw).hexdigest()
    if image_sha256 is not None and image_sha256 != sha:
        return JSONResponse(
            status_code=400,
            content={"error": "image_sha256 mismatch", "expected": image_sha256, "got": sha},
        )

    cache_key = (sha, int(min_region_area_px))
    if cache_key in _SAM_CACHE:
        cached = _SAM_CACHE[cache_key]
        return S2Result(
            image_sha256=cached.image_sha256,
            regions=cached.regions,
            sam_wall_s=cached.sam_wall_s,
            cached=True,
            model_version=cached.model_version,
        ).model_dump()

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_sam, img_rgb)
        try:
            masks = future.result(timeout=SAM_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            log.warning("SAM_TIMEOUT after %.2fs", SAM_TIMEOUT_SECONDS)
            future.cancel()
            return JSONResponse(
                status_code=504,
                content={
                    "error": f"SAM exceeded {SAM_TIMEOUT_SECONDS}s.",
                    "code": "SAM_TIMEOUT",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("SAM inference failed")
            return JSONResponse(
                status_code=500,
                content={"error": f"SAM failed: {type(exc).__name__}: {exc}", "code": "SAM_ERROR"},
            )

    sam_wall_s = time.perf_counter() - t0
    if masks is None or masks.shape[0] == 0:
        regions: list[SAMRegion] = []
    else:
        regions = _build_regions(masks, img_rgb, int(min_region_area_px), int(max_regions))

    entry = _CachedEntry(
        image_sha256=sha,
        regions=regions,
        sam_wall_s=sam_wall_s,
        model_version="sam2.1_b",
    )
    _cache_put(cache_key, entry)
    return S2Result(
        image_sha256=sha,
        regions=regions,
        sam_wall_s=sam_wall_s,
        cached=False,
        model_version="sam2.1_b",
    ).model_dump()
