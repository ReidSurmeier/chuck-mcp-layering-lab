"""FastAPI route for v21_mokuhanga separation.

Thin shim over `services.separation_v21_mokuhanga.separate_mokuhanga()`.
Returns the mokuhanga ZIP bytes with reconstruction metadata in custom headers.
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from services.separation_v21_mokuhanga import (  # type: ignore[import-not-found]
    V21Params,
    V21Result,
    separate_mokuhanga,
)

log = logging.getLogger("woodblock.routes.mokuhanga")

router = APIRouter(prefix="/api", tags=["mokuhanga"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB (matches CF Free-plan cap)


def _verify_api_key(request: Request) -> None:
    """X-API-Key gate when BACKEND_API_KEY env is set."""
    expected = os.environ.get("BACKEND_API_KEY", "")
    if not expected:
        return
    got = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(got, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@router.post("/separate/mokuhanga")
async def separate_mokuhanga_endpoint(
    request: Request,
    image: Annotated[UploadFile, File(description="Source image (PNG / JPEG)")],
    palette_size: Annotated[int, Form(ge=2, le=24)] = 13,
    threshold_tau: Annotated[float, Form(ge=0.0, le=1.0)] = 0.10,
    bleed_tolerance_px: Annotated[int, Form(ge=0, le=64)] = 8,
    color_grouping_weight: Annotated[float, Form(ge=0.0, le=1.0)] = 0.6,
    sliver_threshold_px: Annotated[int | None, Form()] = None,
    print_order_mode: Annotated[str, Form()] = "light_to_dark",
    substrate_r: Annotated[int, Form(ge=0, le=255)] = 255,
    substrate_g: Annotated[int, Form(ge=0, le=255)] = 255,
    substrate_b: Annotated[int, Form(ge=0, le=255)] = 255,
) -> Response:
    """Run the v21_mokuhanga decomposition pipeline.

    Form fields map to V21Params. Returns the mokuhanga ZIP bytes; reconstruction
    metrics surface as response headers:
      X-Mokuhanga-DE-Mean: reconstruction ΔE2000 mean
      X-Mokuhanga-DE-P95:  ΔE2000 p95
      X-Mokuhanga-Block-Count: physical block count after DSATUR
      X-Mokuhanga-Pigment-Count: pigments retained
      X-Mokuhanga-Warnings: warnings joined by " | "
    """
    _verify_api_key(request)

    if print_order_mode not in ("light_to_dark", "dark_to_light"):
        raise HTTPException(
            status_code=400,
            detail=(
                "print_order_mode must be 'light_to_dark' or 'dark_to_light', "
                f"got {print_order_mode!r}"
            ),
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(data)} bytes, max {MAX_UPLOAD_BYTES}).",
        )

    params = V21Params(
        palette_size=palette_size,
        threshold_tau=threshold_tau,
        bleed_tolerance_px=bleed_tolerance_px,
        color_grouping_weight=color_grouping_weight,
        sliver_threshold_px=sliver_threshold_px,
        print_order_mode=print_order_mode,  # type: ignore[arg-type]
        substrate_rgb=(substrate_r, substrate_g, substrate_b),
    )

    log.info(
        "v21_mokuhanga request: size=%d bytes, params=%s, content_type=%s",
        len(data),
        params,
        image.content_type,
    )

    try:
        result: V21Result = separate_mokuhanga(data, params=params)
    except Exception as exc:  # noqa: BLE001 — surface root cause to client
        log.exception("v21_mokuhanga failed")
        raise HTTPException(
            status_code=500,
            detail=f"separate_mokuhanga error: {type(exc).__name__}: {exc}",
        ) from exc

    headers = {
        "Content-Disposition": 'attachment; filename="mokuhanga.zip"',
        "X-Mokuhanga-DE-Mean": f"{result.reconstruction_dE_mean:.4f}",
        "X-Mokuhanga-DE-P95": f"{result.reconstruction_dE_p95:.4f}",
        "X-Mokuhanga-Block-Count": str(result.block_count),
        "X-Mokuhanga-Pigment-Count": str(result.pigment_count),
        "X-Mokuhanga-Warnings": " | ".join(result.warnings) if result.warnings else "",
    }
    return Response(
        content=result.zip_bytes,
        media_type="application/zip",
        headers=headers,
    )


@router.get("/separate/mokuhanga/health")
async def mokuhanga_health() -> dict[str, object]:
    """Lightweight liveness for the v21 pipeline (imports verify modules)."""
    try:
        from algorithms.decomposition import (  # type: ignore[import-not-found]
            adjacency_graph,
            dsatur_color_aware,
            km_forward_render,
            palette_extract,
            print_order,
            tan_rgb_geometry,
        )
        from adapters import mokuhanga_zip_emitter  # type: ignore[import-not-found]
        modules = {
            "tan_rgb_geometry": bool(tan_rgb_geometry),
            "palette_extract": bool(palette_extract),
            "adjacency_graph": bool(adjacency_graph),
            "dsatur_color_aware": bool(dsatur_color_aware),
            "print_order": bool(print_order),
            "km_forward_render": bool(km_forward_render),
            "mokuhanga_zip_emitter": bool(mokuhanga_zip_emitter),
        }
        ok = all(modules.values())
        return {"ok": ok, "mode": "v21_mokuhanga", "modules": modules}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "mode": "v21_mokuhanga", "error": str(exc)}
