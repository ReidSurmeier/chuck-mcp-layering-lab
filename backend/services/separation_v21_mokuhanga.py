"""V21 Mokuhanga separation pipeline — orchestrator.

Composes Tan RGB-geometry decomposition, mixbox pigment snapping,
DSATUR color-aware block assignment, Kubelka-Munk forward rendering,
and CIEDE2000 reconstruction error into the mokuhanga ZIP bundle.

This module does NOT reimplement primitives. Every stage is delegated
to a sibling module.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from adapters.mokuhanga_zip_emitter import emit_mokuhanga_zip
from algorithms.decomposition.adjacency_graph import build_adjacency_graph
from algorithms.decomposition.dsatur_color_aware import BlockAssignment, dsatur_color_aware
from algorithms.decomposition.km_forward_render import (
    composite_delta_e2000,
    forward_render_km,
)
from algorithms.decomposition.palette_extract import (
    Pigment,
    build_palette,
    srgb_to_oklab,
)
from algorithms.decomposition.print_order import (
    Impression,
    compute_luminance_oklab,
    order_impressions,
)
from algorithms.decomposition.tan_rgb_geometry import (
    decompose_image,
    extract_palette_hull,
)

__all__ = ["V21Params", "V21Result", "separate_mokuhanga"]

_LOG = logging.getLogger(__name__)
_U8 = NDArray[np.uint8]
_F32 = NDArray[np.float32]
_Bool = NDArray[np.bool_]
_DE_WARN: float = 1.5


@dataclass(frozen=True, slots=True)
class V21Params:
    """Tunable knobs for the mokuhanga pipeline."""

    palette_size: int = 13
    threshold_tau: float = 0.10
    bleed_tolerance_px: int = 8
    color_grouping_weight: float = 0.6
    sliver_threshold_px: int | None = None
    print_order_mode: Literal["light_to_dark", "dark_to_light"] = "light_to_dark"
    substrate_rgb: tuple[int, int, int] = (255, 255, 255)


@dataclass(slots=True)
class V21Result:
    """End-to-end bundle returned to callers."""

    zip_bytes: bytes
    reconstruction_dE_mean: float
    reconstruction_dE_p95: float
    block_count: int
    pigment_count: int
    palette: list[Pigment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@contextmanager
def _stage(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    _LOG.info("v21.stage.start name=%s", name)
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        _LOG.info("v21.stage.end name=%s ms=%.1f", name, dt)


def _decode_rgb(image_bytes: bytes) -> _U8:
    with Image.open(io.BytesIO(image_bytes)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"decoded image has wrong shape {arr.shape}")
    return arr


def _png(arr: NDArray[Any], mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(max(0, min(255, c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _pigment_hex(p: Pigment) -> str:
    if p.hex:
        h = p.hex if p.hex.startswith("#") else f"#{p.hex}"
        return h.lower()
    return _rgb_hex(p.rgb)


def _palette_oklab(palette: list[Pigment]) -> dict[str, tuple[float, float, float]]:
    """Build pigment_id -> (L, a, b) dict for dsatur_color_aware."""
    rgb_arr = np.asarray([p.rgb for p in palette], dtype=np.uint8)
    lab_arr = srgb_to_oklab(rgb_arr)  # (N, 3) float64
    return {
        palette[i].id: (float(lab_arr[i, 0]), float(lab_arr[i, 1]), float(lab_arr[i, 2]))
        for i in range(len(palette))
    }


def _coverage_pct(mask: _Bool) -> float:
    total = mask.size
    if total == 0:
        return 0.0
    return float(mask.sum()) / float(total) * 100.0


def _build_pigment_meta(
    palette: list[Pigment],
    masks: list[_Bool],
) -> dict[str, dict[str, Any]]:
    """Build pigment_meta for order_impressions: hex, rgb, coverage_pct per pigment id."""
    meta: dict[str, dict[str, Any]] = {}
    for i, p in enumerate(palette):
        cov = _coverage_pct(masks[i]) if i < len(masks) else 0.0
        meta[p.id] = {
            "hex": _pigment_hex(p),
            "rgb": list(p.rgb),
            "coverage_pct": cov,
        }
    return meta


def _trace_path_d(mask: _Bool) -> str:
    """Run potrace on a binary mask -> compound SVG path-d (empty if none)."""
    if not mask.any():
        return ""
    import potrace  # type: ignore[import-not-found]

    bmp = potrace.Bitmap(~mask.astype(np.bool_))
    path = bmp.trace(turdsize=2, alphamax=1.0, opttolerance=0.2)
    sub: list[str] = []
    for curve in path:
        parts: list[str] = [f"M {curve.start_point.x:.3f},{curve.start_point.y:.3f}"]
        for seg in curve:
            if seg.is_corner:
                parts.append(
                    f"L {seg.c.x:.3f},{seg.c.y:.3f} L {seg.end_point.x:.3f},{seg.end_point.y:.3f}"
                )
            else:
                parts.append(
                    f"C {seg.c1.x:.3f},{seg.c1.y:.3f} {seg.c2.x:.3f},{seg.c2.y:.3f} {seg.end_point.x:.3f},{seg.end_point.y:.3f}"
                )
        parts.append("Z")
        sub.append(" ".join(parts))
    return " ".join(sub)


def _svg_open(w: int, h: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" shape-rendering="geometricPrecision">\n'
    )


def _color_svg(mask: _Bool, hex_c: str, w: int, h: int) -> str:
    d = _trace_path_d(mask)
    body = (
        f'  <path d="{d}" fill="{hex_c}" fill-rule="evenodd" stroke="none"/>\n'
        if d
        else ""
    )
    return _svg_open(w, h) + body + "</svg>\n"


def _build_block_svgs(
    assignment: BlockAssignment,
    palette: list[Pigment],
    pid_to_idx: dict[str, int],
    masks: list[_Bool],
    w: int,
    h: int,
) -> dict[int, str]:
    """Group per-pigment masks by block and union into one SVG per block."""
    by_block: dict[int, list[int]] = {}
    for pig_id, blk_idx in assignment.color_to_block.items():
        pig_idx = pid_to_idx.get(pig_id)
        if pig_idx is None:
            continue
        by_block.setdefault(blk_idx, []).append(pig_idx)

    result: dict[int, str] = {}
    for blk_idx in sorted(by_block):
        gs: list[str] = []
        for pig_idx in by_block[blk_idx]:
            if pig_idx >= len(masks):
                continue
            d = _trace_path_d(masks[pig_idx])
            if not d:
                continue
            hx = _pigment_hex(palette[pig_idx])
            gs.append(
                f'  <g fill="{hx}"><path d="{d}" fill-rule="evenodd" stroke="none"/></g>\n'
            )
        result[blk_idx] = _svg_open(w, h) + "".join(gs) + "</svg>\n"
    return result


def _impression_previews(
    per_pigment_alphas: list[NDArray[np.floating]],
    palette_rgbs: list[tuple[int, int, int]],
    print_order_indices: list[int],
    substrate: tuple[int, int, int],
) -> list[bytes]:
    """Cumulative impression PNGs, one per step."""
    out: list[bytes] = []
    for k in range(1, len(print_order_indices) + 1):
        rgb = forward_render_km(
            per_pigment_alphas,
            palette_rgbs,
            print_order=print_order_indices[:k],
            substrate_rgb=substrate,
        )
        out.append(_png(np.clip(np.asarray(rgb), 0, 255).astype(np.uint8), "RGB"))
    return out


def separate_mokuhanga(
    image_bytes: bytes, *, params: V21Params = V21Params()
) -> V21Result:
    """End-to-end mokuhanga separation pipeline."""
    with _stage("decode"):
        rgb = _decode_rgb(image_bytes)
        h, w = rgb.shape[:2]

    # --- Stage 1: palette extraction + mixbox snapping ---
    with _stage("palette_extract"):
        free_palette = extract_palette_hull(
            rgb, target_palette_size=params.palette_size
        )
        # build_palette(palette_rgb_01, deduplicate) — no MIXBOX_PIGMENTS arg
        palette: list[Pigment] = build_palette(free_palette, deduplicate=True)
        # Palette as (N, 3) uint8 for downstream; also as list of tuples for km renderer
        palette_rgb_u8 = np.asarray([p.rgb for p in palette], dtype=np.uint8)  # (N,3)
        palette_rgbs: list[tuple[int, int, int]] = [p.rgb for p in palette]
        # float64 [0,1] for tan decomposition
        palette_rgb_01 = palette_rgb_u8.astype(np.float64) / 255.0
        # stable pigment-id -> palette-index map
        pid_to_idx: dict[str, int] = {p.id: i for i, p in enumerate(palette)}

    # --- Stage 2: Tan RGB-geometry barycentric decomposition ---
    with _stage("decompose"):
        # decompose_image returns (palette, alphas) where alphas is (H,W,K) float32
        _, alphas_hwk = decompose_image(rgb, palette=palette_rgb_01)  # (H,W,N)

    # --- Stage 3: threshold alphas to binary masks ---
    with _stage("threshold"):
        bin3d = alphas_hwk >= float(params.threshold_tau)
        masks: list[_Bool] = [bin3d[..., k] for k in range(bin3d.shape[-1])]

    # --- Stage 4: region adjacency graph — requires color_ids as 2nd positional ---
    with _stage("adjacency"):
        color_ids = [p.id for p in palette]
        graph = build_adjacency_graph(
            masks,
            color_ids,
            bleed_tolerance_px=params.bleed_tolerance_px,
            sliver_threshold_px=params.sliver_threshold_px,
        )

    # --- Stage 5: DSATUR — needs color_oklab dict[pigment_id -> (L,a,b)] ---
    with _stage("dsatur"):
        color_oklab = _palette_oklab(palette)
        assignment: BlockAssignment = dsatur_color_aware(
            graph,
            color_oklab,
            color_grouping_weight=params.color_grouping_weight,
        )

    # --- Stage 6: print order — pigment_meta needs hex + rgb + coverage_pct ---
    with _stage("print_order"):
        pigment_meta = _build_pigment_meta(palette, masks)
        # order_impressions(block_assignments: Mapping[str,int], pigment_meta, direction)
        impressions: list[Impression] = order_impressions(
            assignment.color_to_block,
            pigment_meta,
            params.print_order_mode,
        )
        # indices into palette list for the KM renderer
        print_order_indices: list[int] = [
            pid_to_idx[imp.pigment_id]
            for imp in impressions
            if imp.pigment_id in pid_to_idx
        ]

    # --- Stage 7: forward render — needs list of (H,W) arrays, not (H,W,K) tensor ---
    with _stage("forward_render"):
        per_pigment_alphas: list[NDArray[np.floating]] = [
            alphas_hwk[..., i].astype(np.float64) for i in range(len(palette))
        ]
        rec_u8: _U8 = forward_render_km(
            per_pigment_alphas,
            palette_rgbs,
            print_order=print_order_indices,
            substrate_rgb=params.substrate_rgb,
        )

    with _stage("delta_e"):
        de_mean, de_p95 = composite_delta_e2000(rgb, rec_u8)

    # --- Stage 8: SVG tracing ---
    with _stage("trace_svgs"):
        # Plates: one per non-empty pigment mask (v20-compatible)
        plates: list[dict[str, Any]] = []
        for k, mask in enumerate(masks):
            if not mask.any():
                continue
            p = palette[k]
            hx = _pigment_hex(p)
            plates.append(
                {
                    "rank": k,
                    "index": k,
                    "pigment_id": p.id,
                    "hex": hx,
                    "color_hex": hx,
                    "width": w,
                    "height": h,
                    "png_bytes": _png(mask.astype(np.uint8) * 255, "L"),
                    "svg": _color_svg(mask, hx, w, h),
                }
            )
        block_svgs: dict[int, str] = _build_block_svgs(
            assignment, palette, pid_to_idx, masks, w, h
        )

    # --- Stage 9: cumulative impression previews ---
    with _stage("impressions"):
        impression_previews: list[bytes] = _impression_previews(
            per_pigment_alphas,
            palette_rgbs,
            print_order_indices,
            params.substrate_rgb,
        )

    # --- Stage 10: emit ZIP ---
    with _stage("emit_zip"):
        pigment_palette_dicts: list[dict[str, Any]] = [
            {
                "id": p.id,
                "name": p.name,
                "rgb": list(p.rgb),
                "hex": _pigment_hex(p),
                "alpha_prior": 1.0,
            }
            for p in palette
        ]
        impressions_dicts: list[dict[str, Any]] = [
            {
                "step": imp.step,
                "block_id": imp.block_id,
                "pigment_id": imp.pigment_id,
                "hex": imp.pigment_hex,
                "coverage_pct": imp.coverage_pct,
                "luminance": imp.luminance_okL,
            }
            for imp in impressions
        ]
        zip_bytes = emit_mokuhanga_zip(
            composite_png_bytes=_png(rec_u8, "RGB"),
            plates=plates,
            pigment_palette=pigment_palette_dicts,
            block_assignment={k: v for k, v in assignment.color_to_block.items()},
            impressions=impressions_dicts,
            impression_previews=impression_previews,
            block_svgs=block_svgs,
        )

    warnings: list[str] = (
        [f"reconstruction_dE_mean={de_mean:.2f} exceeds {_DE_WARN}"]
        if de_mean >= _DE_WARN
        else []
    )
    return V21Result(
        zip_bytes=zip_bytes,
        reconstruction_dE_mean=float(de_mean),
        reconstruction_dE_p95=float(de_p95),
        block_count=assignment.block_count,
        pigment_count=len(plates),
        palette=palette,
        warnings=warnings,
    )
