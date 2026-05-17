"""Pull renderer — cell-zone → α-map → K-M overprint on previous proof state.

This is the *one place* the α-map representation is allowed to live
(per Q26 of the design lock). The function takes:

    - a :class:`Plate` (cell-zones + pigment color + opacity)
    - the previous proof state image (H×W×3 sRGB, NORMAL orientation)
    - a per-pull opacity multiplier (how much ink was loaded this time)

and returns the cumulative print after this pull, still in normal
orientation. Mirror lives only on the plate SVG; the printed paper
is right-reading.

Renderer physics — Kubelka-Munk approximation:

We treat each pigment as a partially-opaque overprint layer. For each
pixel that this pull deposits ink at (the rasterized cell-zone mask),
the new color is:

    new_rgb = prev_rgb * (1 - alpha) + pigment_rgb * alpha

That is the standard porter-duff "source over" composite — the t1
Mixbox/K-M tier the locked design ships with (calibrated K-M is
deferred to V2). The α-map is the rasterized inked mask, soft-edged
by a small Gaussian to mimic the brushed-ink boundary, modulated by
opacity * ink_density.

For high-quality V2-onwards work, swap in
``backend.services.v23.core.forward_render_jax`` here; the function
signature is identical.

Performance: each plate is rasterized ONCE and cached. A 132-pull
sequence at 2048×2048 with 27 plates does 27 rasterizations + 132
np.where composites — about 7-10× faster than re-rasterizing per
pull.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Polygon
from shapely.ops import unary_union

from cz_types import Plate

# (id(plate), out_size, src_size, softness) → α-map ndarray
_ALPHA_CACHE: dict[tuple[int, tuple[int, int], tuple[int, int], float], NDArray[np.float32]] = {}


def clear_alpha_cache() -> None:
    """Drop the renderer's α-map cache. Call between runs."""
    _ALPHA_CACHE.clear()


def _rasterize_cell_zones(
    plate: Plate,
    out_size: tuple[int, int],
    src_size: tuple[int, int],
    softness_px: float = 1.5,
) -> NDArray[np.float32]:
    """Rasterize the plate's cell-zones to a soft α-map (H×W float 0..1).

    The α-map is PRIVATE to the renderer; it is not stored anywhere
    outside this function (per design Q26).
    """
    w, h = out_size
    src_w, src_h = float(src_size[0]), float(src_size[1])
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if not plate.cell_zones:
        return np.zeros((h, w), dtype=np.float32)

    union = unary_union([cz.polygon for cz in plate.cell_zones])
    polys: list[Polygon]
    if isinstance(union, Polygon):
        polys = [union] if not union.is_empty else []
    else:
        polys = [p for p in getattr(union, "geoms", []) if not p.is_empty]

    sx, sy = w / src_w, h / src_h
    for poly in polys:
        ext = [(x * sx, y * sy) for (x, y) in poly.exterior.coords]
        if len(ext) >= 3:
            draw.polygon(ext, fill=255)
        for interior in poly.interiors:
            ipts = [(x * sx, y * sy) for (x, y) in interior.coords]
            if len(ipts) >= 3:
                draw.polygon(ipts, fill=0)

    if softness_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=softness_px))
    return np.asarray(mask, dtype=np.float32) / 255.0


def render_pull(
    plate: Plate,
    prev_proof_state: NDArray[np.float32],
    *,
    opacity: float = 1.0,
    ink_density: float = 1.0,
    softness_px: float = 1.5,
    src_size: tuple[int, int] | None = None,
) -> NDArray[np.float32]:
    """Composite the next pull onto ``prev_proof_state`` and return the result.

    Parameters
    ----------
    plate:
        The plate being pulled. Cell-zone polygons drive the α-map.
    prev_proof_state:
        H×W×3 float32 sRGB in 0..1, NORMAL orientation. Pass an empty
        paper-colored array for the very first pull.
    opacity:
        Per-pull opacity multiplier. Multiplied into ``plate.opacity``.
    ink_density:
        Optional extra knob (mostly for ``Pull.ink_density`` plumbing).
    softness_px:
        Gaussian blur radius on the rasterized α-map. Mimics brushed
        ink edges so pulls don't look like vector decals.
    src_size:
        Original image (w, h) the cell-zone coordinates are in. If
        None, inferred from ``prev_proof_state.shape``.

    Returns
    -------
    H×W×3 float32 sRGB in 0..1, normal orientation, cumulative after
    this pull. Same shape as ``prev_proof_state``.
    """
    if prev_proof_state.ndim != 3 or prev_proof_state.shape[-1] != 3:
        raise ValueError(
            f"prev_proof_state must be HxWx3, got {prev_proof_state.shape!r}"
        )
    h, w, _ = prev_proof_state.shape
    if src_size is None:
        src_size = (w, h)

    key = (id(plate), (w, h), src_size, float(softness_px))
    alpha = _ALPHA_CACHE.get(key)
    if alpha is None:
        alpha = _rasterize_cell_zones(
            plate, out_size=(w, h), src_size=src_size, softness_px=softness_px
        )
        _ALPHA_CACHE[key] = alpha
    base_scale = float(plate.opacity) * float(opacity) * float(ink_density)
    # Only touch pixels where alpha is non-trivially > 0. This is
    # the main perf win at 2048×2048.
    mask = alpha > 1e-3
    if not mask.any():
        return prev_proof_state.astype(np.float32, copy=True)
    out = prev_proof_state.astype(np.float32, copy=True)
    eff = np.clip(alpha[mask] * base_scale, 0.0, 1.0)[:, None]
    pigment = np.asarray(plate.pigment_color, dtype=np.float32)[None, :]
    out[mask] = out[mask] * (1.0 - eff) + pigment * eff
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def blank_proof_state(
    size: tuple[int, int],
    paper_rgb: tuple[float, float, float] = (0.96, 0.94, 0.88),
) -> NDArray[np.float32]:
    """Convenience: a fresh paper-colored proof state ``size`` = (w, h)."""
    w, h = size
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[..., 0] = paper_rgb[0]
    out[..., 1] = paper_rgb[1]
    out[..., 2] = paper_rgb[2]
    return out
