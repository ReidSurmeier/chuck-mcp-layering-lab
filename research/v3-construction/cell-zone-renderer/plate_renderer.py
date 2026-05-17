"""Plate.svg / Plate.png renderer.

Renders a single :class:`Plate` as either an SVG (for CNC handoff) or a
PNG preview (for the web UI). The output deliberately mimics the Emma
plate examples in `/srv/woodblock-share/Examples/Screenshot 2026-05-10
at 5.49.23 PM.png` — a wood-grain plywood ground with isolated jigsaw
color regions painted on top.

Design contract (from `docs/reconstruction-plan-2026-05-16.md`):

1. The plate is **mirrored** (horizontal flip) — wood blocks print
   backwards, so the carving is the mirror of the print.
2. The plate is **NOT** a faded final image. It is sparse: only the
   cell-zones belonging to this block are inked. Bare wood shows
   through everywhere else.
3. Kento reference marks are drawn in the corners as a visual
   registration cue. The kagi (L-mark) sits at the bottom-right of
   the print (which is bottom-LEFT of the mirrored plate); the
   hikitsuke (bar) sits along the bottom edge, ~2/3 of the way across
   from the L-mark.
4. The cell-zones are merged where they share a pigment and dilated
   by the mill radius before vectorization, so the carved geometry
   stays brushable.
5. The plate-not-composite property is enforced by construction: only
   the polygons in ``plate.cell_zones`` are emitted; everything else
   is wood ground. There is no global α-map dump.

The wood-grain pattern is built from the CSS-Tricks "pine wood"
feTurbulence snippet (baseFrequency="0.1 0.01", fractalNoise,
feColorMatrix recoloring to plywood-maple tones) and parameterized so
each plate gets a slightly different seed so the contact sheet
doesn't look photocopied.

References:
- https://css-tricks.com/creating-patterns-with-svg-filters/
- https://art-design-glossary.musabi.ac.jp/uchi-kento-soto-kento/
- Mike Lyon, two small woodcuts using a wedged registration jig (2017)
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import svgwrite
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from cz_types import Plate

# ----- public configuration ---------------------------------------------------


@dataclass(frozen=True)
class PlateRenderConfig:
    """Renderer settings shared by SVG + PNG previews."""

    # Output image size in pixels for the PNG preview path.
    preview_size_px: tuple[int, int] = (1024, 1024)
    # Physical paper margin around the print, in fraction of width.
    margin_frac: float = 0.06
    # Mill radius in pixels — cell-zones are dilated by this much before
    # vectorization so the carved boundary stays >= mill diameter wide.
    mill_radius_px: float = 6.0
    # Kento mark sizes, in fraction of the shorter image dimension.
    kento_kagi_frac: float = 0.04
    kento_hikitsuke_frac: float = 0.05
    # Plywood ground colors (sRGB 0..1). Used for both SVG filter
    # recoloring and PNG preview rasterization.
    wood_base: tuple[float, float, float] = (0.78, 0.62, 0.42)
    wood_dark: tuple[float, float, float] = (0.52, 0.36, 0.22)
    # Pigment alpha on plate preview (the paint is wet, not flat).
    pigment_alpha: float = 0.86
    # Optional pigment-color noise to make the brushed paint less plastic.
    pigment_noise_strength: float = 0.05


_DEFAULT_CFG = PlateRenderConfig()


# ----- SVG renderer -----------------------------------------------------------


def _mirrored_polygon_pts(
    poly: Polygon, width: float, mirror: bool
) -> list[tuple[float, float]]:
    """Return polygon exterior points, optionally horizontally mirrored."""
    pts = list(poly.exterior.coords)
    if mirror:
        return [(width - x, y) for (x, y) in pts]
    return [(float(x), float(y)) for (x, y) in pts]


def _polygon_to_svg_path(
    poly: Polygon, *, width: float, mirror: bool, scale: float = 1.0
) -> str:
    """Convert a shapely Polygon (with possible holes) to an SVG path 'd'."""
    parts: list[str] = []

    def _ring_d(coords: list[tuple[float, float]]) -> str:
        if mirror:
            coords = [(width - x, y) for (x, y) in coords]
        coords = [(x * scale, y * scale) for (x, y) in coords]
        if not coords:
            return ""
        head = f"M {coords[0][0]:.2f} {coords[0][1]:.2f} "
        body = " ".join(f"L {x:.2f} {y:.2f}" for x, y in coords[1:])
        return head + body + " Z"

    parts.append(_ring_d(list(poly.exterior.coords)))
    for interior in poly.interiors:
        parts.append(_ring_d(list(interior.coords)))
    return " ".join(p for p in parts if p)


def _merge_cell_zones(
    plate: Plate, *, mill_radius_px: float
) -> MultiPolygon:
    """Union all cell-zone polygons and dilate-erode by the mill radius.

    This is the printability repair step from the vectorization-cnc
    research: a positive then negative buffer kills thin necks and
    1-px-wide spurs that would vanish under tool-radius offset.
    """
    if not plate.cell_zones:
        return MultiPolygon()
    raw = unary_union([cz.polygon for cz in plate.cell_zones])
    # Dilate then erode = closing — fills small holes between adjacent
    # cells of the same plate (they really are one inked region).
    closed = raw.buffer(mill_radius_px).buffer(-mill_radius_px)
    # Then opening — drop thin spurs that are too narrow to carve.
    opened = closed.buffer(-mill_radius_px * 0.5).buffer(mill_radius_px * 0.5)
    if isinstance(opened, Polygon):
        return MultiPolygon([opened]) if not opened.is_empty else MultiPolygon()
    if isinstance(opened, MultiPolygon):
        return opened
    return MultiPolygon()


def _rgb_to_svg_color(rgb: tuple[float, float, float]) -> str:
    r, g, b = (int(max(0, min(1, c)) * 255) for c in rgb)
    return f"rgb({r},{g},{b})"


def _wood_filter_matrix(
    base: tuple[float, float, float],
    dark: tuple[float, float, float],
) -> str:
    """Build feColorMatrix values mapping fractalNoise α → wood tones.

    Each row of the 4×5 matrix is `r' = R*r + G*g + B*b + A*a + bias`.
    feTurbulence emits luma/alpha noise — we want to drive the bias by
    base and the α-weight by (dark - base), so peaks of the noise pull
    the color toward `dark`.
    """
    rows = []
    for i in range(3):
        a_weight = base[i] - dark[i]  # positive: noise alpha lifts toward dark
        bias = dark[i]
        # zero out r/g/b passthrough, drive entirely from alpha + bias
        rows.append(f"0 0 0 {a_weight:.3f} {bias:.3f}")
    rows.append("0 0 0 0 1")
    return "  ".join(rows)


def render_plate_svg(
    plate: Plate,
    output_path: str | Path,
    *,
    width: int = 1024,
    height: int = 1024,
    mirror: bool = True,
    cfg: PlateRenderConfig = _DEFAULT_CFG,
    seed: int | None = None,
) -> Path:
    """Render a Plate to an SVG file.

    The SVG carries:
        - A wood-grain filtered rect as the ground.
        - One or more <path> elements for the merged, mill-safe
          cell-zone polygons, filled with the plate's pigment color.
        - A kagi (L) + hikitsuke (bar) registration cue at the bottom.

    Coordinates are in pixel units of the source image; the SVG
    viewBox is set so the file is resolution-independent.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The cell-zone polygons live in source-image pixel space. We need
    # to know that space's bounding box so the SVG viewBox covers it
    # cleanly. If all zones are inside ``width``×``height``, use that.
    src_w, src_h = float(width), float(height)

    dwg = svgwrite.Drawing(
        str(output_path),
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {src_w} {src_h}",
        profile="tiny",  # looser validator — accepts matrix strings
        debug=False,
    )

    # --- wood-grain filter -----------------------------------------------------
    s = seed if seed is not None else (plate.block_id * 137 + 11) % 9973
    wood_filter = dwg.defs.add(
        dwg.filter(id=f"wood_{plate.block_id}", x="0%", y="0%", width="100%", height="100%")
    )
    wood_filter.feTurbulence(
        type="fractalNoise",
        baseFrequency="0.014 0.0022",
        numOctaves=2,
        seed=s,
        result="noise",
    )
    wood_filter.feColorMatrix(
        in_="noise",
        type="matrix",
        values=_wood_filter_matrix(cfg.wood_base, cfg.wood_dark),
        result="wood",
    )
    # Second turbulence pass: fine grain knots
    wood_filter.feTurbulence(
        type="turbulence",
        baseFrequency="0.6 0.06",
        numOctaves=1,
        seed=(s + 7) % 9973,
        result="grain",
    )
    wood_filter.feColorMatrix(
        in_="grain",
        type="matrix",
        values="0 0 0 0.08 -0.04  0 0 0 0.07 -0.03  0 0 0 0.06 -0.02  0 0 0 0 1",
        result="grain_tinted",
    )
    wood_filter.feComposite(
        in_="grain_tinted", in2="wood", operator="arithmetic",
        k1=0, k2=1, k3=1, k4=0
    )

    # --- ground rect -----------------------------------------------------------
    dwg.add(
        dwg.rect(
            insert=(0, 0),
            size=(src_w, src_h),
            fill=_rgb_to_svg_color(cfg.wood_base),
            filter=f"url(#wood_{plate.block_id})",
        )
    )

    # --- cell-zone inked region ------------------------------------------------
    merged = _merge_cell_zones(plate, mill_radius_px=cfg.mill_radius_px)
    pigment_color = _rgb_to_svg_color(plate.pigment_color)
    pigment_alpha = cfg.pigment_alpha * plate.opacity

    inked_group = dwg.add(dwg.g(
        id=f"plate_{plate.block_id}_inked",
        fill=pigment_color,
        fill_opacity=f"{pigment_alpha:.3f}",
        stroke="none",
    ))
    if not merged.is_empty:
        for poly in merged.geoms:
            d = _polygon_to_svg_path(poly, width=src_w, mirror=mirror)
            if d:
                inked_group.add(dwg.path(d=d))

    # --- kento registration cue (visual only — physical kento is on the jig) --
    short = min(src_w, src_h)
    kagi = cfg.kento_kagi_frac * short
    bar_w = cfg.kento_hikitsuke_frac * short
    bar_h = max(2.0, kagi * 0.18)
    margin = src_w * cfg.margin_frac * 0.5

    kento_group = dwg.add(dwg.g(
        id="kento_marks",
        fill="rgb(80,55,30)",
        fill_opacity="0.55",
    ))

    # Logical placement (on the PRINT): kagi bottom-right, hikitsuke
    # roughly 2/3 across bottom from the kagi (so it sits closer to
    # the centre-left when the print is right-side up).
    print_bottom_right = (src_w - margin, src_h - margin)
    print_hikitsuke = (src_w * 0.45, src_h - margin)

    def _maybe_mirror(p: tuple[float, float]) -> tuple[float, float]:
        return (src_w - p[0], p[1]) if mirror else p

    bx, by = _maybe_mirror(print_bottom_right)
    hx, hy = _maybe_mirror(print_hikitsuke)

    # kagi: L-shape (two thin rects)
    kento_group.add(dwg.rect(insert=(bx - kagi, by - bar_h), size=(kagi, bar_h)))
    kento_group.add(dwg.rect(insert=(bx - bar_h, by - kagi), size=(bar_h, kagi)))
    # hikitsuke: a single straight bar
    kento_group.add(dwg.rect(insert=(hx - bar_w / 2, hy - bar_h), size=(bar_w, bar_h)))

    # --- title metadata --------------------------------------------------------
    dwg.add(
        dwg.text(
            f"block_{plate.block_id:02d} • {plate.role} • {plate.pigment_name}",
            insert=(margin, margin + 10),
            font_family="monospace",
            font_size=11,
            fill="rgb(60,40,20)",
            fill_opacity="0.65",
        )
    )

    dwg.save(pretty=True)
    return output_path


# ----- PNG preview renderer ---------------------------------------------------


def _wood_ground_png(
    width: int, height: int, cfg: PlateRenderConfig, seed: int
) -> Image.Image:
    """Procedural plywood ground for the PNG preview path.

    We rasterize the same idea as the SVG filter: stretched fractal
    noise mapped to wood tones, plus a fine grain pass. Pure numpy +
    PIL, no SciPy.

    The plywood look comes from three superimposed components:
        1. Long horizontal grain stripes (low freq, very anisotropic).
        2. Mid-frequency annular rings (warped sine pattern).
        3. Fine speckle for the printed surface texture.
    """
    rng = np.random.default_rng(seed)

    # --- 1) horizontal grain stripes -----------------------------------------
    # Low x-frequency, very high y-frequency = long parallel stripes.
    stripe_freq = 70 + (seed % 10)  # ~70 stripes vertically
    y_idx = np.arange(height, dtype=np.float32)[:, None]
    phase = rng.uniform(0, 2 * math.pi)
    stripes = 0.5 + 0.5 * np.sin(2 * math.pi * y_idx * stripe_freq / height + phase)
    # Warp stripes slightly so they aren't perfectly straight.
    x_idx = np.arange(width, dtype=np.float32)[None, :]
    warp = 6.0 * np.sin(2 * math.pi * x_idx / width * 1.4 + phase * 1.3)
    warped_y = y_idx + warp
    stripes = 0.5 + 0.5 * np.sin(
        2 * math.pi * warped_y * stripe_freq / height + phase
    )
    # Smooth across the stripe boundary so they're soft, not hard lines.
    stripes_img = Image.fromarray((stripes * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=1.4)
    )
    stripes = np.asarray(stripes_img, dtype=np.float32) / 255.0
    # Squash dynamic range: stripes should be subtle (15-25% modulation).
    stripes = 0.6 + 0.4 * stripes  # 0.6..1.0

    # --- 2) mid-frequency blotches -------------------------------------------
    mid_w, mid_h = max(4, width // 6), max(4, height // 6)
    mid = rng.random((mid_h, mid_w), dtype=np.float32)
    mid_img = Image.fromarray((mid * 255).astype(np.uint8), "L").resize(
        (width, height), Image.Resampling.BILINEAR
    ).filter(ImageFilter.GaussianBlur(radius=2.0))
    mid_arr = np.asarray(mid_img, dtype=np.float32) / 255.0

    # --- 3) fine speckle ------------------------------------------------------
    fine = rng.random((height, width), dtype=np.float32)
    fine_img = Image.fromarray((fine * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=0.5)
    )
    fine_arr = (np.asarray(fine_img, dtype=np.float32) / 255.0 - 0.5) * 0.08

    # --- combine -------------------------------------------------------------
    base = np.array(cfg.wood_base, dtype=np.float32)[None, None, :]
    dark = np.array(cfg.wood_dark, dtype=np.float32)[None, None, :]

    # Stripes drive most of the contrast; mid_arr adds slow color bias;
    # fine_arr adds surface speckle.
    blend = 1.0 - stripes  # darker where stripe is "low"
    blend = blend * 0.65 + (mid_arr - 0.5) * 0.30 + 0.25
    blend = np.clip(blend, 0.0, 1.0)[..., None]
    rgb = base * (1 - blend) + dark * blend + fine_arr[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def _polygon_to_pixels(
    poly: Polygon, w: int, h: int, mirror: bool, src_w: float, src_h: float
) -> list[tuple[float, float]]:
    sx, sy = w / src_w, h / src_h
    pts = list(poly.exterior.coords)
    if mirror:
        pts = [(src_w - x, y) for (x, y) in pts]
    return [(x * sx, y * sy) for (x, y) in pts]


def render_plate_preview(
    plate: Plate,
    output_path: str | Path,
    *,
    src_size: tuple[int, int] = (1024, 1024),
    mirror: bool = True,
    cfg: PlateRenderConfig = _DEFAULT_CFG,
    seed: int | None = None,
) -> Path:
    """Raster PNG preview of a plate (for the web UI block grid).

    The output is at ``cfg.preview_size_px`` and renders a procedural
    plywood ground with the merged, mill-safe inked region on top in
    the plate's pigment color. Same plate-not-composite property as
    ``render_plate_svg``: only the polygons in ``plate.cell_zones``
    are painted.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = cfg.preview_size_px
    src_w, src_h = float(src_size[0]), float(src_size[1])

    s = seed if seed is not None else (plate.block_id * 137 + 11) % 9973
    ground = _wood_ground_png(w, h, cfg, s)

    merged = _merge_cell_zones(plate, mill_radius_px=cfg.mill_radius_px)
    if merged.is_empty:
        ground.save(output_path, "PNG", optimize=True)
        return output_path

    # Paint the pigment region on a transparent overlay so we can use
    # alpha compositing and add slight brush-stroke noise to the
    # pigment colour.
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    pr, pg, pb = (int(max(0, min(1, c)) * 255) for c in plate.pigment_color)
    alpha = int(255 * cfg.pigment_alpha * plate.opacity)
    for poly in merged.geoms:
        pts = _polygon_to_pixels(poly, w, h, mirror, src_w, src_h)
        if len(pts) >= 3:
            draw.polygon(pts, fill=(pr, pg, pb, alpha))
        for interior in poly.interiors:
            ipts = _polygon_to_pixels(
                Polygon(list(interior.coords)), w, h, mirror, src_w, src_h
            )
            if len(ipts) >= 3:
                draw.polygon(ipts, fill=(0, 0, 0, 0))

    # Add a tiny brush-noise modulation to the pigment alpha so the
    # ink doesn't look like flat plastic.
    if cfg.pigment_noise_strength > 0:
        rng = np.random.default_rng(s + 31)
        ov_arr = np.asarray(overlay, dtype=np.float32) / 255.0
        noise = rng.random((h, w), dtype=np.float32)
        ov_arr[..., 3] *= (1.0 - cfg.pigment_noise_strength * (noise - 0.5) * 2.0)
        ov_arr = np.clip(ov_arr, 0.0, 1.0)
        overlay = Image.fromarray((ov_arr * 255).astype(np.uint8), "RGBA")

    composed = Image.alpha_composite(ground.convert("RGBA"), overlay)

    # Draw kento cues last so they sit on top of the paint
    short = min(w, h)
    kagi = int(cfg.kento_kagi_frac * short)
    bar_w = int(cfg.kento_hikitsuke_frac * short)
    bar_h = max(2, int(kagi * 0.18))
    margin = int(w * cfg.margin_frac * 0.5)
    bx = (w - margin) if not mirror else margin
    by = h - margin
    hx = int(w * (0.45 if not mirror else 0.55))
    hy = h - margin
    kdraw = ImageDraw.Draw(composed, "RGBA")
    kc = (80, 55, 30, 170)
    if mirror:
        kdraw.rectangle([bx, by - bar_h, bx + kagi, by], fill=kc)
        kdraw.rectangle([bx, by - kagi, bx + bar_h, by], fill=kc)
    else:
        kdraw.rectangle([bx - kagi, by - bar_h, bx, by], fill=kc)
        kdraw.rectangle([bx - bar_h, by - kagi, bx, by], fill=kc)
    kdraw.rectangle([hx - bar_w // 2, hy - bar_h, hx + bar_w // 2, hy], fill=kc)

    composed.convert("RGB").save(output_path, "PNG", optimize=True)
    return output_path
