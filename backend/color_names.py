"""Map RGB tuples to short human-readable color names.

Used to label exported plate filenames so printers can tell plates apart at
a glance without decoding hex. Buckets are coarse by design (no blush vs.
salmon split) — the goal is scannability, not color science.
"""
from __future__ import annotations

import colorsys
import re

_SAFE = re.compile(r"[^a-z0-9-]")


_PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("gray", (128, 128, 128)),
    ("red", (220, 40, 40)),
    ("orange", (240, 140, 40)),
    ("yellow", (240, 220, 40)),
    ("olive", (150, 150, 50)),
    ("green", (60, 180, 60)),
    ("teal", (40, 170, 170)),
    ("cyan", (60, 220, 240)),
    ("blue", (50, 90, 210)),
    ("navy", (20, 30, 110)),
    ("purple", (140, 70, 200)),
    ("magenta", (220, 70, 200)),
    ("pink", (240, 160, 190)),
    ("brown", (130, 80, 40)),
    ("tan", (210, 180, 140)),
    ("cream", (250, 240, 210)),
]


def _achromatic_name(r: int, g: int, b: int) -> str | None:
    """Short-circuit grayscale values — their hue is meaningless."""
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    spread = max(r, g, b) - min(r, g, b)
    if spread <= 12:
        if lum < 32:
            return "black"
        if lum > 232:
            return "white"
        return "gray"
    return None


def rgb_to_color_name(rgb: tuple[int, int, int] | list[int]) -> str:
    """Return a short color name for the given RGB triple (0-255 ints)."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    achromatic = _achromatic_name(r, g, b)
    if achromatic:
        return achromatic

    # Match in perceptually-weighted RGB to avoid HSL hue wrap edge cases.
    best_name = _PALETTE[0][0]
    best_dist = float("inf")
    for name, (pr, pg, pb) in _PALETTE:
        # Y'CbCr-ish weighting: luma then chroma.
        dl = (0.3 * (r - pr)) ** 2 + (0.59 * (g - pg)) ** 2 + (0.11 * (b - pb)) ** 2
        if dl < best_dist:
            best_dist = dl
            best_name = name
    return best_name


def color_slug(rgb: tuple[int, int, int] | list[int]) -> str:
    """Filename-safe `<name>-<hex>` slug, e.g. 'red-dc2828'."""
    name = rgb_to_color_name(rgb)
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    hex_part = f"{r:02x}{g:02x}{b:02x}"
    return _SAFE.sub("-", f"{name}-{hex_part}")


def plate_filename_stem(rgb: tuple[int, int, int] | list[int], plate_number: int) -> str:
    """ZIP entry stem: `<plateNumber>_<hex>` (no extension).

    Leading number is the print order so an unzipped folder sorts naturally
    (1_, 2_, 3_ …). Trailing hex is the plate color at a glance.
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    hex_part = f"{r:02x}{g:02x}{b:02x}"
    return f"{int(plate_number)}_{hex_part}"
