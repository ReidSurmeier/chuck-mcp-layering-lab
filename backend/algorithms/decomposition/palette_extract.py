"""Snap extracted palette colors to the nearest mixbox 13-pigment primary.

Mixbox (Sochorová & Jamriška, 2021) provides a real-pigment mixing model
trained on 13 canonical artist pigments. Tan & Co.'s palette extraction
yields perceptually salient colors that may sit anywhere in sRGB; before we
hand a palette to the mixbox sampler we snap each color to the nearest of
the 13 calibrated primaries using OKLab distance (Ottosson 2020), which is
much closer to human-perceived color difference than RGB Euclidean.

This keeps the downstream block-decomposition honest: every plate we
generate corresponds to a pigment the mixer actually knows how to blend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import colour
import numpy as np
import numpy.typing as npt

__all__ = [
    "Pigment",
    "MIXBOX_PIGMENTS",
    "srgb_to_oklab",
    "snap_to_mixbox",
    "build_palette",
]


@dataclass(frozen=True, slots=True)
class Pigment:
    """A mixbox primary pigment."""

    id: str
    name: str
    rgb: tuple[int, int, int]
    hex: str


# Canonical mixbox 13-pigment palette. Order matches the mixbox docs so
# downstream consumers can index into mixbox latent vectors if needed.
MIXBOX_PIGMENTS: tuple[Pigment, ...] = (
    Pigment("cadmium_yellow", "Cadmium Yellow", (254, 236, 0), "#feec00"),
    Pigment("hansa_yellow", "Hansa Yellow", (252, 211, 0), "#fcd300"),
    Pigment("cadmium_orange", "Cadmium Orange", (255, 105, 0), "#ff6900"),
    Pigment("cadmium_red", "Cadmium Red", (255, 39, 2), "#ff2702"),
    Pigment("quinacridone_magenta", "Quinacridone Magenta", (128, 2, 46), "#80022e"),
    Pigment("cobalt_violet", "Cobalt Violet", (78, 0, 66), "#4e0042"),
    Pigment("ultramarine_blue", "Ultramarine Blue", (25, 0, 89), "#190059"),
    Pigment("cobalt_blue", "Cobalt Blue", (0, 33, 133), "#002185"),
    Pigment("phthalo_blue", "Phthalo Blue", (13, 27, 68), "#0d1b44"),
    Pigment("phthalo_green", "Phthalo Green", (0, 60, 50), "#003c32"),
    Pigment("permanent_green", "Permanent Green", (7, 109, 22), "#076d16"),
    Pigment("sap_green", "Sap Green", (107, 148, 4), "#6b9404"),
    Pigment("burnt_sienna", "Burnt Sienna", (123, 72, 0), "#7b4800"),
)


def srgb_to_oklab(rgb: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
    """Convert sRGB to OKLab via colour-science.

    Accepts uint8 (0-255) or float (0-1) arrays of shape ``(..., 3)`` and
    returns float64 OKLab of the same leading shape.
    """
    arr = np.asarray(rgb)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected trailing dim of 3, got shape {arr.shape}")
    divisor = 255.0 if np.issubdtype(arr.dtype, np.integer) else 1.0
    rgb_f = arr.astype(np.float64) / divisor
    result = colour.convert(rgb_f, "sRGB", "Oklab")
    return np.asarray(result, dtype=np.float64)


def _pigment_oklab_table(pigments: Sequence[Pigment]) -> npt.NDArray[np.float64]:
    rgbs = np.asarray([p.rgb for p in pigments], dtype=np.uint8)
    return srgb_to_oklab(rgbs)


# Cache OKLab table for the default palette; ~13x3 floats, trivial RAM.
_MIXBOX_OKLAB: npt.NDArray[np.float64] = _pigment_oklab_table(MIXBOX_PIGMENTS)


def snap_to_mixbox(
    palette_rgb_01: npt.NDArray[np.floating],
    available_pigments: Sequence[Pigment] = MIXBOX_PIGMENTS,
) -> list[Pigment]:
    """Snap each palette color to the nearest available pigment in OKLab.

    ``palette_rgb_01`` is shape ``(N, 3)`` in sRGB 0-1. Returns one Pigment
    per input row, in input order (no dedup; see ``build_palette``).
    """
    arr = np.asarray(palette_rgb_01, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected (N, 3) palette, got shape {arr.shape}")

    if available_pigments is MIXBOX_PIGMENTS:
        pigment_lab = _MIXBOX_OKLAB
    else:
        pigment_lab = _pigment_oklab_table(available_pigments)

    palette_lab = srgb_to_oklab(arr)
    # Squared Euclidean is monotonic with Euclidean -- skip the sqrt.
    diff = palette_lab[:, None, :] - pigment_lab[None, :, :]
    dist_sq = np.einsum("nki,nki->nk", diff, diff)
    nearest = np.argmin(dist_sq, axis=1)
    return [available_pigments[int(i)] for i in nearest]


def build_palette(
    palette_rgb_01: npt.NDArray[np.floating],
    deduplicate: bool = True,
) -> list[Pigment]:
    """Snap a palette to mixbox pigments and optionally dedupe (order preserved)."""
    snapped = snap_to_mixbox(palette_rgb_01)
    if not deduplicate:
        return snapped
    seen: set[str] = set()
    unique: list[Pigment] = []
    for pig in snapped:
        if pig.id not in seen:
            seen.add(pig.id)
            unique.append(pig)
    return unique
