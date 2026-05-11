"""Mokuhanga print-order solver.

Sort impressions (block x pigment tuples) by OKLab luminance ascending so that
lighter pigments print first and darker ones last, mirroring traditional
Japanese woodblock (mokuhanga) practice. Ties on luminance are broken by
ascending coverage so that smaller plates lay down before larger ones.

Public API
----------
- ``Impression``: frozen dataclass describing one printed pass.
- ``compute_luminance_oklab(rgb)``: sRGB triplet -> OKLab L scalar.
- ``order_impressions(block_assignments, pigment_meta, direction)``: build the
  ordered list of impressions ready for the press.

Dependencies: numpy + colour-science only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import colour
import numpy as np


__all__ = [
    "Impression",
    "compute_luminance_oklab",
    "order_impressions",
]


_DIRECTION_LIGHT_TO_DARK = "light_to_dark"
_DIRECTION_DARK_TO_LIGHT = "dark_to_light"
_VALID_DIRECTIONS = frozenset({_DIRECTION_LIGHT_TO_DARK, _DIRECTION_DARK_TO_LIGHT})


@dataclass(frozen=True, slots=True)
class Impression:
    """One printed impression: a single block inked with a single pigment.

    Attributes:
        step: 1-indexed position in the print sequence.
        block_id: Carved block identifier this impression uses.
        pigment_id: Stable pigment key (e.g. ``"sumi-black"``).
        pigment_hex: ``"#rrggbb"`` form of the pigment color.
        coverage_pct: Fraction of the print area covered by this pigment
            expressed as a percentage in ``[0, 100]``.
        luminance_okL: OKLab L value of the pigment in ``[0, 1]``.
    """

    step: int
    block_id: int
    pigment_id: str
    pigment_hex: str
    coverage_pct: float
    luminance_okL: float


def compute_luminance_oklab(rgb: tuple[int, int, int] | np.ndarray) -> float:
    """Return the OKLab L channel for an sRGB color.

    Accepts an iterable of three components. Integer 0-255 inputs are
    auto-normalised to 0-1 floats before conversion. Returns a scalar even
    when handed a 2D array (in which case the mean L is returned).

    Args:
        rgb: sRGB color as ``(r, g, b)`` ints in ``[0, 255]`` or floats in
            ``[0, 1]``. May also be a numpy array of shape ``(3,)`` or
            ``(..., 3)``.

    Returns:
        OKLab L value as a Python ``float``.
    """
    rgb_f = np.asarray(rgb, dtype=np.float64)
    if rgb_f.size == 0:
        raise ValueError("rgb must contain at least one color")
    if rgb_f.shape[-1] != 3:
        raise ValueError(f"rgb last dim must be 3, got shape {rgb_f.shape}")
    if float(rgb_f.max()) > 1.0:
        rgb_f = rgb_f / 255.0
    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    lab = colour.convert(rgb_f, "sRGB", "Oklab")
    lab_arr = np.asarray(lab, dtype=np.float64)
    if lab_arr.ndim == 1:
        return float(lab_arr[0])
    return float(lab_arr[..., 0].mean())


def _hex_from_rgb(rgb: Sequence[int] | np.ndarray) -> str:
    """Render an sRGB triple as ``#rrggbb``."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"rgb must have shape (3,), got {arr.shape}")
    if float(arr.max()) <= 1.0:
        arr = arr * 255.0
    clipped = np.clip(np.rint(arr), 0, 255).astype(int)
    return "#{0:02x}{1:02x}{2:02x}".format(int(clipped[0]), int(clipped[1]), int(clipped[2]))


def _normalize_hex(value: str) -> str:
    """Lowercase a ``#rrggbb`` (or ``rrggbb``) hex into canonical form."""
    s = value.strip().lower()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) != 7:
        raise ValueError(f"hex must be #rrggbb, got {value!r}")
    return s


def order_impressions(
    block_assignments: Mapping[str, int],
    pigment_meta: Mapping[str, Mapping[str, object]],
    direction: str = "light_to_dark",
) -> list[Impression]:
    """Order pigment impressions for printing.

    Each pigment in ``block_assignments`` becomes one :class:`Impression`.
    Impressions are sorted by OKLab luminance ascending (light first) with
    ascending coverage as the tiebreaker. Set ``direction='dark_to_light'``
    to reverse the order.

    Args:
        block_assignments: Mapping of ``pigment_id`` to ``block_id``.
        pigment_meta: Mapping of ``pigment_id`` to a dict with keys
            ``hex`` (``#rrggbb``), ``rgb`` (sRGB triple, ints or floats),
            and ``coverage_pct`` (float in ``[0, 100]``).
        direction: ``"light_to_dark"`` (default) or ``"dark_to_light"``.

    Returns:
        Ordered list of :class:`Impression`, with ``step`` numbered
        ``1..M`` sequentially. Empty input yields ``[]``.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        )
    if not block_assignments:
        return []

    rows: list[tuple[float, float, str, int, str]] = []
    for pigment_id, block_id in block_assignments.items():
        if pigment_id not in pigment_meta:
            raise KeyError(f"pigment_meta missing entry for {pigment_id!r}")
        meta = pigment_meta[pigment_id]

        rgb_raw = meta.get("rgb")
        if rgb_raw is None:
            raise KeyError(f"pigment_meta[{pigment_id!r}] missing 'rgb'")
        luminance = compute_luminance_oklab(np.asarray(rgb_raw))  # type: ignore[arg-type]

        hex_value = meta.get("hex")
        pigment_hex = (
            _normalize_hex(str(hex_value))
            if hex_value is not None
            else _hex_from_rgb(np.asarray(rgb_raw))  # type: ignore[arg-type]
        )

        coverage_raw = meta.get("coverage_pct", 0.0)
        coverage_pct = float(coverage_raw)  # type: ignore[arg-type]

        rows.append((luminance, coverage_pct, pigment_id, int(block_id), pigment_hex))

    # mokuhanga light→dark: highest luminance first (white before sumi).
    # Coverage tiebreak: smaller coverage first regardless of direction.
    # So we negate the luminance key for light_to_dark, leave coverage ascending.
    if direction == _DIRECTION_LIGHT_TO_DARK:
        rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    else:
        rows.sort(key=lambda r: (r[0], r[1], r[2]))

    return [
        Impression(
            step=idx + 1,
            block_id=block_id,
            pigment_id=pigment_id,
            pigment_hex=pigment_hex,
            coverage_pct=coverage_pct,
            luminance_okL=luminance,
        )
        for idx, (luminance, coverage_pct, pigment_id, block_id, pigment_hex) in enumerate(rows)
    ]
