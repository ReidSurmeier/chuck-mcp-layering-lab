"""pigment_library.py — Emma pigment-library loader.

Loads `chuck_mcp_v2/pigment_library_emma.yaml` and exposes:

    PigmentEntry           — dataclass per pigment
    load_pigment_library() — YAML -> dict[name, PigmentEntry]
    family_to_pigments()   — index by 7-family taxonomy
    pigment_lab()          — quick lookup (L*, a*, b*)
    EMMA_PIGMENT_LIBRARY_PATH — canonical Path to YAML

V1 deliberately stores catalog-approximation Lab values
(`uncalibrated_v1: true`). V2 will replace with measured swatches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

# Repo root: <repo>/chuck_mcp_v2/pigment_library_emma.yaml
_REPO_ROOT = Path(__file__).resolve().parents[3]
EMMA_PIGMENT_LIBRARY_PATH: Path = (
    _REPO_ROOT / "chuck_mcp_v2" / "pigment_library_emma.yaml"
)

# Canonical 7 families that the rule classifier emits.
PIGMENT_FAMILIES: tuple[str, ...] = (
    "light_yellow",
    "pale_pink",
    "pale_orange",
    "pale_red",
    "pale_blue",
    "pale_green",
    "warm_grey",
)


@dataclass
class PigmentEntry:
    """One concrete pigment in Reid's Emma inventory.

    Lab values are D50 sRGB->Lab catalog approximations; for V1 they ALL carry
    `uncalibrated_v1=True`. opacity_curve is a piecewise-linear (dilution,
    opacity) table; opacity_at_dilution() does the interpolation.
    """

    name: str
    family: str
    lab_values: tuple[float, float, float]
    opacity_curve: list[tuple[float, float]]
    source_note: str
    uncalibrated_v1: bool
    ci_pigment: str = ""
    notes: str = ""

    def opacity_at_dilution(self, dilution: float) -> float:
        """Linearly interpolate opacity at the given dilution in [0, 1].

        opacity_curve is a sorted-by-dilution list of (dilution, opacity)
        anchor points. Edge values clamp to first/last entry.
        """
        if not self.opacity_curve:
            return 0.0
        pts = sorted(self.opacity_curve, key=lambda kv: kv[0])
        x = float(dilution)
        if x <= pts[0][0]:
            return float(pts[0][1])
        if x >= pts[-1][0]:
            return float(pts[-1][1])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return float(y0)
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(pts[-1][1])


def _coerce_pigment_entry(name: str, raw: dict) -> PigmentEntry:
    """Validate + convert one YAML entry dict into a PigmentEntry."""
    required = {"family", "lab_values", "opacity_curve",
                "source_note", "uncalibrated_v1"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"pigment {name!r} missing required keys: {sorted(missing)}"
        )

    family = str(raw["family"])
    if family not in PIGMENT_FAMILIES:
        raise ValueError(
            f"pigment {name!r} has unknown family {family!r}; "
            f"valid families = {PIGMENT_FAMILIES}"
        )

    lab = raw["lab_values"]
    if len(lab) != 3:
        raise ValueError(
            f"pigment {name!r} lab_values must be [L, a, b], got {lab!r}"
        )
    lab_t = (float(lab[0]), float(lab[1]), float(lab[2]))

    curve_raw = raw["opacity_curve"]
    if not isinstance(curve_raw, Iterable):
        raise ValueError(
            f"pigment {name!r} opacity_curve must be iterable of (dilution, opacity)"
        )
    curve: list[tuple[float, float]] = []
    for pt in curve_raw:
        if len(pt) != 2:
            raise ValueError(
                f"pigment {name!r} opacity_curve point {pt!r} must be a (dilution, opacity) pair"
            )
        curve.append((float(pt[0]), float(pt[1])))
    if len(curve) < 2:
        raise ValueError(
            f"pigment {name!r} opacity_curve must have ≥2 points"
        )

    return PigmentEntry(
        name=name,
        family=family,
        lab_values=lab_t,
        opacity_curve=curve,
        source_note=str(raw["source_note"]),
        uncalibrated_v1=bool(raw["uncalibrated_v1"]),
        ci_pigment=str(raw.get("ci_pigment", "")),
        notes=str(raw.get("notes", "")),
    )


def load_pigment_library(
    path: Path | str = EMMA_PIGMENT_LIBRARY_PATH,
) -> dict[str, PigmentEntry]:
    """Parse the YAML library and return name -> PigmentEntry."""
    p = Path(path)
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"pigment library {p} did not parse as a dict")
    out: dict[str, PigmentEntry] = {}
    for name, entry in raw.items():
        if name == "meta":
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"pigment {name!r}: entry must be a dict, got {type(entry).__name__}"
            )
        out[name] = _coerce_pigment_entry(name, entry)
    return out


def family_to_pigments(
    library: dict[str, PigmentEntry],
) -> dict[str, list[PigmentEntry]]:
    """Group pigments by their family."""
    out: dict[str, list[PigmentEntry]] = {}
    for entry in library.values():
        out.setdefault(entry.family, []).append(entry)
    return out


def pigment_lab(library: dict[str, PigmentEntry], name: str) -> tuple[float, float, float]:
    """Return (L, a, b) for pigment `name`. Raises KeyError if unknown."""
    return library[name].lab_values
