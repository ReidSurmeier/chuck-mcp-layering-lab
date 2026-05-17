"""Minimal local domain types for cell-zone renderers.

These mirror the v23/v3 domain types in `docs/v2-design-locked-2026-05-16.md`
without importing anything from the backend (research folder is standalone).

The four canonical objects:

- ``CellZone``: an SNIC superpixel polygon with a target color and an
  assignment to a plate. This is the *primary* representation.
- ``Plate``: a physical block (1..27). Holds a list of cell zones inked on
  this block, a single pigment color, opacity and role.
- ``Pull``: one impression of a plate at one print order step.
- ``ProofState``: cumulative print after N pulls; used for 8-up sheets.

α-maps NEVER appear here. They are produced inside ``render_pull`` as a
private renderer detail (per Q26).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon


Role = Literal["underlayer_light", "local_chroma", "regional_mass", "key_detail"]


@dataclass
class CellZone:
    """One SNIC superpixel polygon assigned to one plate."""

    zone_id: int
    polygon: Polygon                 # shapely polygon in IMAGE pixel coords (un-mirrored)
    target_rgb: tuple[float, float, float]  # 0..1, the target color for QA/scoring
    plate_id: int                    # which plate this zone belongs to
    centroid_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.centroid_xy is None:
            c = self.polygon.centroid
            self.centroid_xy = (float(c.x), float(c.y))


@dataclass
class Plate:
    """A physical block carved with a number of cell-zones inked one color."""

    block_id: int                          # 1..27
    cell_zones: list[CellZone]
    pigment_color: tuple[float, float, float]   # 0..1 RGB the plate is inked with
    pigment_name: str
    opacity: float                         # 0..1 ink load for K-M overprint
    role: Role
    pass_index: int = 1                    # 1..5 — which pull of this block
    mirror: bool = True                    # always True for SVG output


@dataclass
class Pull:
    """One impression of a plate at one absolute print step."""

    pull_id: int                           # 1..132
    plate: Plate
    order_step: int                        # absolute print order
    ink_density: float                     # 0..1


@dataclass
class ProofState:
    """Cumulative print after pulls 1..checkpoint."""

    checkpoint_id: int                     # 1..7
    pulls_so_far: list[Pull]
    rendered_image: NDArray[np.float32]    # H×W×3 in 0..1 sRGB
