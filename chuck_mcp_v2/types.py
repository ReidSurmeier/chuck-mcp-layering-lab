"""Canonical domain types shared by the Chuck MCP research modules.

The v4 research build originally carried separate Plate dataclasses for the
production plan, continuous objective, hybrid optimizer, and v3 renderer.
This module provides one superset Plate while preserving those modules'
legacy construction patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is present in supported envs
    np = None  # type: ignore


Role = Literal["underlayer_light", "local_chroma", "regional_mass", "key_detail"]

ROLES: tuple[Role, ...] = (
    "underlayer_light",
    "local_chroma",
    "regional_mass",
    "key_detail",
)
ROLE_FAMILIES: tuple[Role, ...] = ROLES


@dataclass(init=False)
class Plate:
    """Canonical physical-block / solver-plate schema.

    The fields are a union of the existing research representations:
    production-solver ``PlateSpec``, plate-objective ``Plate``,
    hybrid-optimizer ``FrozenPlate``/``SolvedPlate``, and renderer ``Plate``.
    Optional fields remain unset until the stage that owns them populates them.
    """

    block_id: int
    cell_zone_ids: list[int]
    role: Role | str
    pigment_family: str
    pulls: list[Any]
    region_label: Optional[str]
    mirror: bool
    rationale: str
    provenance: str

    mask: Any
    pigment_lab: Any
    opacity: Any
    pass_index: int

    pigment_id: str
    dilution: float
    pigment_weights: dict[str, float]
    inked_mask: Any
    area_px: int
    repair_stats: dict[str, Any]

    pigment_choices: list[tuple[str, tuple[float, float, float]]]
    initial_opacity: float
    initial_dilution: float

    cell_zones: list[Any]
    pigment_color: tuple[float, float, float] | None
    pigment_name: str

    schema_kind: str = field(default="plate", repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._set_defaults()
        self._apply_legacy_positional_args(args)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f"Plate got an unexpected keyword argument {key!r}")
            setattr(self, key, value)
        self._normalize()

    def _set_defaults(self) -> None:
        self.block_id = 0
        self.cell_zone_ids = []
        self.role = "local_chroma"
        self.pigment_family = ""
        self.pulls = []
        self.region_label = None
        self.mirror = True
        self.rationale = ""
        self.provenance = "algorithm"

        self.mask = None
        self.pigment_lab = None
        self.opacity = 1.0
        self.pass_index = 0

        self.pigment_id = ""
        self.dilution = 0.0
        self.pigment_weights = {}
        self.inked_mask = None
        self.area_px = 0
        self.repair_stats = {}

        self.pigment_choices = []
        self.initial_opacity = 0.5
        self.initial_dilution = 0.3

        self.cell_zones = []
        self.pigment_color = None
        self.pigment_name = ""

        self.schema_kind = "plate"

    def _apply_legacy_positional_args(self, args: tuple[Any, ...]) -> None:
        if not args:
            return
        if len(args) > 9:
            raise TypeError(f"Plate expected at most 9 positional args, got {len(args)}")

        self.block_id = int(args[0])

        # production-solver legacy shape:
        # PlateSpec(block_id, cell_zone_ids, role, pigment_family, ...)
        if len(args) >= 4 and isinstance(args[2], str):
            self.cell_zone_ids = list(args[1])
            self.role = args[2]
            self.pigment_family = str(args[3])
            if len(args) >= 5:
                self.pulls = list(args[4])
            if len(args) >= 6:
                self.region_label = args[5]
            if len(args) >= 7:
                self.mirror = bool(args[6])
            if len(args) >= 8:
                self.rationale = str(args[7])
            if len(args) >= 9:
                self.provenance = str(args[8])
            self.schema_kind = "production"
            return

        # renderer legacy shape:
        # Plate(block_id, cell_zones, pigment_color, pigment_name, opacity, role, ...)
        if len(args) >= 6 and isinstance(args[3], str):
            self.cell_zones = list(args[1])
            self.pigment_color = args[2]
            self.pigment_name = str(args[3])
            self.opacity = args[4]
            self.role = args[5]
            if len(args) >= 7:
                self.pass_index = int(args[6])
            if len(args) >= 8:
                self.mirror = bool(args[7])
            self.schema_kind = "renderer"
            return

        # plate-objective legacy shape:
        # Plate(block_id, mask, pigment_lab, opacity, role, cell_zone_ids=..., ...)
        if len(args) >= 5 and not isinstance(args[2], str):
            self.mask = args[1]
            self.pigment_lab = args[2]
            self.opacity = args[3]
            self.role = args[4]
            self.schema_kind = "objective"
            return

        raise TypeError("Plate positional arguments do not match a known schema")

    def _normalize(self) -> None:
        if self.cell_zone_ids is None:
            self.cell_zone_ids = []
        elif isinstance(self.cell_zone_ids, tuple):
            self.cell_zone_ids = list(self.cell_zone_ids)

        if self.pulls is None:
            self.pulls = []
        if self.pigment_weights is None:
            self.pigment_weights = {}
        if self.repair_stats is None:
            self.repair_stats = {}
        if self.pigment_choices is None:
            self.pigment_choices = []
        if self.cell_zones is None:
            self.cell_zones = []

        if self.schema_kind == "plate":
            if self.inked_mask is not None or self.pigment_id or self.pigment_weights:
                self.schema_kind = "solved"
            elif self.pigment_choices:
                self.schema_kind = "frozen"
            elif self.mask is not None or self.pigment_lab is not None:
                self.schema_kind = "objective"
            elif self.cell_zones or self.pigment_color is not None or self.pigment_name:
                self.schema_kind = "renderer"
            elif self.pigment_family or self.pulls:
                self.schema_kind = "production"

        if self.schema_kind == "renderer" and self.pass_index == 0:
            self.pass_index = 1

    def add_pull(self, pull: Any) -> None:
        if pull.block_id != self.block_id:
            raise ValueError(
                f"PullSpec.block_id={pull.block_id} doesn't match plate.block_id={self.block_id}"
            )
        if pull.role != self.role:
            raise ValueError(
                f"PullSpec.role={pull.role!r} doesn't match plate.role={self.role!r}"
            )
        self.pulls.append(pull)
        if self.schema_kind == "plate":
            self.schema_kind = "production"

    @property
    def pull_count(self) -> int:
        return len(self.pulls)

    @property
    def plate_id(self) -> int:
        return self.block_id

    def to_dict(self, include_mask: bool = False) -> dict[str, Any]:
        if self._is_solved_like():
            return self._to_solved_dict(include_mask=include_mask)
        return self._to_production_dict()

    def _is_solved_like(self) -> bool:
        return (
            self.schema_kind == "solved"
            or self.inked_mask is not None
            or bool(self.pigment_id)
            or bool(self.pigment_weights)
        )

    def _to_production_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "block_id": int(self.block_id),
            "cell_zone_ids": [int(x) for x in self.cell_zone_ids],
            "role": self.role,
            "pigment_family": self.pigment_family,
            "pulls": [
                p.to_dict() if hasattr(p, "to_dict") else p
                for p in self.pulls
            ],
            "region_label": self.region_label,
            "mirror": bool(self.mirror),
            "rationale": self.rationale,
            "provenance": self.provenance,
        }
        # v5 mediapipe spatial constraint: emit when present so production
        # plan JSON carries which face regions the plate is bounded to.
        constraint = getattr(self, "face_region_constraint", None)
        if constraint:
            d["face_region_constraint"] = [str(x) for x in constraint]
        return d

    def _to_solved_dict(self, include_mask: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "block_id": int(self.block_id),
            "cell_zone_ids": [int(x) for x in self.cell_zone_ids],
            "pigment_id": str(self.pigment_id),
            "opacity": float(self.opacity),
            "dilution": float(self.dilution),
            "role": str(self.role),
            "pass_index": int(self.pass_index),
            "pigment_weights": {k: float(v) for k, v in self.pigment_weights.items()},
            "area_px": int(self.area_px),
            "repair_stats": _jsonify(self.repair_stats),
            "mirror": bool(self.mirror),
        }
        if include_mask and self.inked_mask is not None:
            d["inked_mask_shape"] = list(self.inked_mask.shape)
            d["inked_mask_dtype"] = str(self.inked_mask.dtype)
        return d


def _jsonify(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    if np is not None:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    return o


__all__ = ["Plate", "Role", "ROLES", "ROLE_FAMILIES"]
