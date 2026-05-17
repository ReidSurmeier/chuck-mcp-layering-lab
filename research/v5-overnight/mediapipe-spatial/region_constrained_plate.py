"""chuck-mcp v5 — Region-Constrained Plate Filtering

Given a Plate (chuck_mcp_v2.types.Plate) and a cell->region map from
`merge_cells_with_regions.assign_cells_to_regions`, drop cells that violate
the plate's `face_region_constraint`.

Why: an underlayer_light plate roled to "skin_warm_support" with constraint
["left_cheek", "right_cheek", "left_temple", "right_temple"] must never ink
cells whose primary region is "hair" or "background". Without this filter,
the auto-partitioner can put hair-colored cells onto a skin plate, producing
a structurally wrong woodblock plan.

Public API:
    filter_plate_cells(plate, cell_to_region, allowed_regions=None) -> Plate
    validate_plate_against_regions(plate, cell_to_region) -> (ok, violations)

`filter_plate_cells` returns a NEW Plate; the input is never mutated.
"""
from __future__ import annotations

import copy
from typing import Iterable, Optional

from chuck_mcp_v2.types import Plate


def _resolve_constraint(plate: Plate, allowed_regions: Optional[Iterable[str]]
                        ) -> Optional[set[str]]:
    """Pick the active region constraint set, or None if the plate is
    unconstrained (passthrough)."""
    if allowed_regions is not None:
        allowed = set(allowed_regions)
        return allowed if allowed else None
    attr = getattr(plate, "face_region_constraint", None)
    if attr:
        return set(attr)
    return None


def filter_plate_cells(
    plate: Plate,
    cell_to_region: dict[int, str],
    allowed_regions: Optional[Iterable[str]] = None,
) -> Plate:
    """Return a copy of `plate` with cells filtered to `allowed_regions`.

    If `allowed_regions` is None and `plate.face_region_constraint` is also
    None/empty, the plate is returned as-is (no filtering).

    If `allowed_regions` is provided, it overrides any
    `plate.face_region_constraint` value on the input.

    Cells missing from `cell_to_region` are treated as violations and removed.
    """
    allowed = _resolve_constraint(plate, allowed_regions)

    if allowed is None:
        # Unconstrained: shallow copy to preserve no-mutation contract.
        return _clone_plate(plate)

    kept: list[int] = []
    for cid in plate.cell_zone_ids:
        region = cell_to_region.get(int(cid))
        if region in allowed:
            kept.append(int(cid))

    new_plate = _clone_plate(plate)
    new_plate.cell_zone_ids = sorted(set(kept))
    # Persist the active constraint on the clone so downstream code (and
    # serializers) can see exactly which regions this plate was bounded to.
    new_plate.face_region_constraint = sorted(allowed)
    return new_plate


def validate_plate_against_regions(
    plate: Plate,
    cell_to_region: dict[int, str],
) -> tuple[bool, list[tuple[int, str]]]:
    """Check whether `plate` honors its `face_region_constraint`.

    Returns (ok, violations) where violations is a list of (cell_id, region)
    pairs for every cell on the plate whose primary region is NOT in the
    constraint.

    Plates with no constraint always pass (ok=True, violations=[]).
    """
    allowed = _resolve_constraint(plate, None)
    if allowed is None:
        return True, []
    violations: list[tuple[int, str]] = []
    for cid in plate.cell_zone_ids:
        region = cell_to_region.get(int(cid), "<unknown>")
        if region not in allowed:
            violations.append((int(cid), region))
    return (not violations), violations


# ----------------------------------------------------------------------------
# Internal: Plate cloning
# ----------------------------------------------------------------------------

def _clone_plate(plate: Plate) -> Plate:
    """Return a Plate with the same content as `plate` but no shared mutable
    state in `cell_zone_ids`. We deep-copy through to_dict isn't reversible
    on the production schema, so just rebuild via the canonical constructor.
    """
    # Build a brand-new Plate via the keyword constructor; copy through every
    # known attribute except `face_region_constraint` (handled separately).
    new = Plate(
        block_id=int(plate.block_id),
        cell_zone_ids=list(plate.cell_zone_ids),
        role=plate.role,
        pigment_family=plate.pigment_family,
        region_label=plate.region_label,
        rationale=plate.rationale,
        provenance=plate.provenance,
        mirror=plate.mirror,
    )
    # Carry over pulls (shallow list copy — PullSpec instances are immutable
    # in practice and the v3 builder hands us fresh ones).
    new.pulls = list(plate.pulls)
    # Forward optional metadata (region_label, pigment_id, etc.) only when
    # set on the source to avoid clobbering defaults.
    for attr in (
        "pigment_id", "dilution", "pigment_weights", "area_px", "repair_stats",
        "pigment_choices", "initial_opacity", "initial_dilution",
    ):
        v = getattr(plate, attr, None)
        if v is not None:
            setattr(new, attr, copy.copy(v))
    constraint = getattr(plate, "face_region_constraint", None)
    if constraint is not None:
        new.face_region_constraint = list(constraint)
    return new


__all__ = [
    "filter_plate_cells",
    "validate_plate_against_regions",
]
