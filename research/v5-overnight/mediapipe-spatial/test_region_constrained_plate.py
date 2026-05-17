"""TDD Cycle 3 — Plate cell_zone_ids respect their face_region_constraint.

A plate with role 'underlayer_light' assigned to ['left_cheek', 'right_cheek']
must contain ONLY cells whose primary region is left_cheek or right_cheek.
Cells from 'hair' or 'background' are REMOVED.

API under test:
    region_constrained_plate.filter_plate_cells(plate, cell_to_region, allowed_regions)
    region_constrained_plate.validate_plate_against_regions(plate, cell_to_region)

Acceptance:
    - filter returns a NEW plate (immutable on the input) whose cell_zone_ids
      are a subset of the original, all in allowed_regions
    - validator returns (ok, violations) listing every cell that violates
      the region constraint
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

# Make sure chuck_mcp_v2 is importable (its types.Plate is the canonical schema).
_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from chuck_mcp_v2.types import Plate


def _make_plate(block_id: int, cell_ids: list[int], region_constraint=None) -> Plate:
    p = Plate(
        block_id=block_id,
        cell_zone_ids=list(cell_ids),
        role="underlayer_light",
        pigment_family="light_yellow",
        region_label="skin_warm_support",
        rationale="test plate",
    )
    # Pin the face_region_constraint as an attribute so the test mirrors how
    # plan_emma will tag it after the integration patch.
    if region_constraint is not None:
        p.face_region_constraint = list(region_constraint)
    return p


def test_filter_plate_drops_cells_outside_constraint() -> None:
    import region_constrained_plate as rcp
    # cells 1..6 spread across cheeks + hair + background
    cell_to_region = {
        1: "left_cheek", 2: "left_cheek",
        3: "right_cheek",
        4: "hair",
        5: "background",
        6: "forehead",
    }
    plate = _make_plate(
        block_id=7, cell_ids=[1, 2, 3, 4, 5, 6],
        region_constraint=["left_cheek", "right_cheek"],
    )

    filtered = rcp.filter_plate_cells(plate, cell_to_region,
                                      allowed_regions=plate.face_region_constraint)

    assert filtered.cell_zone_ids == [1, 2, 3], (
        f"plate must drop hair/background/forehead cells when constraint is "
        f"[left_cheek, right_cheek]; got {filtered.cell_zone_ids}"
    )
    # Input must NOT be mutated
    assert plate.cell_zone_ids == [1, 2, 3, 4, 5, 6], "input plate was mutated"
    # role, pigment, region_label preserved
    assert filtered.role == "underlayer_light"
    assert filtered.region_label == "skin_warm_support"


def test_filter_plate_no_constraint_is_passthrough() -> None:
    """If a plate has no face_region_constraint, the filter is a no-op."""
    import region_constrained_plate as rcp
    cell_to_region = {1: "hair", 2: "background", 3: "left_cheek"}
    plate = _make_plate(block_id=2, cell_ids=[1, 2, 3], region_constraint=None)
    filtered = rcp.filter_plate_cells(plate, cell_to_region, allowed_regions=None)
    assert filtered.cell_zone_ids == [1, 2, 3]


def test_validate_plate_flags_constraint_violations() -> None:
    import region_constrained_plate as rcp
    cell_to_region = {
        10: "left_cheek",
        11: "hair",       # VIOLATION on a cheek-constrained plate
        12: "left_cheek",
        13: "background", # VIOLATION
    }
    plate = _make_plate(
        block_id=5, cell_ids=[10, 11, 12, 13],
        region_constraint=["left_cheek"],
    )
    ok, violations = rcp.validate_plate_against_regions(
        plate, cell_to_region
    )
    assert not ok
    # Violations is a list of (cell_id, region_name)
    violation_cells = {c for c, _ in violations}
    assert violation_cells == {11, 13}, (
        f"expected violations on cells 11+13, got {violation_cells}"
    )


def test_validate_plate_passes_when_all_cells_in_constraint() -> None:
    import region_constrained_plate as rcp
    cell_to_region = {1: "left_cheek", 2: "left_cheek"}
    plate = _make_plate(block_id=1, cell_ids=[1, 2],
                        region_constraint=["left_cheek", "right_cheek"])
    ok, violations = rcp.validate_plate_against_regions(plate, cell_to_region)
    assert ok
    assert violations == []


def test_underlayer_plate_only_contains_cells_in_assigned_region() -> None:
    """End-to-end assertion mirroring the mission statement: a plate with
    role 'underlayer_light' and assigned region 'cheek+temple' MUST NOT
    contain any cells whose primary region is 'hair' or 'background'.
    """
    import region_constrained_plate as rcp
    cell_to_region = {
        100: "left_cheek",   101: "left_cheek",
        102: "right_cheek",  103: "left_temple",
        104: "right_temple",
        200: "hair",         201: "hair",
        300: "background",   301: "background",
        400: "lips",
    }
    plate = _make_plate(
        block_id=9,
        cell_ids=list(cell_to_region.keys()),
        region_constraint=[
            "left_cheek", "right_cheek", "left_temple", "right_temple",
        ],
    )
    filtered = rcp.filter_plate_cells(
        plate, cell_to_region,
        allowed_regions=plate.face_region_constraint,
    )
    for cid in filtered.cell_zone_ids:
        assert cell_to_region[cid] in plate.face_region_constraint, (
            f"cell {cid} on cheek+temple plate has region "
            f"{cell_to_region[cid]} — must be filtered out"
        )
    # And critically: hair + background + lips must all be gone
    forbidden = {200, 201, 300, 301, 400}
    assert not (set(filtered.cell_zone_ids) & forbidden), (
        f"forbidden cells leaked through: "
        f"{set(filtered.cell_zone_ids) & forbidden}"
    )
