"""ProductionPlan dataclass + PullSpec + PlateSpec.

The ProductionPlan is the FIRST-STAGE output of the v4 alternating optimizer.
It carries:
    - adaptive plate count (24-30 prior for Emma)
    - multi-pull-per-block schedule as first-class variables
    - block_id <-> cell_zone assignment
    - pull order (1..~132 for Emma)

It is consumed downstream by:
    - JAX continuous solver (optimizes opacity/dilution/color/load per PullSpec)
    - morphology repair (fixes printability per plate)
    - validators (plate_not_composite, role_purity, jigsaw_separation, ...)

Domain mirrors `docs/v2-design-locked-2026-05-16.md` §Domain objects and
`research/v3-construction/cell-zone-renderer/cz_types.py`, but specialized
for the BEFORE-JAX-solve representation.

α-maps NEVER appear here. They are renderer internals (per Q26).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from chuck_mcp_v2.types import Plate as PlateSpec
from chuck_mcp_v2.types import ROLES, Role


# ----------------------------------------------------------------------------
# PullSpec — one impression of one plate at one absolute print step.
# ----------------------------------------------------------------------------
@dataclass
class PullSpec:
    """One pull (impression) variable for the JAX solver.

    Variables marked CONTINUOUS are what JAX optimizes; the rest are
    structural (set by this proposal stage and held FIXED during JAX solve).
    """

    pull_id: int                       # 1..N global pull index
    block_id: int                      # which physical plate
    pigment_id: str                    # from Reid's inventory; may be a premix recipe id
    opacity: float                     # CONTINUOUS  ∈ [0,1]   K-M ink load
    dilution: float                    # CONTINUOUS  ∈ [0,1]   water cut (1.0 = full pigment)
    order_step: int                    # 1..N absolute print order
    pass_index: int                    # 1..5 which pull-of-this-block (1 = first)
    role: Role                         # inherited from PlateSpec
    mask_subset: Optional[list[int]] = None
    """If None: pull inks ALL cell_zones on its parent plate.
    If a list of zone_ids: pull inks only that subset (sub-pull masking).
    Used for graduated opacity within a block (e.g. mid-build over shadow
    side only) WITHOUT inventing a new physical plate.
    """

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ----------------------------------------------------------------------------
# ProductionPlan — the full first-stage output.
# ----------------------------------------------------------------------------
@dataclass
class ProductionPlan:
    """The full production layout BEFORE JAX solves anything.

    Invariants (enforced by ``validate()``):
        I1. plate_count ∈ [20, 35]
        I2. every plate has a non-empty cell_zone_ids list
        I3. every plate has at least one PullSpec (>= 1 pull per block)
        I4. PullSpec.pass_index is contiguous 1..plate.pull_count per plate
        I5. NO orphan cells: every cell in the cell_graph is on at least one plate
        I6. NO zone-overlap on same block: cell_zone_ids on one plate are a set
            (no duplicates within a plate)
        I7. role membership ⊆ ROLES
        I8. pull_id is unique across the plan
        I9. order_step is unique across the plan and forms a contiguous range
    """

    plan_id: str
    plates: list[PlateSpec]
    total_pulls: int
    cell_count: int                          # total cells in input cell graph
    image_shape: tuple[int, int]             # (H, W)
    meta: dict = field(default_factory=dict)

    @property
    def plate_count(self) -> int:
        return len(self.plates)

    def all_pulls(self) -> list[PullSpec]:
        out: list[PullSpec] = []
        for p in self.plates:
            out.extend(p.pulls)
        return sorted(out, key=lambda x: x.order_step)

    def assigned_cell_ids(self) -> set[int]:
        s: set[int] = set()
        for p in self.plates:
            s.update(p.cell_zone_ids)
        return s

    def validate(self, all_cell_ids: Optional[set[int]] = None) -> tuple[bool, list[str]]:
        """Run all invariants. Returns (ok, list_of_failure_messages).

        If ``all_cell_ids`` is provided, I5 (no-orphan-cells) is checked
        against it. Otherwise I5 is skipped.
        """
        errors: list[str] = []

        # I1
        if not (20 <= self.plate_count <= 35):
            errors.append(
                f"I1: plate_count={self.plate_count} outside [20, 35]"
            )

        # I7
        pull_ids: list[int] = []
        order_steps: list[int] = []
        for p in self.plates:
            if p.role not in ROLES:
                errors.append(f"I7: plate {p.block_id} bad role={p.role!r}")

            # I2
            if not p.cell_zone_ids:
                errors.append(f"I2: plate {p.block_id} has empty cell_zone_ids")

            # I3
            if not p.pulls:
                errors.append(f"I3: plate {p.block_id} has no pulls")

            # I4
            for k, pull in enumerate(sorted(p.pulls, key=lambda x: x.pass_index), 1):
                if pull.pass_index != k:
                    errors.append(
                        f"I4: plate {p.block_id} pass_index "
                        f"non-contiguous: expected {k}, got {pull.pass_index}"
                    )

            # I6
            if len(set(p.cell_zone_ids)) != len(p.cell_zone_ids):
                errors.append(
                    f"I6: plate {p.block_id} has duplicate cell_zone_ids"
                )

            for pull in p.pulls:
                pull_ids.append(pull.pull_id)
                order_steps.append(pull.order_step)

        # I8 — pull_id uniqueness
        if len(set(pull_ids)) != len(pull_ids):
            dup = [pid for pid in pull_ids if pull_ids.count(pid) > 1]
            errors.append(f"I8: duplicate pull_ids={sorted(set(dup))}")

        # I9 — order_step uniqueness + contiguous
        if len(set(order_steps)) != len(order_steps):
            dup = [s for s in order_steps if order_steps.count(s) > 1]
            errors.append(f"I9: duplicate order_steps={sorted(set(dup))}")
        if order_steps:
            order_steps_sorted = sorted(order_steps)
            expected = list(range(1, len(order_steps) + 1))
            if order_steps_sorted != expected:
                errors.append(
                    f"I9: order_steps not contiguous 1..N "
                    f"(got [{order_steps_sorted[0]}..{order_steps_sorted[-1]}])"
                )

        # Sanity: total_pulls matches
        if self.total_pulls != len(pull_ids):
            errors.append(
                f"total_pulls={self.total_pulls} != sum(plate.pulls)={len(pull_ids)}"
            )

        # I5 — orphan cells
        if all_cell_ids is not None:
            assigned = self.assigned_cell_ids()
            orphans = all_cell_ids - assigned
            if orphans:
                # Show only first 5 in the message
                preview = sorted(orphans)[:5]
                errors.append(
                    f"I5: {len(orphans)} orphan cells (e.g. {preview})"
                )

        return (len(errors) == 0, errors)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "plate_count": self.plate_count,
            "total_pulls": self.total_pulls,
            "cell_count": self.cell_count,
            "image_shape": list(self.image_shape),
            "plates": [p.to_dict() for p in self.plates],
            "meta": self.meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def role_distribution(self) -> dict[str, int]:
        out: dict[str, int] = {r: 0 for r in ROLES}
        for p in self.plates:
            out[p.role] += 1
        return out
