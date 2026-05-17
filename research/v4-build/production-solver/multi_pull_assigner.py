"""Multi-pull-per-block assigner.

Phase 2 of `audit-response-and-reconstruction-plan-2026-05-17.md`:

    "multiple pulls per block as first-class variables: opacity, dilution,
     repeat count, and order"

For Emma-class portraits:
    - 24-30 physical blocks
    - 1-5 pulls per block
    - ~132 total pulls (matches Pace progressive-proof scan)

Pulls-per-block follow role budgets (Sultan/Shiff 2003 forensic):

    underlayer_light   1-2 pulls (the first plate often hits just once)
    local_chroma       3-5 pulls (lip/cheek can compound 4-5 layers)
    regional_mass      4-5 pulls (hair/background carry tone-stack depth)
    key_detail         2-3 pulls (eyeline/contour is dense but selective)

Mean across roles ≈ 5 pulls/block for Emma scale. With 26-27 plates that
hits the ~132 target. The exact total is solved with role-weighted
distribution + tail adjustment so the global sum lands near the target.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np

from production_plan import PullSpec, ROLES, Role  # noqa: E402


# Per-role pull budgets (min, max, default_mean)
# Targets the Pace/Shibata Emma scan: 132 pulls / 26 plates ≈ 5.08 pulls/block.
# Default means SUM to ~132 for the 6/9/8/3 role mix, then the global-sum
# adjuster nudges within ±tolerance.
#
# Math check (26 Emma plates, role mix 6/9/8/3):
#     6*3 + 9*5 + 8*5 + 3*4 = 18 + 45 + 40 + 12 = 115  (within ±17 of 132)
# Adjuster then pushes upward toward 132 by adding pulls to regional_mass
# and local_chroma blocks until the global sum is within ±tolerance.
ROLE_PULL_BUDGET: dict[Role, tuple[int, int, float]] = {
    "underlayer_light": (1, 3, 3.0),    # support pulls: 1-3 (light coat + boost)
    "local_chroma":     (3, 5, 5.0),    # cheek/lip/eye color: 3-5 layered
    "regional_mass":    (4, 5, 5.0),    # hair/background: max-stacked
    "key_detail":       (3, 5, 4.0),    # contour/eye: deep + repeats
}

# Hard global bounds
MIN_PULLS_PER_BLOCK = 1
MAX_PULLS_PER_BLOCK = 5

# Emma-class target
EMMA_TOTAL_PULLS = 132
EMMA_TOTAL_PULLS_TOL = 12      # ±12 considered "around 132"

# Role default pigment_id (pre-JAX-solve hints — JAX may rewrite)
_ROLE_DEFAULT_PIGMENT: dict[Role, str] = {
    "underlayer_light": "PY3_holbein_pale",
    "local_chroma":     "PR122_holbein_dilute",
    "regional_mass":    "PB15_holbein_dilute",
    "key_detail":       "PBk7_sumi_dense",
}

# Role default opacity & dilution starting points
_ROLE_DEFAULT_OPACITY: dict[Role, float] = {
    "underlayer_light": 0.20,
    "local_chroma":     0.50,
    "regional_mass":    0.65,
    "key_detail":       0.85,
}
_ROLE_DEFAULT_DILUTION: dict[Role, float] = {
    "underlayer_light": 0.30,
    "local_chroma":     0.55,
    "regional_mass":    0.70,
    "key_detail":       0.90,
}


# ---------------------------------------------------------------------------
# Pull-count selector
# ---------------------------------------------------------------------------
def _target_pulls_for_role(role: Role) -> int:
    lo, hi, mean = ROLE_PULL_BUDGET[role]
    # Round to nearest int, clamp
    return max(lo, min(hi, int(round(mean))))


def _adjust_to_global_target(
    per_block: dict[int, int],
    role_per_block: dict[int, Role],
    target_total: int,
    target_tol: int,
) -> dict[int, int]:
    """If sum-of-pulls drifts > tol from target, add/remove pulls
    on regional_mass and local_chroma blocks until in range.
    """
    cur = sum(per_block.values())

    def block_priority_for_add(b: int) -> int:
        # Higher number = better candidate to add a pull
        r = role_per_block[b]
        return {"regional_mass": 3, "local_chroma": 2,
                "key_detail": 1, "underlayer_light": 0}.get(r, 0)

    def block_priority_for_drop(b: int) -> int:
        # Higher number = better candidate to drop a pull
        r = role_per_block[b]
        return {"underlayer_light": 3, "key_detail": 2,
                "local_chroma": 1, "regional_mass": 0}.get(r, 0)

    iters = 0
    max_iters = 10 * max(1, len(per_block))
    while abs(cur - target_total) > target_tol and iters < max_iters:
        iters += 1
        delta = target_total - cur
        if delta > 0:
            # Need MORE pulls. Pick block where we can add and priority is high.
            candidates = [
                b for b, n in per_block.items()
                if n < ROLE_PULL_BUDGET[role_per_block[b]][1]
                and n < MAX_PULLS_PER_BLOCK
            ]
            if not candidates:
                break
            candidates.sort(key=lambda b: (-block_priority_for_add(b), per_block[b]))
            b = candidates[0]
            per_block[b] += 1
            cur += 1
        else:
            # Need FEWER pulls
            candidates = [
                b for b, n in per_block.items()
                if n > ROLE_PULL_BUDGET[role_per_block[b]][0]
                and n > MIN_PULLS_PER_BLOCK
            ]
            if not candidates:
                break
            candidates.sort(key=lambda b: (-block_priority_for_drop(b), -per_block[b]))
            b = candidates[0]
            per_block[b] -= 1
            cur -= 1
    return per_block


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def assign_pulls_per_block(
    plate_count: int,
    role_assignments: dict[int, Role],
    *,
    target_image: Optional[np.ndarray] = None,
    target_total_pulls: int = EMMA_TOTAL_PULLS,
    target_pull_tolerance: int = EMMA_TOTAL_PULLS_TOL,
    starting_order_step: int = 1,
    starting_pull_id: int = 1,
) -> dict[int, list[PullSpec]]:
    """Assign 1-5 PullSpec records per block.

    Args:
        plate_count: number of physical plates (block_ids will be 1..plate_count
            unless role_assignments uses other keys; we use role_assignments
            keys verbatim as block_ids).
        role_assignments: mapping block_id -> Role.
        target_image: unused at this stage but kept in signature for v4.5
            (per-block target color analysis could adjust dilution per pull).
        target_total_pulls: globally desired pull count (Emma=132).
        target_pull_tolerance: ± tolerance for the global sum.
        starting_order_step: first absolute print order step.
        starting_pull_id: first pull_id.

    Returns:
        dict[block_id, list[PullSpec]] — each list is sorted by pass_index.

    The pull order across blocks follows light→dark role order:
    underlayer_light first, then local_chroma, then regional_mass, then
    key_detail; ties broken by block_id ascending.
    """
    if plate_count <= 0:
        return {}
    if not role_assignments:
        raise ValueError("role_assignments cannot be empty")

    # Per-block target count from role
    role_per_block: dict[int, Role] = dict(role_assignments)
    per_block: dict[int, int] = {
        b: _target_pulls_for_role(r) for b, r in role_per_block.items()
    }

    # Global sum adjust
    per_block = _adjust_to_global_target(
        per_block, role_per_block,
        target_total=target_total_pulls,
        target_tol=target_pull_tolerance,
    )

    # Hard clamp 1..5 (defensive)
    for b in per_block:
        per_block[b] = max(MIN_PULLS_PER_BLOCK,
                           min(MAX_PULLS_PER_BLOCK, per_block[b]))

    # ------------------------------------------------------------------
    # Sequence pulls into a global print order:
    # 1. group blocks by role in light→dark order
    # 2. for each role group, interleave one pull per block (round-robin)
    #    so the first pulls of every underlayer come before any second
    #    pulls (matches Pace early-light-large rule)
    # ------------------------------------------------------------------
    role_order = ["underlayer_light", "local_chroma", "regional_mass", "key_detail"]
    blocks_by_role: dict[Role, list[int]] = {r: [] for r in role_order}
    for b, r in role_per_block.items():
        blocks_by_role[r].append(b)
    for r in role_order:
        blocks_by_role[r].sort()

    order_step = starting_order_step
    pull_id = starting_pull_id

    per_block_pulls: dict[int, list[PullSpec]] = {b: [] for b in role_per_block}

    for role in role_order:
        blocks = blocks_by_role[role]
        if not blocks:
            continue
        max_pulls_this_role = max((per_block[b] for b in blocks), default=0)
        # Round-robin: pass_index 1 across all blocks of this role first,
        # then pass_index 2, etc.
        for pass_idx in range(1, max_pulls_this_role + 1):
            for b in blocks:
                if pass_idx > per_block[b]:
                    continue
                pull = PullSpec(
                    pull_id=pull_id,
                    block_id=b,
                    pigment_id=_ROLE_DEFAULT_PIGMENT[role],
                    opacity=_ROLE_DEFAULT_OPACITY[role],
                    dilution=_ROLE_DEFAULT_DILUTION[role],
                    order_step=order_step,
                    pass_index=pass_idx,
                    role=role,
                    mask_subset=None,
                )
                per_block_pulls[b].append(pull)
                order_step += 1
                pull_id += 1

    # Sort each list by pass_index for determinism (already sorted but explicit)
    for b in per_block_pulls:
        per_block_pulls[b].sort(key=lambda p: p.pass_index)

    return per_block_pulls
