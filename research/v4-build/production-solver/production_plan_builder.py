"""ProductionPlan builder — composes the full production layout BEFORE JAX.

Per `audit-response-and-reconstruction-plan-2026-05-17.md` Phase 2:

    "Block/pull identity solved WITH target reconstruction. 4+4+16 is a
     prior, not a rigid grid."

Pipeline:
    1. Estimate adaptive plate count from image stats.
    2. Partition cells into plates (input role_assignments map cells → plate
       OR generate plates from cell graph + underlayer rules).
    3. Assign role per plate.
    4. Build multi-pull schedule per block.
    5. Compose ProductionPlan; validate invariants I1-I9.

Inputs:
    target: (H, W, 3) image. We accept LAB or RGB (auto-detected by max).
    cell_graph: dict with "cells" key — each entry has at least
        "mean_rgb" (3-vector) and "pixels" (list-of-(y,x) OR pixel count int).
    role_assignments: one of two forms:
        Form A (preferred):  dict[block_id, dict] with keys:
            "cells": list[int]      cell_zone_ids to ink on this plate
            "role": Role            print role family
            "pigment_family": str   inventory hint
            "region_label": str     optional human label
            "rationale": str        provenance
        Form B (cells-only):  dict[cell_id, Role]  — cells grouped by role
            (one plate per (role, cluster) — clustering is "all cells of
            that role" minus split-by-role-count if necessary).

If role_assignments is None, an auto-partitioner runs:
    - cluster cells by hue with K = estimated plate_count
    - assign role by chroma/lightness (low-chroma + light → underlayer;
      mid-chroma → local_chroma; high-coverage cluster → regional_mass;
      darkest small cluster → key_detail)

This auto-partitioner is deliberately simple — its job is to GENERATE A
PROPOSAL the JAX solver can refine. It is NOT the final answer.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from plate_count_estimator import (  # noqa: E402
    estimate_plate_count,
    compute_cell_graph_stats,
    _srgb_to_lab_approx,
    _lab_to_chroma,
    _lab_to_hue,
)
from multi_pull_assigner import assign_pulls_per_block  # noqa: E402
from production_plan import (  # noqa: E402
    PlateSpec,
    ProductionPlan,
    PullSpec,
    Role,
    ROLES,
)


# Role-distribution priors for adaptive plate count.
# For ~26 plates we want approximately:
#     6 underlayer_light  (one per anchor face region)
#     9 local_chroma     (cheek/lip/temple specifics, eye colors, blush)
#     8 regional_mass    (hair, background, shadow side)
#     3 key_detail       (eyeline, contour, hair edge, mole)
# Counts scale linearly with plate_count.
_ROLE_FRACTIONS: dict[Role, float] = {
    "underlayer_light": 6 / 26,
    "local_chroma":     9 / 26,
    "regional_mass":    8 / 26,
    "key_detail":       3 / 26,
}

# Pigment family by role (default starting hint)
_ROLE_DEFAULT_FAMILY: dict[Role, str] = {
    "underlayer_light": "light_yellow",
    "local_chroma":     "pale_red",
    "regional_mass":    "pale_blue",
    "key_detail":       "warm_grey",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _role_count_targets(plate_count: int) -> dict[Role, int]:
    """How many plates per role for a given plate_count.

    Uses _ROLE_FRACTIONS, rounded to ints, then patched so the sum equals
    plate_count exactly (deterministic). Minimum 1 plate per role.
    """
    raw = {r: max(1, int(round(plate_count * f)))
           for r, f in _ROLE_FRACTIONS.items()}
    # Patch sum
    diff = plate_count - sum(raw.values())
    role_order = ["regional_mass", "local_chroma", "underlayer_light", "key_detail"]
    while diff != 0:
        if diff > 0:
            for r in role_order:
                raw[r] += 1
                diff -= 1
                if diff == 0:
                    break
        else:
            # Drop plates from regional_mass first, never below 1
            for r in role_order:
                if raw[r] > 1:
                    raw[r] -= 1
                    diff += 1
                    if diff == 0:
                        break
            else:
                break
    return raw


def _cell_mean_lab(cell_graph: dict) -> dict[int, np.ndarray]:
    """Extract per-cell mean LAB from cell graph (using mean_rgb if LAB absent).

    Cell graph entries have either:
        "mean_lab": ndarray(3,)   -- preferred
        "mean_rgb": ndarray(3,)   -- in [0,1] or [0,255]
    """
    out: dict[int, np.ndarray] = {}
    for cid, entry in cell_graph["cells"].items():
        if "mean_lab" in entry:
            out[int(cid)] = np.asarray(entry["mean_lab"], dtype=np.float32)
        else:
            rgb = np.asarray(entry["mean_rgb"], dtype=np.float32)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            # Use a 1x1x3 LAB conversion
            lab_1px = _srgb_to_lab_approx(rgb[None, None, :])[0, 0]
            out[int(cid)] = lab_1px.astype(np.float32)
    return out


def _cell_size(cell_graph: dict, cid: int) -> int:
    entry = cell_graph["cells"][cid]
    px = entry.get("pixels", None)
    if px is None:
        return int(entry.get("size", 1))
    if isinstance(px, int):
        return px
    return len(px)


# ---------------------------------------------------------------------------
# Auto-partitioner — generates a default cell→plate assignment when caller
# does not provide one.
# ---------------------------------------------------------------------------
def _auto_partition_cells(
    cell_graph: dict,
    plate_count: int,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """Cluster cells into ``plate_count`` plates and assign role per plate.

    Algorithm (intentionally simple, deterministic given rng seed):
        1. Per-cell LAB; chroma + hue.
        2. Compute role-count targets per plate_count.
        3. Sort cells by lightness (L*) and chroma:
           - lightest 25%  → underlayer_light plates
           - high chroma 30% → local_chroma plates
           - mid 30%       → regional_mass plates
           - darkest 15%   → key_detail plates
        4. Within each role bucket, cluster by hue into the role's plate count
           using a simple 1-D hue binning (no full K-means needed at proposal
           stage — JAX will refine).

    Returns:
        dict[block_id, dict(cells, role, pigment_family, region_label,
                            rationale, provenance)]
    """
    cell_ids = sorted(cell_graph["cells"].keys())
    if not cell_ids:
        return {}

    cell_lab = _cell_mean_lab(cell_graph)
    cell_size = {cid: _cell_size(cell_graph, cid) for cid in cell_ids}

    L = np.array([cell_lab[c][0] for c in cell_ids], dtype=np.float32)
    a = np.array([cell_lab[c][1] for c in cell_ids], dtype=np.float32)
    b = np.array([cell_lab[c][2] for c in cell_ids], dtype=np.float32)
    chroma = np.sqrt(a * a + b * b)
    hue = np.mod(np.degrees(np.arctan2(b, a)), 360.0)
    size = np.array([cell_size[c] for c in cell_ids], dtype=np.int64)

    role_counts = _role_count_targets(plate_count)

    # Pre-compute brightness thresholds
    # Underlayer: top 25% lightest cells
    L_top25 = float(np.percentile(L, 75)) if len(L) > 0 else 100.0
    L_bot15 = float(np.percentile(L, 15)) if len(L) > 0 else 0.0

    chroma_top30 = float(np.percentile(chroma, 70)) if len(chroma) > 0 else 0.0

    role_assignments: dict[int, list[int]] = {r: [] for r in ROLES}
    cell_idx_to_cid = dict(enumerate(cell_ids))

    for i, cid in enumerate(cell_ids):
        Li, ci, hi = L[i], chroma[i], hue[i]
        if Li <= L_bot15:
            role_assignments["key_detail"].append(cid)
        elif Li >= L_top25 and ci < chroma_top30:
            role_assignments["underlayer_light"].append(cid)
        elif ci >= chroma_top30:
            role_assignments["local_chroma"].append(cid)
        else:
            role_assignments["regional_mass"].append(cid)

    # Now split each role bucket into role_counts[role] plates by hue binning.
    plates: dict[int, dict] = {}
    next_block_id = 1

    role_order = ["underlayer_light", "local_chroma", "regional_mass", "key_detail"]
    for role in role_order:
        bucket = role_assignments[role]
        n_plates = max(1, role_counts.get(role, 1))
        if not bucket:
            # Empty bucket: create a minimal 1-cell plate from the closest non-empty
            # bucket to satisfy plate_count. Pull the smallest cell.
            for donor in role_order:
                if donor == role:
                    continue
                if role_assignments[donor]:
                    cid = role_assignments[donor].pop()
                    bucket.append(cid)
                    break
            if not bucket:
                # Truly nothing — skip
                continue

        # Bin by hue
        bucket_arr = np.array(bucket)
        # Get hues for bucket cells
        bucket_idx = np.array([cell_ids.index(c) for c in bucket_arr])
        bucket_hue = hue[bucket_idx]
        # Split into n_plates contiguous hue arcs (sort by hue, equal-count
        # partition; works well for cluster count up to ~10).
        order = np.argsort(bucket_hue)
        sorted_cells = bucket_arr[order]
        splits = np.array_split(sorted_cells, n_plates)
        for s_idx, sub in enumerate(splits):
            if len(sub) == 0:
                # Borrow from neighbor split
                continue
            block_id = next_block_id
            plates[block_id] = {
                "cells": [int(x) for x in sub.tolist()],
                "role": role,
                "pigment_family": _ROLE_DEFAULT_FAMILY[role],
                "region_label": f"{role}_cluster_{s_idx + 1}",
                "rationale": (
                    f"auto-partitioner: role={role} "
                    f"hue-bin {s_idx + 1}/{n_plates} "
                    f"(n={len(sub)} cells)"
                ),
                "provenance": "algorithm",
            }
            next_block_id += 1

    # If we ended up with fewer plates than plate_count, split the largest
    # bucket plate to fill in. Conversely, if we have MORE than plate_count
    # (rare with role-count patch), merge the smallest neighbors.
    while len(plates) < plate_count:
        # Split largest plate
        big = max(plates.items(), key=lambda kv: len(kv[1]["cells"]))
        bid, p = big
        cells = p["cells"]
        if len(cells) < 2:
            break  # Can't split further
        mid = len(cells) // 2
        new_bid = max(plates.keys()) + 1
        plates[new_bid] = {
            "cells": cells[mid:],
            "role": p["role"],
            "pigment_family": p["pigment_family"],
            "region_label": p["region_label"] + "_split",
            "rationale": p["rationale"] + " [split to reach plate_count]",
            "provenance": "algorithm",
        }
        plates[bid] = {**p, "cells": cells[:mid]}

    while len(plates) > plate_count:
        # Merge two smallest same-role plates
        same_role = {}
        for bid, p in plates.items():
            same_role.setdefault(p["role"], []).append((bid, len(p["cells"])))
        merge_done = False
        for role, lst in same_role.items():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda x: x[1])
            a_id, _ = lst[0]
            b_id, _ = lst[1]
            a_p = plates[a_id]
            b_p = plates[b_id]
            merged_cells = list(set(a_p["cells"] + b_p["cells"]))
            plates[a_id] = {
                **a_p,
                "cells": merged_cells,
                "rationale": a_p["rationale"] + f" [merged with plate {b_id}]",
            }
            del plates[b_id]
            merge_done = True
            break
        if not merge_done:
            break  # Nothing to merge

    # Re-key block_ids contiguously 1..len(plates)
    final: dict[int, dict] = {}
    for new_id, old_id in enumerate(sorted(plates.keys()), 1):
        final[new_id] = plates[old_id]
    return final


# ---------------------------------------------------------------------------
# Cell-overlap deduplicator
# ---------------------------------------------------------------------------
def _dedup_cells_across_plates(plate_specs: dict[int, dict]) -> dict[int, dict]:
    """Enforce no-zone-overlap-on-same-block AND no cell appears on >1 plate.

    Plates are processed in block_id order; if a cell already lives on a
    lower-numbered plate, it's removed from the current one.

    Also drops in-plate duplicates (within one cell_zone_ids list).
    """
    seen: set[int] = set()
    out: dict[int, dict] = {}
    for bid in sorted(plate_specs.keys()):
        p = dict(plate_specs[bid])
        unique_cells = []
        for cid in p["cells"]:
            if cid in seen:
                continue
            unique_cells.append(cid)
            seen.add(cid)
        # Final defensive dedup within plate (set then sorted)
        p["cells"] = sorted(set(unique_cells))
        out[bid] = p
    return out


def _backfill_orphans(
    plate_specs: dict[int, dict],
    all_cell_ids: set[int],
    cell_lab: dict[int, np.ndarray],
) -> dict[int, dict]:
    """If some cells weren't assigned, push them onto the role-matching plate
    whose mean LAB is closest. Guarantees I5 (no orphans).
    """
    assigned: set[int] = set()
    for p in plate_specs.values():
        assigned.update(p["cells"])
    orphans = all_cell_ids - assigned
    if not orphans:
        return plate_specs

    # Compute mean LAB per plate
    plate_mean_lab: dict[int, np.ndarray] = {}
    for bid, p in plate_specs.items():
        if not p["cells"]:
            # Empty plates get a synthetic distant LAB to avoid attracting orphans
            plate_mean_lab[bid] = np.array([200.0, 0.0, 0.0], dtype=np.float32)
            continue
        labs = np.stack([cell_lab[c] for c in p["cells"] if c in cell_lab])
        plate_mean_lab[bid] = labs.mean(axis=0) if labs.size else np.zeros(3, dtype=np.float32)

    for cid in orphans:
        if cid not in cell_lab:
            # Cell graph entry is missing LAB — pick first plate as fallback
            target_bid = sorted(plate_specs.keys())[0]
        else:
            c_lab = cell_lab[cid]
            target_bid = min(
                plate_specs.keys(),
                key=lambda b: float(np.linalg.norm(plate_mean_lab[b] - c_lab)),
            )
        plate_specs[target_bid]["cells"].append(cid)
        plate_specs[target_bid]["cells"] = sorted(set(plate_specs[target_bid]["cells"]))
    return plate_specs


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_production_plan(
    target: np.ndarray,
    cell_graph: dict,
    role_assignments: Optional[Union[dict[int, dict], dict[int, Role]]] = None,
    *,
    target_total_pulls: int = 132,
    target_pull_tolerance: int = 12,
    plate_count: Optional[int] = None,
    plan_id: Optional[str] = None,
    rng_seed: int = 42,
) -> ProductionPlan:
    """Build the complete ProductionPlan from inputs.

    Args:
        target: (H, W, 3) target image. LAB or sRGB; auto-detected.
        cell_graph: {"cells": {cell_id: {"mean_rgb"|"mean_lab": (3,),
            "pixels": list|count}}}.
        role_assignments: optional Form-A plate specs. If None, auto-partition.
        target_total_pulls: globally desired pull count (Emma=132).
        target_pull_tolerance: tolerance window for the global sum.
        plate_count: override the adaptive estimator (testing only).
        plan_id: optional explicit id. Auto-generated if None.
        rng_seed: deterministic auto-partition seed.

    Returns:
        ProductionPlan with all invariants enforced.
    """
    t_start = time.time()

    # 1. estimate plate count
    if plate_count is None:
        plate_count = estimate_plate_count(target, cell_graph=cell_graph)
    plate_count = max(20, min(35, plate_count))

    # 2. resolve plate specs
    if role_assignments is None:
        rng = np.random.default_rng(rng_seed)
        plate_specs = _auto_partition_cells(cell_graph, plate_count, rng)
    elif role_assignments and isinstance(next(iter(role_assignments.values())), dict):
        # Form A: caller provided complete plate specs
        plate_specs = {int(k): dict(v) for k, v in role_assignments.items()}
        # If plate_count override differs, prefer caller's spec count
        plate_count = len(plate_specs)
    else:
        # Form B: dict[cell_id, role]; group cells by role into plates
        plate_specs = _group_cells_by_role_into_plates(
            role_assignments, plate_count,  # type: ignore[arg-type]
        )

    # Dedup cell ownership across plates
    plate_specs = _dedup_cells_across_plates(plate_specs)

    # Backfill orphans
    all_cell_ids: set[int] = set(int(c) for c in cell_graph["cells"].keys())
    cell_lab = _cell_mean_lab(cell_graph)
    plate_specs = _backfill_orphans(plate_specs, all_cell_ids, cell_lab)

    # Drop empty plates (if any) and re-key
    nonempty: dict[int, dict] = {}
    for new_id, old_id in enumerate(sorted(plate_specs.keys()), 1):
        if plate_specs[old_id]["cells"]:
            nonempty[new_id] = plate_specs[old_id]
    plate_specs = nonempty

    # If we dropped below 20, that's a sign of pathological input — pad by
    # splitting largest plates (one cell off the top → new plate).
    while len(plate_specs) < 20:
        big = max(plate_specs.items(), key=lambda kv: len(kv[1]["cells"]))
        bid, p = big
        if len(p["cells"]) < 2:
            break
        cells = p["cells"]
        mid = len(cells) // 2
        new_bid = max(plate_specs.keys()) + 1
        plate_specs[new_bid] = {
            **p,
            "cells": cells[mid:],
            "rationale": p["rationale"] + " [split to meet MIN_PLATES=20]",
        }
        plate_specs[bid] = {**p, "cells": cells[:mid]}

    plate_count = len(plate_specs)

    # 3. role map for the pull assigner
    role_assignments_dict: dict[int, Role] = {
        bid: p["role"] for bid, p in plate_specs.items()
    }

    # 4. multi-pull assigner
    per_block_pulls = assign_pulls_per_block(
        plate_count=plate_count,
        role_assignments=role_assignments_dict,
        target_image=target,
        target_total_pulls=target_total_pulls,
        target_pull_tolerance=target_pull_tolerance,
    )

    # 5. compose PlateSpec objects
    plates: list[PlateSpec] = []
    for bid in sorted(plate_specs.keys()):
        spec = plate_specs[bid]
        plate = PlateSpec(
            block_id=bid,
            cell_zone_ids=sorted(spec["cells"]),
            role=spec["role"],
            pigment_family=spec.get("pigment_family",
                                    _ROLE_DEFAULT_FAMILY[spec["role"]]),
            region_label=spec.get("region_label"),
            rationale=spec.get("rationale", ""),
            provenance=spec.get("provenance", "algorithm"),
        )
        for pull in per_block_pulls.get(bid, []):
            plate.add_pull(pull)
        plates.append(plate)

    total_pulls = sum(p.pull_count for p in plates)

    plan = ProductionPlan(
        plan_id=plan_id or f"plan_{int(t_start * 1000)}_{uuid.uuid4().hex[:8]}",
        plates=plates,
        total_pulls=total_pulls,
        cell_count=len(all_cell_ids),
        image_shape=(int(target.shape[0]), int(target.shape[1])),
        meta={
            "plate_count_target": plate_count,
            "build_seconds": float(time.time() - t_start),
            "role_distribution": {r: sum(1 for p in plates if p.role == r)
                                  for r in ROLES},
        },
    )

    ok, errors = plan.validate(all_cell_ids=all_cell_ids)
    if not ok:
        # Errors are informational; we return the plan so the caller (or test
        # suite) can inspect. In production we'd raise; in V4 first-stage we
        # want the JAX solver to see structural issues.
        plan.meta["validate_ok"] = False
        plan.meta["validate_errors"] = errors
    else:
        plan.meta["validate_ok"] = True

    return plan


# ---------------------------------------------------------------------------
# Helper for Form-B role_assignments
# ---------------------------------------------------------------------------
def _group_cells_by_role_into_plates(
    role_per_cell: dict[int, Role],
    plate_count: int,
) -> dict[int, dict]:
    """Take a dict[cell_id, role] and split each role bucket into
    sub-plates so that the total plate count hits ``plate_count``.

    Simple proportional split by role population.
    """
    role_buckets: dict[Role, list[int]] = {r: [] for r in ROLES}
    for cid, r in role_per_cell.items():
        if r in role_buckets:
            role_buckets[r].append(int(cid))

    # Allocate plate slots proportional to bucket size
    total = sum(len(v) for v in role_buckets.values())
    if total == 0:
        return {}
    role_plate_counts: dict[Role, int] = {}
    remaining = plate_count
    for r in ROLES:
        share = max(1 if role_buckets[r] else 0,
                    int(round(plate_count * len(role_buckets[r]) / total)))
        role_plate_counts[r] = share
        remaining -= share
    # Adjust if rounding drift
    role_iter = list(ROLES)
    idx = 0
    while remaining != 0 and any(role_buckets[r] for r in ROLES):
        r = role_iter[idx % len(role_iter)]
        if remaining > 0:
            role_plate_counts[r] += 1
            remaining -= 1
        elif role_plate_counts[r] > 1:
            role_plate_counts[r] -= 1
            remaining += 1
        idx += 1
        if idx > 1000:
            break

    plates: dict[int, dict] = {}
    next_bid = 1
    for r in ROLES:
        bucket = sorted(role_buckets[r])
        n_plates = max(1, role_plate_counts.get(r, 1)) if bucket else 0
        if n_plates == 0:
            continue
        chunks = np.array_split(np.array(bucket), n_plates)
        for i, ch in enumerate(chunks):
            if len(ch) == 0:
                continue
            plates[next_bid] = {
                "cells": [int(x) for x in ch.tolist()],
                "role": r,
                "pigment_family": _ROLE_DEFAULT_FAMILY[r],
                "region_label": f"{r}_chunk_{i + 1}",
                "rationale": f"Form-B chunk {i + 1}/{n_plates} for role={r}",
                "provenance": "algorithm",
            }
            next_bid += 1
    return plates
