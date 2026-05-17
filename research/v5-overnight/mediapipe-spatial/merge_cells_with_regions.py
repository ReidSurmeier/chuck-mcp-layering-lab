"""chuck-mcp v5 — Merge Cells with Face Regions

Joins SNIC cell IDs to canonical face region names produced by
`face_spatial.extract_face_regions`. Wraps the v3 merge helper to:

  1. Run the v3 centroid/overlap join (region_name -> [cell_ids]).
  2. Resolve each cell to a SINGLE primary region using the v3 priority list
     (most specific wins ties: e.g. upper_lip > lips > face).
  3. Guarantee total coverage — any cell not claimed by any face region is
     labelled "background" (which is itself a vocabulary region).

This is the data structure the plate-builder needs to enforce region
constraints: dict[cell_id -> region_name].
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

import face_spatial


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def assign_cells_to_regions(
    face_regions: dict,
    snic_labels: np.ndarray,
    *,
    strategy: str = "centroid",
) -> dict[int, str]:
    """Assign every SNIC cell ID in `snic_labels` to exactly one face region.

    Returns dict[cell_id -> region_name] where region_name is in the
    canonical v3 vocabulary (`face_spatial.list_supported_regions()`).

    Cells that fall outside every face region are labelled "background" so the
    return dict covers ALL cells (no orphans).
    """
    # Stage 1: region -> [cell_ids] using v3 centroid/overlap join.
    region_to_cells: dict[str, list[int]] = (
        face_spatial.merge_face_regions_with_snic_cells(
            face_regions, snic_labels, strategy=strategy
        )
    )

    # Stage 2: invert with v3 priority ordering — most specific region wins.
    primary: dict[int, str] = face_spatial.resolve_cell_to_primary_region(
        region_to_cells
    )

    # Stage 3: backfill any cell with no primary assignment as "background".
    # The background region in the v3 vocabulary already covers all non-face
    # pixels, so this is a stable canonical label.
    out: dict[int, str] = {}
    for cid in np.unique(snic_labels):
        cid_int = int(cid)
        out[cid_int] = primary.get(cid_int, "background")
    return out


def assign_regions_to_cells(
    face_regions: dict,
    snic_labels: np.ndarray,
    *,
    strategy: str = "centroid",
) -> dict[str, list[int]]:
    """Inverse view of `assign_cells_to_regions`: region_name -> [cell_ids].

    Unlike the raw v3 merger output (which may double-count a cell across
    overlapping regions), this view uses the primary assignment from
    `assign_cells_to_regions` so each cell appears in exactly one region.
    """
    primary = assign_cells_to_regions(
        face_regions, snic_labels, strategy=strategy
    )
    out: dict[str, list[int]] = {}
    for cid, region in primary.items():
        out.setdefault(region, []).append(cid)
    for r in out:
        out[r].sort()
    return out


def lookup_cells_for_regions(
    region_to_cells: dict[str, list[int]],
    region_names: Iterable[str],
) -> set[int]:
    """Return the union of cell IDs across the named regions.

    Convenience helper for plate constraint filtering — given a plate's
    declared face_region_constraint = ["left_cheek", "right_cheek", "temple"],
    return the set of allowed cell IDs.
    """
    out: set[int] = set()
    for name in region_names:
        out.update(region_to_cells.get(name, []))
    return out


__all__ = [
    "assign_cells_to_regions",
    "assign_regions_to_cells",
    "lookup_cells_for_regions",
]
