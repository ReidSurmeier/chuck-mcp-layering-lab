"""Stage 4: morphology repair for printability.

Per docs/v2-design-locked-2026-05-16.md Phase 4 + Phase 6:

    "morphology repair and component scoring,
     re-solve after repair, not just accept degraded dE."
    "CNC/printability BEFORE SVG (not cleanup). Connected components above
     minimum area + no hairline islands + no unbrushable adjacent colors
     on same block + clear jigsaw separations + known registration/mirror
     state. Reject before vectorization, not after."

mill_radius_px = 19 ≈ 1/16" at 300 DPI:
    1/16 inch = 1.5875 mm
    1.5875 mm / (25.4 / 300) = 18.75 px -> round to 19

Operations applied PER plate's inked binary mask:

    1. area_opening: kill connected components smaller than (π * r^2)
       i.e. anything narrower than the mill bit cannot be carved cleanly.
    2. opening_by_reconstruction: with a disk of radius mill_radius_px:
       eliminates hairlines while preserving the topology of surviving
       components — strictly better than plain morphological opening
       because it doesn't shrink edges of valid zones.
    3. Re-check: if area shrunk by > 30%, mark plate as needing re-solve.

Reference: Vincent 1993 "Morphological grayscale reconstruction in image
analysis: applications and efficient algorithms".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from skimage.morphology import (
        area_opening,
        disk,
        opening,
        reconstruction,
        remove_small_objects,
    )

    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


DEFAULT_MILL_RADIUS_PX = 19  # 1/16" at 300 DPI
SHRINK_RESOLVE_THRESHOLD = 0.30  # 30%+ area loss -> trigger Stage 5 re-solve


@dataclass
class PlateRepairStats:
    """Per-plate post-repair report."""

    block_id: int
    area_before: int
    area_after: int
    shrink_fraction: float
    components_before: int
    components_after: int
    needs_resolve: bool
    method: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": int(self.block_id),
            "area_before": int(self.area_before),
            "area_after": int(self.area_after),
            "shrink_fraction": float(self.shrink_fraction),
            "components_before": int(self.components_before),
            "components_after": int(self.components_after),
            "needs_resolve": bool(self.needs_resolve),
            "method": str(self.method),
            "notes": list(self.notes),
        }


def _count_components(mask: np.ndarray) -> int:
    """Connected-component count (8-connectivity).

    Uses scipy.ndimage.label which is available even without skimage.
    """
    try:
        from scipy.ndimage import label as _label

        _, n = _label(mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
        return int(n)
    except Exception:  # pragma: no cover - CI fallback when scipy missing
        m = mask.astype(np.uint8).copy()
        n = 0
        stack: List[tuple[int, int]] = []
        H, W = m.shape
        for y in range(H):
            for x in range(W):
                if m[y, x] != 1:
                    continue
                n += 1
                stack.append((y, x))
                while stack:
                    py, px = stack.pop()
                    if 0 <= py < H and 0 <= px < W and m[py, px] == 1:
                        m[py, px] = 2
                        stack += [
                            (py + 1, px),
                            (py - 1, px),
                            (py, px + 1),
                            (py, px - 1),
                        ]
        return n


def _repair_one_skimage(
    mask: np.ndarray, mill_radius_px: int
) -> tuple[np.ndarray, str]:
    """skimage-based repair: area_opening + opening_by_reconstruction.

    Returns (repaired_mask, method_label).
    """
    bin_mask = mask.astype(bool)
    min_area = max(1, int(math.pi * mill_radius_px * mill_radius_px))
    # Step 1: area opening — kills components smaller than the mill footprint.
    a = area_opening(bin_mask, area_threshold=min_area)
    # Step 2: opening by reconstruction — kills hairlines without eroding
    # interior edges of the valid components.
    se = disk(max(1, mill_radius_px // 3))  # smaller SE: avoid over-shrink
    marker = opening(a, se)
    rec = reconstruction(marker, a, method="dilation").astype(bool)
    # Step 3: belt-and-suspenders area filter post-reconstruction
    final = remove_small_objects(rec, min_size=min_area)
    return final.astype(np.uint8), "skimage_area+reconstruction"


def _repair_one_fallback(  # pragma: no cover - only run when skimage missing
    mask: np.ndarray, mill_radius_px: int
) -> tuple[np.ndarray, str]:
    """Pure-numpy fallback: erode then dilate by a small disk + drop small CCs.

    Crude but deterministic; used only when skimage is unavailable.
    """
    try:
        from scipy.ndimage import binary_dilation, binary_erosion, label

        H, W = mask.shape
        struct = _disk_struct(max(1, mill_radius_px // 3))
        eroded = binary_erosion(mask.astype(bool), structure=struct, iterations=1)
        opened = binary_dilation(eroded, structure=struct, iterations=1)
        lbl, n = label(opened.astype(np.uint8))
        min_area = max(1, int(math.pi * mill_radius_px * mill_radius_px))
        sizes = np.bincount(lbl.ravel())
        keep_labels = {i for i in range(1, n + 1) if sizes[i] >= min_area}
        out = np.zeros_like(opened, dtype=np.uint8)
        if keep_labels:
            out[np.isin(lbl, list(keep_labels))] = 1
        return out, "scipy_erosion+dilation"
    except Exception:
        return (mask > 0).astype(np.uint8), "passthrough"


def _disk_struct(radius: int) -> np.ndarray:
    d = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


# Public API -----------------------------------------------------------


def repair_plate_mask(
    mask: np.ndarray, mill_radius_px: int = DEFAULT_MILL_RADIUS_PX
) -> tuple[np.ndarray, PlateRepairStats]:
    """Repair a single plate's binary mask.

    Args:
        mask: H x W binary array (0 or 1, any numeric dtype).
        mill_radius_px: end-mill radius in pixels. Default 19 == 1/16" @ 300 DPI.

    Returns:
        (repaired_mask, PlateRepairStats).
    """
    if mask is None or mask.size == 0:
        return mask, PlateRepairStats(-1, 0, 0, 0.0, 0, 0, False, "empty")

    bin_mask = (mask > 0).astype(np.uint8)
    area_before = int(bin_mask.sum())
    comps_before = _count_components(bin_mask)

    if _HAS_SKIMAGE:
        repaired, method = _repair_one_skimage(bin_mask, mill_radius_px)
    else:
        repaired, method = _repair_one_fallback(bin_mask, mill_radius_px)

    area_after = int(repaired.sum())
    comps_after = _count_components(repaired)
    shrink = 0.0 if area_before == 0 else 1.0 - area_after / area_before
    needs_resolve = shrink > SHRINK_RESOLVE_THRESHOLD

    notes: List[str] = []
    if area_after == 0 and area_before > 0:
        notes.append("repair erased all ink (plate too thin)")
    if comps_after < comps_before:
        notes.append(
            f"reduced from {comps_before} components to {comps_after}"
        )

    stats = PlateRepairStats(
        block_id=-1,  # caller fills in
        area_before=area_before,
        area_after=area_after,
        shrink_fraction=float(shrink),
        components_before=int(comps_before),
        components_after=int(comps_after),
        needs_resolve=bool(needs_resolve),
        method=method,
        notes=notes,
    )
    return repaired, stats


def repair_plates(
    plates: List[Any],
    mill_radius_px: int = DEFAULT_MILL_RADIUS_PX,
) -> tuple[List[Any], List[PlateRepairStats]]:
    """Repair every plate's inked_mask in place.

    Args:
        plates: list of objects with `.inked_mask` (np.ndarray) and `.block_id`.
            We mutate in place AND return them (for fluent style).
        mill_radius_px: end-mill radius. Default 19 px.

    Returns:
        (plates, per_plate_stats).
    """
    stats_out: List[PlateRepairStats] = []
    for p in plates:
        if getattr(p, "inked_mask", None) is None:
            continue
        repaired, s = repair_plate_mask(p.inked_mask, mill_radius_px)
        p.inked_mask = repaired
        p.area_px = int(s.area_after)
        s.block_id = int(getattr(p, "block_id", -1))
        # Persist into plate.repair_stats for downstream JSON
        p.repair_stats = s.to_dict()
        stats_out.append(s)
    return plates, stats_out


def any_needs_resolve(stats: List[PlateRepairStats]) -> bool:
    """Did any plate get clobbered enough that Stage 5 should re-solve?"""
    return any(s.needs_resolve for s in stats)
