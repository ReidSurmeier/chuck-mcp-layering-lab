"""Validator 3: jigsaw_separation_score.

Cells/zones living on the same physical plate must be far enough apart
that the printer can brush ink into each zone without bleeding into
neighbors. Per docs/v2-design-locked-2026-05-16.md row 3:

    "Min separation between zones on one block >= 5mm physical"

A "zone" here is a connected component of inked cells on the plate.
This validator measures the minimum distance between ANY pair of
disjoint zones, in physical millimeters.

The cell IDs are expected to come with pixel coordinates so we can
compute physical distances given a DPI and block physical width.

Inputs (two supported representations):

1. Cell-graph + pixel positions:
     cells_in_plate: list of cell IDs on this plate
     cell_pixel_positions: dict[cell_id -> (y, x) pixel centroid]
     cell_pixel_radii: dict[cell_id -> approx half-width in pixels]
     adjacency: dict[cell_id -> list[cell_id]] (cells touching in graph)
     dpi: image DPI
     OR

2. Inked binary mask:
     plate_mask: 2D binary numpy array (H, W) of inked pixels
     dpi: image DPI

Threshold: 5 mm.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

MIN_SEPARATION_MM = 5.0
DEFAULT_DPI = 300.0  # typical CNC + washi target


def _mm_per_pixel(dpi: float) -> float:
    return 25.4 / dpi


def _zones_from_mask(plate_mask: np.ndarray, min_zone_px: int = 16) -> list[dict]:
    """Connected-component zones in the inked mask.

    min_zone_px filters out tiny noise specks (< 16px area).
    """
    if plate_mask.ndim != 2:
        plate_mask = plate_mask.mean(axis=-1) if plate_mask.ndim == 3 else plate_mask
    binary = (plate_mask > 0.5).astype(np.uint8)
    lbl = label(binary, connectivity=2)
    zones = []
    for r in regionprops(lbl):
        if r.area < min_zone_px:
            continue
        zones.append({
            "label": int(r.label),
            "centroid_yx": r.centroid,
            "bbox": r.bbox,  # (min_row, min_col, max_row, max_col)
            "area_px": int(r.area),
            "coords": r.coords,  # (N, 2) pixel locations
        })
    return zones


def _zone_min_distance_px(zone_a: dict, zone_b: dict, sample_cap: int = 500) -> float:
    """Min pixel distance between two connected components.

    Subsamples coordinate arrays for speed if either zone has many
    pixels (>500 points). Worst case is O(sample_cap^2) per pair.
    """
    ca = zone_a["coords"]
    cb = zone_b["coords"]
    if len(ca) > sample_cap:
        idx = np.linspace(0, len(ca) - 1, sample_cap).astype(np.int64)
        ca = ca[idx]
    if len(cb) > sample_cap:
        idx = np.linspace(0, len(cb) - 1, sample_cap).astype(np.int64)
        cb = cb[idx]
    # Pairwise distances
    diff = ca[:, None, :] - cb[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return float(d.min())


def score_from_mask(
    plate_mask: np.ndarray,
    dpi: float = DEFAULT_DPI,
    return_components: bool = False,
):
    """Compute separation directly from a binary inked mask."""
    zones = _zones_from_mask(plate_mask)
    mm_per_px = _mm_per_pixel(dpi)

    if len(zones) <= 1:
        # Trivially passes if 0 or 1 zone (no inter-zone distance to enforce)
        if return_components:
            return {
                "min_separation_mm": float("inf"),
                "passes": True,
                "n_zones": len(zones),
                "min_threshold_mm": MIN_SEPARATION_MM,
                "violations": [],
                "dpi": dpi,
            }
        return float("inf")

    min_d_px = float("inf")
    violations = []
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            d_px = _zone_min_distance_px(zones[i], zones[j])
            d_mm = d_px * mm_per_px
            if d_mm < MIN_SEPARATION_MM:
                violations.append({
                    "zone_a": zones[i]["label"],
                    "zone_b": zones[j]["label"],
                    "distance_mm": float(d_mm),
                })
            if d_px < min_d_px:
                min_d_px = d_px

    min_d_mm = min_d_px * mm_per_px
    passing = min_d_mm >= MIN_SEPARATION_MM

    if return_components:
        return {
            "min_separation_mm": float(min_d_mm),
            "passes": bool(passing),
            "n_zones": len(zones),
            "min_threshold_mm": MIN_SEPARATION_MM,
            "violations": violations,
            "dpi": dpi,
        }
    return float(min_d_mm)


def score(
    plate=None,
    cells_in_plate: Optional[Iterable[int]] = None,
    cell_pixel_positions: Optional[dict] = None,
    adjacency: Optional[dict] = None,
    plate_mask: Optional[np.ndarray] = None,
    dpi: float = DEFAULT_DPI,
    return_components: bool = False,
):
    """Top-level entry. Dispatches to the right representation.

    If plate_mask supplied -> mask-based.
    Else uses cell positions + adjacency to derive zones via graph
    connected components.
    """
    if plate_mask is not None:
        return score_from_mask(plate_mask, dpi, return_components=return_components)

    if cells_in_plate is None or cell_pixel_positions is None:
        raise ValueError("Either plate_mask OR (cells_in_plate + cell_pixel_positions) required")

    cells = list(cells_in_plate)
    if not cells:
        if return_components:
            return {
                "min_separation_mm": float("inf"),
                "passes": True,
                "n_zones": 0,
                "min_threshold_mm": MIN_SEPARATION_MM,
                "violations": [],
                "dpi": dpi,
            }
        return float("inf")

    # Group cells into zones via adjacency BFS, but only counting
    # adjacency among cells that are BOTH on this plate.
    cell_set = set(cells)
    adj = adjacency or {}
    visited: set[int] = set()
    zones: list[list[int]] = []
    for c in cells:
        if c in visited:
            continue
        # BFS
        zone = []
        stack = [c]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            zone.append(cur)
            for nb in adj.get(cur, []):
                if nb in cell_set and nb not in visited:
                    stack.append(nb)
        zones.append(zone)

    mm_per_px = _mm_per_pixel(dpi)
    if len(zones) <= 1:
        if return_components:
            return {
                "min_separation_mm": float("inf"),
                "passes": True,
                "n_zones": len(zones),
                "min_threshold_mm": MIN_SEPARATION_MM,
                "violations": [],
                "dpi": dpi,
            }
        return float("inf")

    # Each zone -> array of pixel coords (use centroids as a stand-in)
    zone_centroids = []
    for z in zones:
        coords = np.array([cell_pixel_positions[c] for c in z])
        zone_centroids.append(coords)

    min_d_px = float("inf")
    violations = []
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            ca, cb = zone_centroids[i], zone_centroids[j]
            diff = ca[:, None, :] - cb[None, :, :]
            d = np.sqrt((diff ** 2).sum(axis=-1))
            d_px = float(d.min())
            d_mm = d_px * mm_per_px
            if d_mm < MIN_SEPARATION_MM:
                violations.append({
                    "zone_a_idx": i, "zone_b_idx": j,
                    "distance_mm": float(d_mm),
                })
            if d_px < min_d_px:
                min_d_px = d_px

    min_d_mm = min_d_px * mm_per_px
    passing = min_d_mm >= MIN_SEPARATION_MM

    if return_components:
        return {
            "min_separation_mm": float(min_d_mm),
            "passes": bool(passing),
            "n_zones": len(zones),
            "min_threshold_mm": MIN_SEPARATION_MM,
            "violations": violations,
            "dpi": dpi,
        }
    return float(min_d_mm)


def passes(*args, **kwargs) -> bool:
    out = score(*args, return_components=True, **{k: v for k, v in kwargs.items() if k != "return_components"})
    return bool(out["passes"])


if __name__ == "__main__":
    # Smoke: two zones far apart vs touching
    H, W = 100, 100
    mask_good = np.zeros((H, W), dtype=np.uint8)
    mask_good[10:30, 10:30] = 1
    mask_good[60:80, 60:80] = 1
    print("FAR-APART zones @ 300dpi:", score_from_mask(mask_good, dpi=300, return_components=True))

    mask_bad = np.zeros((H, W), dtype=np.uint8)
    mask_bad[10:30, 10:30] = 1
    mask_bad[31:50, 10:30] = 1  # 1px gap
    print("TOUCHING zones @ 300dpi:", score_from_mask(mask_bad, dpi=300, return_components=True))
