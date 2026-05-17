"""
chuck-mcp v3 — Merge Face Regions with SNIC Cells
=================================================

Bridges the spatial / semantic gap:
  - face_region_mapper.py produces region polygons/masks keyed by name
    ("hair", "left_cheek", "upper_lip"...)
  - chuck-mcp's existing SNIC step produces a (H, W) integer-labelled
    superpixel map: each pixel holds its cell_id (0..N-1)
  - the solver needs: "give me the set of SNIC cell IDs that constitute
    'the cheek'"

This module performs that join.

Two strategies:

  centroid-in-polygon  (cheap, default)
      A cell belongs to a region iff its centroid (mean pixel position)
      falls inside the region polygon.
      Pro: O(N_cells * polygon_complexity). Stable. Trivially explainable.
      Con: cells straddling a region edge get assigned 0/1, not partially.

  area-overlap         (precise, opt-in)
      A cell belongs to a region iff > fraction (default 0.5) of its
      pixels fall inside the region mask.
      Pro: more accurate boundary handling.
      Con: O(N_cells * cell_pixel_count). Slower for high-res images.

Usage from MCP solver code:

    from face_region_mapper import extract_face_regions
    from merge_regions_with_cells import merge_face_regions_with_snic_cells

    regions = extract_face_regions(image_path)
    cell_assign = merge_face_regions_with_snic_cells(regions, snic_labels)
    cell_assign["left_cheek"]   # -> [cell_id_42, cell_id_43, cell_id_51, ...]

The Plate role-assignment pass then has direct access to which SNIC cells
belong to each face region, and Reid's prompt ("yellow under the cheek")
resolves to constraints on a specific list of cells.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import cv2
from shapely.geometry import Point, Polygon as ShPolygon


# ----------------------------------------------------------------------------
# Centroid table — computed ONCE per SNIC label image, reused per region.
# ----------------------------------------------------------------------------

@dataclass
class SNICCellGeometry:
    cell_id: int
    centroid: tuple[float, float]   # (x, y) pixel coords
    pixel_count: int
    bbox: tuple[int, int, int, int] # (xmin, ymin, xmax, ymax)


def compute_snic_cell_geometry(snic_labels: np.ndarray) -> dict[int, SNICCellGeometry]:
    """
    Given an (H, W) integer array of SNIC cell IDs, compute the centroid +
    pixel count + bbox for every cell, in one pass.

    Returns dict[cell_id -> SNICCellGeometry].
    """
    if snic_labels.dtype != np.int32 and snic_labels.dtype != np.int64:
        snic_labels = snic_labels.astype(np.int32)
    H, W = snic_labels.shape
    # Vectorized: for each unique cell, compute mean of its (x, y) pixel coords.
    unique_ids = np.unique(snic_labels)
    geometry: dict[int, SNICCellGeometry] = {}
    # Use cv2.connectedComponents-style approach: iterate, but use np.where
    # bulk masks (fast).
    ys, xs = np.indices((H, W))
    for cid in unique_ids:
        mask = snic_labels == cid
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        cx = float(xs[mask].mean())
        cy = float(ys[mask].mean())
        x_coords = xs[mask]
        y_coords = ys[mask]
        bbox = (int(x_coords.min()), int(y_coords.min()),
                int(x_coords.max()), int(y_coords.max()))
        geometry[int(cid)] = SNICCellGeometry(
            cell_id=int(cid), centroid=(cx, cy),
            pixel_count=cnt, bbox=bbox,
        )
    return geometry


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------

def _assign_by_centroid(
    region_polygon: list[tuple[int, int]] | None,
    region_mask: np.ndarray,
    cell_geometry: dict[int, SNICCellGeometry],
) -> list[int]:
    """Returns cell IDs whose centroid is inside the region."""
    if region_polygon is not None and len(region_polygon) >= 3:
        try:
            poly = ShPolygon(region_polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)  # fix self-intersections
            out: list[int] = []
            for cid, g in cell_geometry.items():
                if poly.contains(Point(g.centroid)):
                    out.append(cid)
            return sorted(out)
        except Exception:
            pass
    # fallback: test centroid against pixel mask directly
    out = []
    H, W = region_mask.shape
    for cid, g in cell_geometry.items():
        cx, cy = g.centroid
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < W and 0 <= iy < H and region_mask[iy, ix] > 0:
            out.append(cid)
    return sorted(out)


def _assign_by_overlap(
    region_mask: np.ndarray,
    snic_labels: np.ndarray,
    cell_geometry: dict[int, SNICCellGeometry],
    overlap_threshold: float = 0.5,
) -> list[int]:
    """
    Returns cell IDs where region_mask covers >= overlap_threshold * cell pixels.
    """
    region_bool = region_mask > 0
    out: list[int] = []
    for cid, g in cell_geometry.items():
        cell_mask = snic_labels == cid
        inter = int(np.logical_and(cell_mask, region_bool).sum())
        if g.pixel_count == 0:
            continue
        frac = inter / g.pixel_count
        if frac >= overlap_threshold:
            out.append(cid)
    return sorted(out)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

Strategy = Literal["centroid", "overlap"]


def merge_face_regions_with_snic_cells(
    face_regions: dict,                # dict[str -> FaceRegion]
    snic_labels: np.ndarray,           # (H, W) int array
    strategy: Strategy = "centroid",
    overlap_threshold: float = 0.5,
    precomputed_geometry: dict[int, SNICCellGeometry] | None = None,
) -> dict[str, list[int]]:
    """
    For each face region (left_cheek, upper_lip, hair, ...), determine which
    SNIC cells belong to that region. Returns dict[region_name -> sorted list
    of cell_ids].

    A cell may appear in multiple regions if regions overlap (e.g. "lips" and
    "upper_lip" both claim a cell — that's by design; the solver picks the
    most-specific region per cell when there is a conflict).
    """
    geometry = precomputed_geometry or compute_snic_cell_geometry(snic_labels)
    result: dict[str, list[int]] = {}
    for name, region in face_regions.items():
        if strategy == "centroid":
            ids = _assign_by_centroid(region.polygon, region.mask, geometry)
        elif strategy == "overlap":
            ids = _assign_by_overlap(region.mask, snic_labels, geometry, overlap_threshold)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        result[name] = ids
    return result


def resolve_cell_to_primary_region(
    cell_assignments: dict[str, list[int]],
    region_priority: list[str] | None = None,
) -> dict[int, str]:
    """
    Inverse of merge_face_regions_with_snic_cells: for each cell ID, return its
    single "primary" region name. When a cell is claimed by multiple regions,
    region_priority breaks the tie (earlier == more specific).

    Default priority puts small/specific regions before large/generic ones, so
    "upper_lip" wins over "lips" wins over "face".
    """
    priority = region_priority or [
        # most specific first
        "left_iris", "right_iris",
        "upper_lip", "lower_lip", "lips",
        "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow",
        "nose",
        "left_temple", "right_temple",
        "left_cheek", "right_cheek",
        "chin", "left_jaw", "right_jaw",
        "forehead",
        "face",
        "hair",
        "background",
    ]
    rank = {n: i for i, n in enumerate(priority)}
    out: dict[int, str] = {}
    for region_name, cell_ids in cell_assignments.items():
        r = rank.get(region_name, 1_000)
        for cid in cell_ids:
            if cid not in out or rank.get(out[cid], 1_000) > r:
                out[cid] = region_name
    return out


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from face_region_mapper import extract_face_regions

    image_path = (sys.argv[1] if len(sys.argv) > 1
                  else str(Path(__file__).parent / "close_emma_2002_2048.jpg"))
    print(f"loading regions for {image_path}")
    regs = extract_face_regions(image_path)

    # Build a synthetic SNIC-like label image (real SNIC will replace this).
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    print("synthesizing SLIC superpixels as a SNIC stand-in (skimage)")
    from skimage.segmentation import slic
    labels = slic(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                  n_segments=600, compactness=10.0, start_label=0).astype(np.int32)
    print(f"got {labels.max() + 1} superpixels")
    assign = merge_face_regions_with_snic_cells(regs, labels, strategy="centroid")
    primary = resolve_cell_to_primary_region(assign)
    print(f"\nRegion -> # cells (centroid strategy):")
    for name in sorted(assign):
        print(f"  {name:14s}  {len(assign[name]):4d} cells")
    print(f"\n{len(primary)} cells received a primary region label")
    unique_primary_regions = set(primary.values())
    print(f"Primary regions used: {sorted(unique_primary_regions)}")
