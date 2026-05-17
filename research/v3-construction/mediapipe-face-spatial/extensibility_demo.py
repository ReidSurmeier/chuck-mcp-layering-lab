"""
chuck-mcp v3 — Extensibility patterns.

How to add a new face region the V1 vocabulary doesn't cover. Three
patterns demonstrated:

  1. mesh_band   — "above the eyebrow" — vertical offset of an existing ring
  2. mesh_hull   — "between the eyes" — convex hull of two existing point sets
  3. derived     — "lower_face_third" — proportional split of an existing region

Each pattern returns a new FaceRegion that plugs into the existing
merge_face_regions_with_snic_cells() pipeline.
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from face_region_mapper import (
    extract_face_regions, FaceRegion,
    _detect_facemesh_landmarks, _polygon_to_mask,
    _ring_to_polygon, _points_to_convex_hull,
    _centroid_of_mask, _bbox_of_mask,
)
from region_vocabulary import (
    LEFT_EYEBROW_BOUNDARY, RIGHT_EYEBROW_BOUNDARY,
    LEFT_EYE_BOUNDARY, RIGHT_EYE_BOUNDARY,
    LEFT_CHEEK_POINTS, RIGHT_CHEEK_POINTS,
)


def add_above_eyebrow(image_bgr, landmarks, side: str = "left",
                      band_px: int = 60) -> FaceRegion:
    """
    Build a band region above an eyebrow by translating its ring upward.
    """
    h, w = image_bgr.shape[:2]
    ring = LEFT_EYEBROW_BOUNDARY if side == "left" else RIGHT_EYEBROW_BOUNDARY
    pts = []
    # Top arc: shifted up by band_px
    for i in ring:
        pts.append((int(landmarks[i, 0]),
                    int(landmarks[i, 1]) - band_px))
    # Bottom arc: the eyebrow itself, in reverse
    for i in reversed(ring):
        pts.append((int(landmarks[i, 0]), int(landmarks[i, 1])))
    mask = _polygon_to_mask(pts, h, w, dilate_px=2)
    return FaceRegion(
        name=f"above_{side}_eyebrow", polygon=pts, mask=mask,
        source="derived:mesh_band(offset)", confidence=0.7,
        centroid=_centroid_of_mask(mask), bbox=_bbox_of_mask(mask),
    )


def add_between_eyes(image_bgr, landmarks) -> FaceRegion:
    """
    Build the 'glabella' region — between the inner corners of the eyes.
    Uses 4 indices: 168 (top of nose bridge), 6 (mid bridge), 122 + 351
    (left/right inner brow ends).
    """
    h, w = image_bgr.shape[:2]
    # 168, 6, 197 = upper bridge, 122 + 351 = inner brow tips
    region_pts = (168, 6, 197, 351, 285, 8, 55, 122)
    poly = _points_to_convex_hull(landmarks, region_pts)
    mask = _polygon_to_mask(poly, h, w, dilate_px=4)
    return FaceRegion(
        name="between_eyes", polygon=poly, mask=mask,
        source="derived:mesh_hull(custom)", confidence=0.8,
        centroid=_centroid_of_mask(mask), bbox=_bbox_of_mask(mask),
    )


def add_lower_face_third(image_bgr, regions: dict) -> FaceRegion:
    """
    Build a 'lower face third' by intersecting the face region with the
    bottom third of its bbox.
    """
    h, w = image_bgr.shape[:2]
    face = regions["face"]
    x0, y0, x1, y1 = face.bbox
    third_top = y0 + 2 * (y1 - y0) // 3
    region_mask = np.zeros_like(face.mask)
    region_mask[third_top:y1 + 1, x0:x1 + 1] = 255
    region_mask = cv2.bitwise_and(region_mask, face.mask)
    return FaceRegion(
        name="lower_face_third", polygon=None, mask=region_mask,
        source="derived:proportional_split(face)", confidence=0.8,
        centroid=_centroid_of_mask(region_mask), bbox=_bbox_of_mask(region_mask),
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(_HERE / "close_emma_2002_2048.jpg")
    print(f"Demo: extending the vocabulary on {path}")
    image_bgr = cv2.imread(path)
    regs = extract_face_regions(path)
    landmarks, strat = _detect_facemesh_landmarks(image_bgr)
    if landmarks is None:
        print("Landmark detection failed — cannot demo extensions.")
        sys.exit(1)

    extras = {
        "above_left_eyebrow":  add_above_eyebrow(image_bgr, landmarks, "left",  band_px=80),
        "above_right_eyebrow": add_above_eyebrow(image_bgr, landmarks, "right", band_px=80),
        "between_eyes":        add_between_eyes(image_bgr, landmarks),
        "lower_face_third":    add_lower_face_third(image_bgr, regs),
    }
    print("\nNew regions:")
    for n, r in extras.items():
        print(f"  {n:22s}  src={r.source:34s}  conf={r.confidence:.2f}  "
              f"px={int((r.mask > 0).sum()):>8d}  bbox={r.bbox}")

    # Render
    out = image_bgr.copy()
    palette = {
        "above_left_eyebrow":  (60, 200, 80),
        "above_right_eyebrow": (60, 200, 80),
        "between_eyes":        (40, 100, 230),
        "lower_face_third":    (200, 60, 200),
    }
    for n, r in extras.items():
        layer = np.zeros_like(out)
        layer[r.mask > 0] = palette[n]
        out = cv2.addWeighted(out, 1.0, layer, 0.5, 0)
        if r.polygon is not None:
            pts = np.array(r.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=palette[n], thickness=3)
    out_path = _HERE / "emma_extensibility_demo.png"
    cv2.imwrite(str(out_path), out)
    print(f"wrote {out_path}")
