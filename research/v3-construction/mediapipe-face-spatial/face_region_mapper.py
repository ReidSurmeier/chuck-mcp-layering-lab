"""
chuck-mcp v3 — Face Region Mapper
=================================

Single entry point:

    regions = extract_face_regions(image_path)

`regions` is a dict[str, FaceRegion] keyed by canonical region name
(see region_vocabulary.REGION_VOCABULARY). Each FaceRegion has:
  - polygon: list of (x, y) in pixel coords (None for non-mesh regions)
  - mask:    HxW uint8 binary mask in pixel coords (0/255)
  - source:  "mediapipe_facemesh" | "selfie_multiclass" | "fallback_heuristic"
  - confidence: float in [0, 1]

Hair and background are NOT in mediapipe FaceMesh. They are built from
the multi-class selfie segmenter (categories: 0=bg, 1=hair, 2=body-skin,
3=face-skin, 4=clothes, 5=others). If segmenter is unavailable, we fall
back to the face-bbox-extension heuristic.

If FaceMesh fails to detect a face (rare for Chuck Close-style head-on
portraits, common for heavily abstracted prints), we fall back to a
"centered portrait" assumption with proportional regions. The caller
gets `source="fallback_heuristic"` and `confidence=0.3`.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon as ShPolygon, MultiPoint

# Local imports
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from region_vocabulary import (
    REGION_VOCABULARY, RegionSpec, FACE_OVAL_BOUNDARY,
    LEFT_CHEEK_POINTS, RIGHT_CHEEK_POINTS,
)


# ----------------------------------------------------------------------------
# Output dataclass
# ----------------------------------------------------------------------------

@dataclass
class FaceRegion:
    name: str
    polygon: Optional[list[tuple[int, int]]]   # closed, pixel coords
    mask: np.ndarray                            # uint8, 0/255, shape HxW
    source: str
    confidence: float
    # Optional metadata
    centroid: Optional[tuple[float, float]] = None
    bbox: Optional[tuple[int, int, int, int]] = None  # xmin,ymin,xmax,ymax


# ----------------------------------------------------------------------------
# MediaPipe lazy-imports (so this module imports cleanly even if mediapipe
# is missing — degrades to fallback).
# ----------------------------------------------------------------------------

_MP_AVAILABLE = False
_MP_TASKS_AVAILABLE = False
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        _MP_TASKS_AVAILABLE = True
    except Exception:
        _MP_TASKS_AVAILABLE = False
except Exception:
    mp = None


# Path to the multi-class selfie segmenter model.
SEG_MODEL_PATH = _HERE / "models" / "selfie_multiclass_256x256.tflite"
# Path to the FaceLandmarker .task bundle (Tasks API replacement for legacy
# FaceMesh).
FACE_LANDMARKER_PATH = _HERE / "models" / "face_landmarker.task"

# Segmentation class indices (from MediaPipe docs).
SEG_BG = 0
SEG_HAIR = 1
SEG_BODY_SKIN = 2
SEG_FACE_SKIN = 3
SEG_CLOTHES = 4
SEG_OTHERS = 5


# ----------------------------------------------------------------------------
# Core: mesh-landmark detection
# ----------------------------------------------------------------------------

def _run_landmarker_once(image_bgr: np.ndarray, conf: float) -> Optional[np.ndarray]:
    """Run FaceLandmarker on a (H,W,3) BGR image. Returns (N,2) pixel coords or None."""
    if not (_MP_TASKS_AVAILABLE and FACE_LANDMARKER_PATH.exists()):
        return None
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    base = mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=conf,
        min_face_presence_confidence=conf,
        min_tracking_confidence=conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    arr = np.zeros((len(lm), 2), dtype=np.float32)
    for i, p in enumerate(lm):
        arr[i, 0] = p.x * w
        arr[i, 1] = p.y * h
    return arr


def _detect_facemesh_landmarks(image_bgr: np.ndarray
                               ) -> tuple[Optional[np.ndarray], str]:
    """
    Returns ((N,2) landmarks in ORIGINAL pixel coords, strategy_label) or (None, "").
    N = 468 (no iris) or 478 (with iris).

    Detection cascade — applies fallback strategies for stylized portraits
    (Chuck Close dot-mosaic, woodblock prints, line drawings):
      1. raw image at multiple confidence levels
      2. gaussian-blurred image (smooths brush-dot mosaics — required for
         Chuck Close 2002 Emma; without this, detection fails)
      3. blurred + downscaled (smaller, smoother — last resort)

    When detection succeeds on a transformed image, landmarks are rescaled
    back to original pixel coordinates so masks line up with the original.
    """
    if not (_MP_TASKS_AVAILABLE and FACE_LANDMARKER_PATH.exists()):
        return None, ""

    H, W = image_bgr.shape[:2]

    strategies = [
        # (transform_fn, label)
        (lambda im: im, "raw"),
        (lambda im: cv2.GaussianBlur(im, (21, 21), 0), "gauss21"),
        (lambda im: cv2.GaussianBlur(im, (41, 41), 0), "gauss41"),
        (lambda im: cv2.resize(cv2.GaussianBlur(im, (21, 21), 0),
                               (512, int(512 * H / W))), "gauss21_down512"),
    ]
    for transform, label in strategies:
        timg = transform(image_bgr)
        for conf in (0.5, 0.2, 0.05):
            lm = _run_landmarker_once(timg, conf)
            if lm is None:
                continue
            # Rescale landmarks back to original pixel space (transforms are
            # blur (preserves coords) or downscale (changes coords)).
            th, tw = timg.shape[:2]
            sx, sy = W / tw, H / th
            lm[:, 0] *= sx
            lm[:, 1] *= sy
            return lm, f"{label}@conf={conf}"
    return None, ""


# ----------------------------------------------------------------------------
# Multi-class selfie segmentation (hair / face-skin / background)
# ----------------------------------------------------------------------------

def _segment_multiclass(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns HxW uint8 category mask (values 0..5) or None on failure.
    """
    if not (_MP_TASKS_AVAILABLE and SEG_MODEL_PATH.exists()):
        return None
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    base = mp_tasks.BaseOptions(model_asset_path=str(SEG_MODEL_PATH))
    opts = mp_vision.ImageSegmenterOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    with mp_vision.ImageSegmenter.create_from_options(opts) as segmenter:
        result = segmenter.segment(mp_image)
    if result.category_mask is None:
        return None
    cat = result.category_mask.numpy_view()  # uint8, HxW or HxWx1
    cat = np.squeeze(cat)
    cat_full = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)
    return cat_full


# ----------------------------------------------------------------------------
# Polygon / mask construction
# ----------------------------------------------------------------------------

def _ring_to_polygon(landmarks: np.ndarray, ring: tuple[int, ...]) -> list[tuple[int, int]]:
    pts = [(int(round(landmarks[i, 0])), int(round(landmarks[i, 1]))) for i in ring]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _points_to_convex_hull(landmarks: np.ndarray, idxs: tuple[int, ...]) -> list[tuple[int, int]]:
    pts = [(float(landmarks[i, 0]), float(landmarks[i, 1])) for i in idxs]
    hull = MultiPoint(pts).convex_hull
    if hull.geom_type == "Polygon":
        coords = list(hull.exterior.coords)
    elif hull.geom_type == "LineString":
        coords = list(hull.coords) + [hull.coords[0]]
    else:
        coords = pts + [pts[0]]
    return [(int(round(x)), int(round(y))) for x, y in coords]


def _polygon_to_mask(polygon: list[tuple[int, int]], h: int, w: int,
                     dilate_px: int = 0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    if dilate_px > 0:
        k = max(1, dilate_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _bbox_of_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _centroid_of_mask(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ----------------------------------------------------------------------------
# Fallback: heuristic centered-portrait regions
# ----------------------------------------------------------------------------

def _build_fallback_regions(h: int, w: int) -> dict[str, FaceRegion]:
    """
    Used when MediaPipe finds no face. Assumes a centered head-on portrait
    that occupies the middle ~60% of the frame, head ~70% of frame height.
    """
    out: dict[str, FaceRegion] = {}

    cy = int(h * 0.52)
    cx = int(w * 0.50)
    face_w = int(w * 0.55)
    face_h = int(h * 0.75)

    # Face = ellipse
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(face_mask, (cx, cy), (face_w // 2, face_h // 2),
                0, 0, 360, 255, -1)
    out["face"] = FaceRegion(
        name="face", polygon=None, mask=face_mask,
        source="fallback_heuristic", confidence=0.3,
        centroid=_centroid_of_mask(face_mask),
        bbox=_bbox_of_mask(face_mask),
    )

    # Hair = top crown extending above face
    hair_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(hair_mask, (cx, cy - face_h // 4),
                (int(face_w * 0.6), int(face_h * 0.45)), 0, 180, 360, 255, -1)
    hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_mask))
    out["hair"] = FaceRegion(
        name="hair", polygon=None, mask=hair_mask,
        source="fallback_heuristic", confidence=0.2,
        centroid=_centroid_of_mask(hair_mask),
        bbox=_bbox_of_mask(hair_mask),
    )

    head_mask = cv2.bitwise_or(face_mask, hair_mask)
    bg_mask = cv2.bitwise_not(head_mask)
    out["background"] = FaceRegion(
        name="background", polygon=None, mask=bg_mask,
        source="fallback_heuristic", confidence=0.3,
        centroid=_centroid_of_mask(bg_mask),
        bbox=_bbox_of_mask(bg_mask),
    )

    # Sub-regions (left/right cheek, forehead, chin) as box thirds
    third = face_h // 5
    sub_rects = {
        "forehead":   (cx - face_w // 3, cy - face_h // 2,        cx + face_w // 3, cy - face_h // 2 + third),
        "left_eye":   (cx,               cy - face_h // 6,        cx + face_w // 3, cy - face_h // 6 + third // 2),
        "right_eye":  (cx - face_w // 3, cy - face_h // 6,        cx,               cy - face_h // 6 + third // 2),
        "left_cheek": (cx,               cy,                       cx + face_w // 3, cy + third),
        "right_cheek":(cx - face_w // 3, cy,                       cx,               cy + third),
        "nose":       (cx - face_w // 8, cy - face_h // 8,         cx + face_w // 8, cy + face_h // 8),
        "lips":       (cx - face_w // 4, cy + third,               cx + face_w // 4, cy + third + third // 2),
        "chin":       (cx - face_w // 4, cy + 2 * third,           cx + face_w // 4, cy + face_h // 2),
    }
    for name, (x0, y0, x1, y1) in sub_rects.items():
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(m, (max(0, x0), max(0, y0)),
                      (min(w - 1, x1), min(h - 1, y1)), 255, -1)
        m = cv2.bitwise_and(m, face_mask)
        out[name] = FaceRegion(
            name=name, polygon=None, mask=m,
            source="fallback_heuristic", confidence=0.2,
            centroid=_centroid_of_mask(m),
            bbox=_bbox_of_mask(m),
        )

    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def extract_face_regions(image_path: str | os.PathLike) -> dict[str, FaceRegion]:
    """
    Detect face landmarks + (optional) selfie segmentation, build the
    full region vocabulary as a dict[name -> FaceRegion].

    On failure (no face): returns heuristic fallback regions with
    confidence ≤ 0.3.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h, w = image_bgr.shape[:2]

    landmarks, strategy = _detect_facemesh_landmarks(image_bgr)
    if landmarks is None:
        # ---- fallback path ----
        regs = _build_fallback_regions(h, w)
        for r in regs.values():
            r.source = "fallback_heuristic"
        return regs
    # Confidence trims for stylized-input strategies (less trust than 'raw').
    conf_for_source = 0.92
    if "gauss" in strategy:
        conf_for_source = 0.78
    if "down" in strategy:
        conf_for_source = 0.70

    out: dict[str, FaceRegion] = {}

    # 1. Build face_mesh + mesh_hull regions
    for name, spec in REGION_VOCABULARY.items():
        if spec.kind == "face_mesh":
            poly = _ring_to_polygon(landmarks, spec.boundary)
            mask = _polygon_to_mask(poly, h, w, dilate_px=spec.dilate_px)
        elif spec.kind == "mesh_hull":
            poly = _points_to_convex_hull(landmarks, spec.points)
            mask = _polygon_to_mask(poly, h, w, dilate_px=spec.dilate_px)
        else:
            continue  # outside_face_*  handled below
        out[name] = FaceRegion(
            name=name, polygon=poly, mask=mask,
            source=f"mediapipe_facemesh[{strategy}]",
            confidence=conf_for_source,
            centroid=_centroid_of_mask(mask),
            bbox=_bbox_of_mask(mask),
        )

    # 2. Hair + background.
    #    Strategy: use the multi-class segmenter only when its output is
    #    plausible (hair pixels overlap with the bbox-extended-up region).
    #    On stylized portraits (Chuck Close mosaic), the segmenter often
    #    labels everything "clothes" or "others" — useless. In that case,
    #    fall back to the bbox-extension heuristic.
    seg = _segment_multiclass(image_bgr)
    face_oval_poly = _ring_to_polygon(landmarks, tuple(FACE_OVAL_BOUNDARY))
    face_oval_mask = _polygon_to_mask(face_oval_poly, h, w)
    x0, y0, x1, y1 = _bbox_of_mask(face_oval_mask) or (0, 0, w - 1, h - 1)
    hh, ww = y1 - y0, x1 - x0
    hair_zone = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(hair_zone,
                  (max(0, x0 - int(ww * 0.30)), max(0, y0 - int(hh * 0.75))),
                  (min(w - 1, x1 + int(ww * 0.30)), y0 + int(hh * 0.20)),
                  255, -1)
    hair_zone = cv2.bitwise_and(hair_zone, cv2.bitwise_not(face_oval_mask))

    seg_hair_plausible = False
    if seg is not None:
        cand_hair = ((seg == SEG_HAIR).astype(np.uint8)) * 255
        overlap = np.logical_and(cand_hair > 0, hair_zone > 0).sum()
        if overlap > 1000 and cand_hair.sum() > 0:
            seg_hair_plausible = True

    if seg_hair_plausible:
        hair_mask = ((seg == SEG_HAIR).astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        hair_src = "selfie_multiclass"
        hair_conf = 0.85
    else:
        hair_mask = hair_zone
        hair_src = "bbox_extend_heuristic"
        hair_conf = 0.5

    # Background = whatever isn't face or hair.
    head_mask = cv2.bitwise_or(face_oval_mask, hair_mask)
    bg_mask = cv2.bitwise_not(head_mask)
    # If segmenter gave us a plausible bg, intersect with it for cleaner edges.
    if seg is not None:
        cand_bg = ((seg == SEG_BG).astype(np.uint8)) * 255
        if cand_bg.sum() > h * w * 0.05:  # at least 5% of image looks like bg
            bg_mask = cv2.bitwise_and(bg_mask, cand_bg)
            bg_src = "selfie_multiclass+bbox_complement"
            bg_conf = 0.8
        else:
            bg_src = "bbox_complement"
            bg_conf = 0.6
    else:
        bg_src = "bbox_complement"
        bg_conf = 0.5

    out["hair"] = FaceRegion(
        name="hair", polygon=None, mask=hair_mask,
        source=hair_src, confidence=hair_conf,
        centroid=_centroid_of_mask(hair_mask),
        bbox=_bbox_of_mask(hair_mask),
    )
    out["background"] = FaceRegion(
        name="background", polygon=None, mask=bg_mask,
        source=bg_src, confidence=bg_conf,
        centroid=_centroid_of_mask(bg_mask),
        bbox=_bbox_of_mask(bg_mask),
    )
    return out


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(_HERE / "close_emma_2002_2048.jpg")
    print(f"Extracting face regions from {path}")
    regs = extract_face_regions(path)
    print(f"Got {len(regs)} regions:")
    for name, r in regs.items():
        n_pixels = int((r.mask > 0).sum())
        print(f"  {name:14s}  src={r.source:20s}  conf={r.confidence:.2f}  "
              f"px={n_pixels:>9d}  bbox={r.bbox}")
