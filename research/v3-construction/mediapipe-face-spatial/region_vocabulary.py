"""
chuck-mcp v3 — Face Region Vocabulary
=====================================

Maps human-readable region names (the words Reid can use in a text prompt:
"blue underlayer under the hair, yellow under the cheek and temple, pink
for the lip line.") to MediaPipe FaceMesh landmark index sets and to
non-mesh region builders (hair, background).

Two flavors per landmark group:
  *_BOUNDARY: ordered ring of indices that traces the region's polygon boundary
              (suitable for Polygon construction)
  *_POINTS:   loose bag of indices inside the region (suitable for centroid /
              convex hull / dilated mask construction)

Landmark indices below are sourced from:
  - mediapipe/python/solutions/face_mesh_connections.py
    (FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,
     FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL,
     FACEMESH_NOSE)
  - Asadullah-Dal17/fd71c31bac74ee84e6a31af50fa62961 (community-verified
    region index lists)
  - sanderdesnaijer.com/blog/mediapipe-face-mesh-landmarks (the 478-point
    interactive explorer; cheek/temple/forehead surface indices)

V1 vocabulary covers 13 named regions. Extending to "above the eyebrow",
"upper cheek", etc. is a documented pattern (see NOTES.md "Extending").
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ----------------------------------------------------------------------------
# 1. Boundary rings — used to build closed polygons.
#    Order matters: traverse the ring clockwise (image coords).
# ----------------------------------------------------------------------------

# Full face outline (jawline + cheek edge + forehead crown).
# 36 indices, ordered around the perimeter.
FACE_OVAL_BOUNDARY = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

# Lips: full outer contour. Upper + lower in one ring.
LIPS_OUTER_BOUNDARY = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# Inner mouth ring (opening).
LIPS_INNER_BOUNDARY = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
]

# Upper-lip-only ring (between outer-upper and inner-upper).
UPPER_LIP_BOUNDARY = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
]

# Lower-lip-only ring.
LOWER_LIP_BOUNDARY = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
]

LEFT_EYE_BOUNDARY = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

RIGHT_EYE_BOUNDARY = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

LEFT_EYEBROW_BOUNDARY = [
    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,
]

RIGHT_EYEBROW_BOUNDARY = [
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
]


# ----------------------------------------------------------------------------
# 2. Region surface point sets — interior/anchor landmarks.
#    Used for centroid, region-anchor lookup, convex hull build.
# ----------------------------------------------------------------------------

# Cheek SURFACE indices (interior, NOT face oval boundary).
# Subject's left (image right) cheek:
LEFT_CHEEK_POINTS = [266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 329, 277, 280, 330, 371]
# Subject's right (image left) cheek:
RIGHT_CHEEK_POINTS = [36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 100, 47, 50, 101, 142]

# Temple SURFACE indices (between eye outer corner and ear / hairline).
# Use a tight cluster, NOT the full forehead ring.
LEFT_TEMPLE_POINTS = [251, 284, 332, 297, 389, 356]
RIGHT_TEMPLE_POINTS = [21, 54, 103, 67, 162, 127]

# Forehead anchors (above the eyebrows; sparse coverage in mediapipe).
FOREHEAD_POINTS = [10, 151, 9, 8, 168, 109, 108, 107, 67, 69, 104, 105,
                   338, 337, 336, 297, 299, 334, 333, 332]

# Chin = lowest point + neighbors.
CHIN_POINTS = [152, 175, 199, 200, 18, 32, 208, 428, 262, 396, 369, 140, 170, 171]

# Jaw line proper (under the cheek, going from ear to chin).
LEFT_JAW_POINTS = [288, 397, 365, 379, 378, 400, 377, 152]
RIGHT_JAW_POINTS = [58, 132, 172, 136, 150, 149, 176, 148, 152]

# Nose anatomy.
NOSE_TIP_POINT = 4
NOSE_BRIDGE_POINTS = [6, 197, 195, 5, 168, 4, 1, 2, 19]
NOSE_LEFT_NOSTRIL = [331, 279, 278, 344, 360]
NOSE_RIGHT_NOSTRIL = [102, 49, 48, 115, 131]
NOSE_POINTS = NOSE_BRIDGE_POINTS + NOSE_LEFT_NOSTRIL + NOSE_RIGHT_NOSTRIL + [NOSE_TIP_POINT]

# Eye interior anchor (iris center, only if refine_landmarks=True).
LEFT_IRIS_POINTS = [468, 469, 470, 471, 472]   # center + 4 ring
RIGHT_IRIS_POINTS = [473, 474, 475, 476, 477]


# ----------------------------------------------------------------------------
# 3. Region metadata — for solver + LLM prompt translation
# ----------------------------------------------------------------------------

RegionKind = Literal[
    "face_mesh",        # built from a closed mesh-landmark ring
    "mesh_hull",        # built from convex hull of a point set
    "outside_face_up",  # face-bounding-box, extended upward (hair region)
    "outside_face_all", # everything outside hair + face (background)
    "mesh_band",        # band along a ring (e.g. "above eyebrow")
]


@dataclass(frozen=True)
class RegionSpec:
    name: str
    kind: RegionKind
    # For face_mesh: list of landmark indices that form the boundary ring.
    boundary: tuple[int, ...] = ()
    # For mesh_hull: list of landmark indices whose convex hull defines region.
    points: tuple[int, ...] = ()
    # Optional dilation (pixels) applied AFTER mask build for soft regions.
    dilate_px: int = 0
    # Optional vertical band offset for mesh_band ("above_eyebrow" etc.).
    band_offset_px: int = 0
    band_height_px: int = 0
    # Human-friendly synonyms — LLM should normalize to canonical name.
    synonyms: tuple[str, ...] = ()


REGION_VOCABULARY: dict[str, RegionSpec] = {
    "face":            RegionSpec("face",            "face_mesh", boundary=tuple(FACE_OVAL_BOUNDARY)),
    "left_cheek":      RegionSpec("left_cheek",      "mesh_hull", points=tuple(LEFT_CHEEK_POINTS),
                                  synonyms=("subject_left_cheek", "cheek_left",
                                            "her_left_cheek", "left side of face")),
    "right_cheek":     RegionSpec("right_cheek",     "mesh_hull", points=tuple(RIGHT_CHEEK_POINTS),
                                  synonyms=("subject_right_cheek", "cheek_right",
                                            "her_right_cheek", "right side of face")),
    "left_temple":     RegionSpec("left_temple",     "mesh_hull", points=tuple(LEFT_TEMPLE_POINTS),
                                  dilate_px=8,
                                  synonyms=("temple_left", "left temple area")),
    "right_temple":    RegionSpec("right_temple",    "mesh_hull", points=tuple(RIGHT_TEMPLE_POINTS),
                                  dilate_px=8,
                                  synonyms=("temple_right", "right temple area")),
    "forehead":        RegionSpec("forehead",        "mesh_hull", points=tuple(FOREHEAD_POINTS),
                                  dilate_px=4,
                                  synonyms=("brow_region", "above the eyebrows", "frontal_bone")),
    "chin":            RegionSpec("chin",            "mesh_hull", points=tuple(CHIN_POINTS),
                                  synonyms=("jaw_tip", "lower chin")),
    "left_jaw":        RegionSpec("left_jaw",        "mesh_hull", points=tuple(LEFT_JAW_POINTS),
                                  dilate_px=4,
                                  synonyms=("jaw_left", "left jawline")),
    "right_jaw":       RegionSpec("right_jaw",       "mesh_hull", points=tuple(RIGHT_JAW_POINTS),
                                  dilate_px=4,
                                  synonyms=("jaw_right", "right jawline")),
    "upper_lip":       RegionSpec("upper_lip",       "face_mesh", boundary=tuple(UPPER_LIP_BOUNDARY),
                                  synonyms=("top lip", "upper lip line")),
    "lower_lip":       RegionSpec("lower_lip",       "face_mesh", boundary=tuple(LOWER_LIP_BOUNDARY),
                                  synonyms=("bottom lip", "lower lip line")),
    "lips":            RegionSpec("lips",            "face_mesh", boundary=tuple(LIPS_OUTER_BOUNDARY),
                                  synonyms=("mouth", "lip line", "the lip")),
    "left_eye":        RegionSpec("left_eye",        "face_mesh", boundary=tuple(LEFT_EYE_BOUNDARY),
                                  synonyms=("eye_left", "her left eye")),
    "right_eye":       RegionSpec("right_eye",       "face_mesh", boundary=tuple(RIGHT_EYE_BOUNDARY),
                                  synonyms=("eye_right", "her right eye")),
    "left_eyebrow":    RegionSpec("left_eyebrow",    "face_mesh", boundary=tuple(LEFT_EYEBROW_BOUNDARY),
                                  dilate_px=6,
                                  synonyms=("eyebrow_left", "left brow")),
    "right_eyebrow":   RegionSpec("right_eyebrow",   "face_mesh", boundary=tuple(RIGHT_EYEBROW_BOUNDARY),
                                  dilate_px=6,
                                  synonyms=("eyebrow_right", "right brow")),
    "nose":            RegionSpec("nose",            "mesh_hull", points=tuple(NOSE_POINTS),
                                  synonyms=("nose bridge", "nasal area")),
    # --- non-mesh regions: built from segmentation + heuristics ---
    "hair":            RegionSpec("hair",            "outside_face_up",
                                  synonyms=("scalp", "the hair", "hairline above")),
    "background":      RegionSpec("background",      "outside_face_all",
                                  synonyms=("backdrop", "outside the face")),
}


def list_supported_regions() -> list[str]:
    """Canonical region names available to the LLM prompt translator."""
    return sorted(REGION_VOCABULARY.keys())


def resolve_region_name(text: str) -> str | None:
    """
    Map a free-form region phrase (from a Reid text prompt) to a canonical
    region name. LLM should pre-normalize; this is a safety net.
    """
    t = text.lower().strip()
    if t in REGION_VOCABULARY:
        return t
    for name, spec in REGION_VOCABULARY.items():
        for syn in spec.synonyms:
            if syn.lower() in t or t in syn.lower():
                return name
    return None


if __name__ == "__main__":
    names = list_supported_regions()
    print(f"V1 vocabulary: {len(names)} regions")
    for n in names:
        spec = REGION_VOCABULARY[n]
        print(f"  {n:18s}  kind={spec.kind:18s}  syns={list(spec.synonyms)}")
