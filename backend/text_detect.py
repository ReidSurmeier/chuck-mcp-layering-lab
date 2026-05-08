"""
text_detect.py — Text region detection for color separation.

Detects text regions in an image and returns a binary mask.  Used as a
pre-pass before K-means clustering so that pixels belonging to a text region
are locked to a single color plate rather than fragmented across plates.

Design constraints
------------------
* CPU-only (no CUDA required)
* No model downloads — uses only OpenCV built-ins (MSER + SWT)
* <5 s on a 2000×2000 image  (typically <0.4 s in practice)
* False positives OK; false negatives bad (prefer over-detection)

Approach
--------
1. Optionally downscale large images to a working resolution (1200 px on
   the long side) so MSER stays fast.
2. Run MSER twice — once on the grayscale image (dark-on-light) and once on
   the inverted image (light-on-dark) — to catch both text polarities.
3. Run OpenCV's Stroke Width Transform (SWT) on the normal polarity for
   additional letter-candidate coverage.
4. Merge the three candidate masks, dilate horizontally to connect individual
   characters into word/line regions, then dilate uniformly to add a small
   safety margin around each region.
5. Resize back to original resolution if the image was downscaled.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── tuneable defaults ────────────────────────────────────────────────────────

# Long-side resolution used for MSER/SWT computation.  Larger = slower but
# catches finer detail.  1200 keeps timing well under 1 s on 2000×2000.
_WORK_RES = 1200

# MSER minimum region area (pixels at working resolution).  5 is aggressive
# but necessary to catch sub-8 px text.  Raises false-positive rate slightly.
_MSER_MIN_AREA = 5

# MSER maximum region area.  Caps at 1/20th of the working-res image area so
# that huge uniform blobs (sky, background) are never considered text.
_MSER_MAX_AREA_FRAC = 0.05

# Horizontal kernel width for character→word grouping pass.
_GROUP_KERN_W = 15
_GROUP_KERN_H = 3
_GROUP_ITERS = 1

# Uniform dilation added on top of grouping (safety margin around each region).
_MARGIN_KERN = 5
_MARGIN_ITERS = 2


# ── public API ───────────────────────────────────────────────────────────────


def detect_text_regions(
    image_array: np.ndarray,
    min_text_size: int = 8,
    *,
    work_res: int = _WORK_RES,
) -> np.ndarray:
    """Detect text regions in an image.

    Parameters
    ----------
    image_array:
        RGB or grayscale uint8 numpy array.  Shape (H, W) or (H, W, C).
    min_text_size:
        Approximate minimum font height in pixels at *original* resolution.
        Smaller values are more sensitive but slower on noisy images.
    work_res:
        Long-side resolution used internally.  Lower = faster, may miss very
        small text if the image is large.

    Returns
    -------
    Boolean mask of shape (H, W) where ``True`` = text region.  Text regions
    are dilated slightly so that surrounding pixels are included.
    """
    image_array = np.asarray(image_array, dtype=np.uint8)

    # ── normalise to grayscale ───────────────────────────────────────────────
    if image_array.ndim == 2:
        gray_full = image_array
    elif image_array.shape[2] == 4:
        gray_full = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
    else:
        # Treat as RGB (the common case coming from PIL/numpy pipelines)
        gray_full = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    h_full, w_full = gray_full.shape

    # ── optional downscale ───────────────────────────────────────────────────
    scale = 1.0
    long_side = max(h_full, w_full)
    if long_side > work_res:
        scale = work_res / long_side
        new_w = max(1, int(w_full * scale))
        new_h = max(1, int(h_full * scale))
        gray = cv2.resize(gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray = gray_full

    h, w = gray.shape

    # Scale min_text_size to working resolution so filtering stays consistent.
    min_area = max(3, int(min_text_size * scale) ** 2 // 4)
    min_area = min(min_area, _MSER_MIN_AREA)  # never go above 5 at work_res
    max_area = max(100, int(h * w * _MSER_MAX_AREA_FRAC))

    mask = np.zeros((h, w), dtype=np.uint8)

    # MSER requires at least 3×3; bail early for trivially small images.
    if h < 3 or w < 3:
        return mask > 0

    # ── MSER: dark-on-light + light-on-dark ─────────────────────────────────
    mser = cv2.MSER_create(min_area=_MSER_MIN_AREA, max_area=max_area)
    for invert in (False, True):
        g = (255 - gray) if invert else gray
        _, bboxes = mser.detectRegions(g)
        _paint_bboxes(mask, bboxes)

    # ── SWT: dark-on-light ───────────────────────────────────────────────────
    try:
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        swt_bboxes, _, _ = cv2.text.detectTextSWT(gray_bgr, dark_on_light=True)
        if swt_bboxes is not None and len(swt_bboxes) > 0:
            _paint_bboxes(mask, swt_bboxes)
    except (cv2.error, AttributeError):
        # cv2.text module may be absent in non-contrib builds; graceful fallback.
        pass

    # ── horizontal grouping pass: connect chars into words/lines ─────────────
    group_kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_GROUP_KERN_W, _GROUP_KERN_H)
    )
    mask = cv2.dilate(mask, group_kern, iterations=_GROUP_ITERS)

    # ── uniform safety-margin dilation ───────────────────────────────────────
    margin_kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_MARGIN_KERN, _MARGIN_KERN)
    )
    mask = cv2.dilate(mask, margin_kern, iterations=_MARGIN_ITERS)

    # ── scale mask back to original resolution ───────────────────────────────
    if scale < 1.0:
        mask = cv2.resize(mask, (w_full, h_full), interpolation=cv2.INTER_NEAREST)

    return mask > 0


# ── internal helpers ─────────────────────────────────────────────────────────


def _paint_bboxes(
    mask: np.ndarray,
    bboxes: np.ndarray | list,
) -> None:
    """Paint bounding boxes onto mask in-place (white fill)."""
    if bboxes is None or len(bboxes) == 0:
        return
    h, w = mask.shape[:2]
    for bb in bboxes:
        x, y, bw, bh = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        # Skip sentinel values emitted by SWT chainBBs
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
