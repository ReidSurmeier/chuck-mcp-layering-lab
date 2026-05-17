"""Validator 6: final_match_score.

Color/structure match between target and final composite, after all
~132 pulls. Per docs/v2-design-locked-2026-05-16.md row 6:

    "delta-E_2000 mean across visible regions, advisory not gating"

This validator is ADVISORY ONLY — it reports but does not block
sign-off. (v13 hit dE 4.98 mean and still produced bad plates; that's
why the other 5 validators are the actual gates.)

Returns a dict with:
    delta_e_mean, delta_e_median, delta_e_p95
    plus per-region breakdowns when a region mask is supplied.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

ArrayLike = Union[np.ndarray, str, "Image.Image"]

# Advisory thresholds (NOT gating — informational only)
DE_TARGET_MEAN = 5.0   # Aspirational. v13 hit 4.98 mean and was still broken.
DE_TARGET_P95 = 12.0   # Worst-area tolerance.

# Downsample for speed — CIEDE2000 on full-res 1.5 Mpx takes ~1s
DOWNSAMPLE_LONGEST_EDGE = 512


def _to_lab(x: ArrayLike, downsample_px: int = DOWNSAMPLE_LONGEST_EDGE) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x_for_pil = (x.clip(0, 1 if x.max() <= 1.5 else 255) * (255 if x.max() <= 1.5 else 1)).astype(np.uint8)
        else:
            x_for_pil = x
        pil = Image.fromarray(x_for_pil)
    elif isinstance(x, Image.Image):
        pil = x.convert("RGB")
    else:
        pil = Image.open(x).convert("RGB")
    if max(pil.size) > downsample_px:
        scale = downsample_px / max(pil.size)
        new_size = (max(1, int(pil.size[0] * scale)), max(1, int(pil.size[1] * scale)))
        pil = pil.resize(new_size, Image.BILINEAR)
    rgb = np.asarray(pil).astype(np.float32) / 255.0
    return rgb2lab(rgb)


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    from skimage.transform import resize
    h, w = a.shape[:2]
    b = resize(b, (h, w), anti_aliasing=True, preserve_range=True).astype(a.dtype)
    return a, b


def score(
    target: ArrayLike,
    final_composite: ArrayLike,
    visibility_mask: Optional[np.ndarray] = None,
    region_labels: Optional[np.ndarray] = None,
    return_components: bool = True,  # Always returns detailed dict (advisory)
):
    """Compute ΔE_2000 stats between target and final composite.

    Args:
        target: source target image.
        final_composite: rendered cumulative print after all pulls.
        visibility_mask: optional boolean mask of "visible regions" to score.
            None -> score whole image.
        region_labels: optional integer label map for per-region breakdown.

    Returns:
        dict with delta_e stats and pass/fail (advisory).
    """
    lab_t = _to_lab(target)
    lab_f = _to_lab(final_composite)
    lab_t, lab_f = _resize_to_match(lab_t, lab_f)

    de = deltaE_ciede2000(lab_t, lab_f)  # shape (H, W)

    if visibility_mask is not None:
        m = visibility_mask
        if m.shape != de.shape:
            from skimage.transform import resize
            m = resize(m.astype(float), de.shape, order=0, preserve_range=True) > 0.5
        de_vals = de[m]
    else:
        de_vals = de.flatten()

    if de_vals.size == 0:
        return {
            "delta_e_mean": float("nan"),
            "delta_e_median": float("nan"),
            "delta_e_p95": float("nan"),
            "delta_e_max": float("nan"),
            "advisory_passes": False,
            "advisory_note": "no visible pixels to score",
            "per_region": None,
            "advisory_only": True,
        }

    out = {
        "delta_e_mean": float(de_vals.mean()),
        "delta_e_median": float(np.median(de_vals)),
        "delta_e_p95": float(np.percentile(de_vals, 95)),
        "delta_e_max": float(de_vals.max()),
        "advisory_threshold_mean": DE_TARGET_MEAN,
        "advisory_threshold_p95": DE_TARGET_P95,
        "advisory_passes": (
            float(de_vals.mean()) <= DE_TARGET_MEAN
            and float(np.percentile(de_vals, 95)) <= DE_TARGET_P95
        ),
        "advisory_only": True,
        "n_pixels_scored": int(de_vals.size),
    }

    if region_labels is not None:
        if region_labels.shape != de.shape:
            from skimage.transform import resize
            region_labels = resize(
                region_labels.astype(np.int64), de.shape,
                order=0, preserve_range=True
            ).astype(np.int64)
        unique_regions = np.unique(region_labels)
        per_region = {}
        for r in unique_regions:
            if r == 0:
                continue
            rmask = region_labels == r
            rvals = de[rmask]
            if rvals.size == 0:
                continue
            per_region[int(r)] = {
                "mean": float(rvals.mean()),
                "p95": float(np.percentile(rvals, 95)),
                "n_pixels": int(rvals.size),
            }
        out["per_region"] = per_region
    else:
        out["per_region"] = None

    return out


def passes(target: ArrayLike, final_composite: ArrayLike) -> bool:
    """Advisory pass. Always returns True (does NOT gate sign-off)."""
    score(target, final_composite)
    return True  # Advisory only — never blocks.


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: final_match.py <target.png> <final.png>")
        sys.exit(1)
    out = score(sys.argv[1], sys.argv[2])
    for k, v in out.items():
        print(f"  {k}: {v}")
