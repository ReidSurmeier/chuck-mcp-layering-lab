"""Validator 1: plate_not_composite_score.

Penalizes a block that looks like the final image.

This is the v13 smoking-gun detector. v13's late blocks (24, 25, 26) were
basically faded full-face composites - residual alpha-map dumps masquerading
as plates. A real plate is a sparse set of inked jigsaw zones, not a faded
copy of the whole portrait.

FORMULA (per docs/v2-design-locked-2026-05-16.md row 1):

    score = 1.0 - (cosine_similarity(downsampled(plate), downsampled(final))
                   + coverage_concentration(plate)) / 2.0

    REJECT if score > 0.6

Interpretation:
- cosine_similarity high  -> plate looks like composite (BAD)
- coverage_concentration high -> ink is spread thin everywhere (BAD)
- We want plates that are LOW-similarity to final AND HIGH-concentration
  (i.e., the inked zones are clustered into a few clean jigsaw regions)

Note on the formula in the design doc: it says
    1.0 - (cos_sim + coverage_concentration) / 2
which means HIGH score = good plate. The doc says "Reject if > 0.6" which
on first read seems backwards. We interpret this as the BUG score in the
codebase: HIGH cos_sim + LOW concentration -> BAD plate. We implement
`badness_score` (high = bad) and apply threshold `> 0.6 -> REJECT`. See NOTES.md.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

ArrayLike = Union[np.ndarray, str, "Image.Image"]

# Threshold from docs/v2-design-locked-2026-05-16.md
REJECT_THRESHOLD = 0.6

# Downsample target — keeps the validator fast (sub-second per block)
# and prevents the cosine from being dominated by high-freq noise.
DOWNSAMPLE_PX = 128


def _to_array(x: ArrayLike, mode: str = "RGB", target_px: int = None) -> np.ndarray:
    """Load to numpy in 0..1 float, optionally downsampled to `target_px` longest edge.

    PIL's resampling is ~5-10x faster than skimage for the same-quality result.
    """
    if isinstance(x, np.ndarray):
        arr = x
        if target_px is not None and max(arr.shape[:2]) > target_px:
            # Round-trip through PIL for fast resize
            pil = Image.fromarray(arr if arr.dtype == np.uint8 else (arr * 255).clip(0, 255).astype(np.uint8))
            h, w = arr.shape[:2]
            scale = target_px / max(h, w)
            pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            arr = np.asarray(pil)
    else:
        if isinstance(x, Image.Image):
            pil = x.convert(mode)
        else:
            pil = Image.open(x).convert(mode)
        if target_px is not None and max(pil.size) > target_px:
            scale = target_px / max(pil.size)
            new_size = (max(1, int(pil.size[0] * scale)), max(1, int(pil.size[1] * scale)))
            pil = pil.resize(new_size, Image.BILINEAR)
        arr = np.asarray(pil)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def _downsample(img: np.ndarray, px: int = DOWNSAMPLE_PX) -> np.ndarray:
    """Resize so longest edge is `px`, preserve aspect.

    Uses PIL (5-10x faster than skimage). Already-downsampled input is
    returned unchanged.
    """
    h, w = img.shape[:2]
    if max(h, w) <= px:
        return img
    scale = px / max(h, w)
    pil = Image.fromarray(
        img if img.dtype == np.uint8 else (img * 255).clip(0, 255).astype(np.uint8)
    )
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    pil = pil.resize(new_size, Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32)
    if out.max() > 1.5:
        out = out / 255.0
    return out


def _to_inked_mask(plate_or_mask: np.ndarray, wood_grain_threshold: float = 0.85) -> np.ndarray:
    """Derive a binary inked-zone mask from a plate preview.

    A plate preview is a wood-grain field with inked zones. Inked = darker
    than the wood field. If the input already looks like a binary mask
    (>90% values near {0,1}), treat it as one directly.
    """
    if plate_or_mask.ndim == 2:
        gray = plate_or_mask
    else:
        gray = plate_or_mask.mean(axis=-1)

    if _looks_binary_mask(gray):
        return (gray > 0.5).astype(np.float32)

    # Otherwise: inked = significantly darker than wood
    return (gray < wood_grain_threshold).astype(np.float32)


def _looks_binary_mask(gray: np.ndarray) -> bool:
    """True when an input is already a white-ink/black-background mask."""
    gray_int = (gray * 10).clip(0, 10).astype(np.int32).ravel()
    bins = np.bincount(gray_int, minlength=11)
    return (bins[0] + bins[-1]) / max(bins.sum(), 1) > 0.9


def _resize_gray_to_shape(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a normalized grayscale image to `shape` using bilinear sampling."""
    if img.shape == shape:
        return img
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32)
    if out.max() > 1.5:
        out = out / 255.0
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Flattened cosine similarity in [-1, 1]; here we expect [0, 1]."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a_flat), np.linalg.norm(b_flat)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (na * nb))


def _coverage_concentration(mask: np.ndarray) -> float:
    """How spread-out is the inked area?

    Returns 1.0 when ink covers everything uniformly (worst — looks like a
    cumulative composite); returns 0.0 when ink is concentrated in tight
    jigsaw zones (best — looks like a real plate).

    Implementation: compare actual mask area fraction against the *spatial
    spread* of inked pixels. If most pixels are "inked" -> spread is high,
    plate is composite-like.
    """
    total = mask.size
    inked = float(mask.sum())
    if inked < 1.0:
        return 0.0  # Empty plate is "concentrated" by definition (nothing to spread)
    area_fraction = inked / total
    # Penalize area fractions > 0.4 hard (a plate should only ink a portion).
    # The map: area_fraction in [0, 0.15] -> very concentrated -> 0.0
    #          area_fraction in [0.4, 1.0] -> fully composite -> 1.0
    return float(np.clip((area_fraction - 0.15) / (0.40 - 0.15), 0.0, 1.0))


def score(
    plate_mask: ArrayLike,
    final_composite: ArrayLike,
    return_components: bool = False,
):
    """Run the plate-not-composite validator.

    Args:
        plate_mask: plate preview image (path, ndarray, or PIL image).
            Can be either an RGB plate preview (wood-grain + inked zones)
            or a binary inked-zone mask.
        final_composite: the final cumulative print composite.
        return_components: if True, return dict with breakdown.

    Returns:
        badness score in [0, 1]. > REJECT_THRESHOLD means REJECT.
        If return_components=True, returns dict with all internal values.
    """
    # Read directly at downsample resolution -- skips a full-res decode
    plate_small = _to_array(plate_mask, target_px=DOWNSAMPLE_PX)
    final_small = _to_array(final_composite, target_px=DOWNSAMPLE_PX)

    # Pixel-similarity in luminance (most direct "does this plate look
    # like the final image" test). For RGB, average channels.
    plate_lum = plate_small.mean(axis=-1) if plate_small.ndim == 3 else plate_small
    final_lum = final_small.mean(axis=-1) if final_small.ndim == 3 else final_small

    # Use downsampled plate for the inked-area calc too -- the absolute
    # fraction is preserved under uniform downsampling and we save
    # 50-100x on a 1133x1400 plate vs full-res.
    inked = _to_inked_mask(plate_small)

    # In a binary validator mask, white is the printed area. In a wood-grain
    # preview, darker pixels are the printed area. Treat those separately so
    # a correct white-ink mask is not scored as "full background signal."
    if _looks_binary_mask(plate_lum):
        plate_signal = inked
    else:
        plate_signal = 1.0 - plate_lum
    final_signal = 1.0 - final_lum
    final_signal = _resize_gray_to_shape(final_signal, plate_signal.shape)

    cos_sim = _cosine_similarity(plate_signal, final_signal)
    cos_sim = max(0.0, cos_sim)  # negative similarity not meaningful here

    concentration_badness = _coverage_concentration(inked)

    # Higher = worse. Both 0..1.
    badness = (cos_sim + concentration_badness) / 2.0

    if return_components:
        return {
            "badness_score": float(badness),
            "passes": bool(badness <= REJECT_THRESHOLD),
            "cosine_similarity": float(cos_sim),
            "coverage_spread_badness": float(concentration_badness),
            "inked_area_fraction": float(inked.mean()),
            "reject_threshold": REJECT_THRESHOLD,
        }
    return float(badness)


def passes(plate_mask: ArrayLike, final_composite: ArrayLike) -> bool:
    """Boolean gate."""
    return score(plate_mask, final_composite) <= REJECT_THRESHOLD


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: plate_not_composite.py <plate.png> <final.png>")
        sys.exit(1)
    out = score(sys.argv[1], sys.argv[2], return_components=True)
    for k, v in out.items():
        print(f"  {k}: {v}")
