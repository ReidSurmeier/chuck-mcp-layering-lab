"""Validator 4: proof_progression_score.

Proof states (cumulative renders at checkpoints 1..7) must add a visible
family of new marks at each step. Per the reconstruction doc:

    "Each proof checkpoint adds >= N pixels of new significant color
     shift" -> if any consecutive pair stalls below threshold, REJECT.

A stalled proof = bad sign: the solver wasn't earning the next pull's
place, or the load-bearing test isn't biting.

Implementation:
    For each consecutive pair (proof_n, proof_{n+1}):
        delta_E_per_pixel = colour.delta_E(proof_n, proof_{n+1}, method='CIE 2000')
        n_shifted = sum(delta_E_per_pixel > MIN_DE_FOR_SHIFT)
        if n_shifted < MIN_SHIFT_PIXELS -> REJECT

Returns:
    overall pass/fail + per-checkpoint shift counts
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

ArrayLike = Union[np.ndarray, str, "Image.Image"]

# Threshold tunables (calibrate later from real proof series)
MIN_DE_FOR_SHIFT = 3.0          # delta-E threshold to count a pixel as "shifted"
MIN_SHIFT_PIXELS_FRACTION = 0.005  # 0.5% of image area minimum per checkpoint
# At 1133x1400 = 1.586M px -> 0.5% = ~7930 px. Reasonable floor for "visible family of marks".

# Performance knob: downsample proof states for ΔE_2000 calc.
# CIEDE2000 is expensive (~200ms per megapixel); 512px-on-longest-edge
# gives 12-16x speedup with negligible accuracy loss for fraction-of-pixels-shifted.
DOWNSAMPLE_LONGEST_EDGE = 384  # 384px on long edge -> ~0.15 Mpx, ~80ms ΔE2000/proof


def _to_lab(x: ArrayLike, downsample_px: int = DOWNSAMPLE_LONGEST_EDGE) -> np.ndarray:
    """Load image and convert to CIELAB (skimage convention: L 0..100).

    Downsamples to `downsample_px` on longest edge for speed using PIL
    (5-10x faster than skimage.resize).
    """
    # Load via PIL with optional thumbnail-style resize
    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr_for_pil = (arr.clip(0, 1 if arr.max() <= 1.5 else 255) * (255 if arr.max() <= 1.5 else 1)).astype(np.uint8)
        else:
            arr_for_pil = arr
        pil = Image.fromarray(arr_for_pil)
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


def _delta_e_2000_pixelwise(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 from skimage if available, else fast approx.

    skimage.color.deltaE_ciede2000 supports per-pixel arrays directly.
    """
    try:
        from skimage.color import deltaE_ciede2000
        return deltaE_ciede2000(lab_a, lab_b)
    except Exception:
        # Fallback: simple Euclidean in Lab (delta_E_76)
        d = lab_a - lab_b
        return np.sqrt((d ** 2).sum(axis=-1))


def score(
    proof_state_sequence: List[ArrayLike],
    min_de_for_shift: float = MIN_DE_FOR_SHIFT,
    min_shift_fraction: float = MIN_SHIFT_PIXELS_FRACTION,
    return_components: bool = False,
):
    """Run proof progression validator.

    Args:
        proof_state_sequence: ordered list of cumulative proof images.
            Length must be >= 2.
        min_de_for_shift: delta-E threshold for counting a shifted pixel.
        min_shift_fraction: minimum fraction of pixels that must shift
            between consecutive proofs.

    Returns:
        progression score in [0, 1] = fraction of intervals that passed.
        Or dict with per-interval breakdown if return_components.
    """
    if len(proof_state_sequence) < 2:
        if return_components:
            return {
                "progression_score": 0.0,
                "passes": False,
                "n_intervals": 0,
                "intervals": [],
                "fail_reason": "Need >= 2 proof states",
            }
        return 0.0

    labs = [_to_lab(p) for p in proof_state_sequence]
    intervals = []
    n_pass = 0
    for k in range(len(labs) - 1):
        a, b = labs[k], labs[k + 1]
        if a.shape != b.shape:
            intervals.append({
                "from": k, "to": k + 1,
                "passes": False,
                "fail_reason": f"shape mismatch {a.shape} vs {b.shape}",
                "shift_pixels": 0,
                "shift_fraction": 0.0,
            })
            continue
        de = _delta_e_2000_pixelwise(a, b)
        shifted = int((de > min_de_for_shift).sum())
        total = int(de.size)
        frac = shifted / max(total, 1)
        passes_interval = frac >= min_shift_fraction
        if passes_interval:
            n_pass += 1
        intervals.append({
            "from": k,
            "to": k + 1,
            "passes": passes_interval,
            "shift_pixels": shifted,
            "total_pixels": total,
            "shift_fraction": float(frac),
            "mean_delta_e": float(de.mean()),
            "p95_delta_e": float(np.percentile(de, 95)),
        })

    n_intervals = max(1, len(labs) - 1)
    progression_score = n_pass / n_intervals
    overall_pass = all(iv.get("passes", False) for iv in intervals)

    if return_components:
        return {
            "progression_score": float(progression_score),
            "passes": bool(overall_pass),
            "n_intervals": n_intervals,
            "n_passed": n_pass,
            "intervals": intervals,
            "min_de_for_shift": min_de_for_shift,
            "min_shift_fraction": min_shift_fraction,
            "fail_reason": (
                None
                if overall_pass
                else f"{n_intervals - n_pass}/{n_intervals} intervals stalled"
            ),
        }
    return float(progression_score)


def passes(proof_state_sequence: List[ArrayLike]) -> bool:
    out = score(proof_state_sequence, return_components=True)
    return bool(out["passes"])


if __name__ == "__main__":
    # Smoke: progression vs stall
    H, W = 200, 200
    base = np.zeros((H, W, 3), dtype=np.uint8) + 240  # nearly white
    a = base.copy()
    b = base.copy(); b[10:60, 10:60] = [180, 60, 60]   # new red mark
    c = b.copy(); c[80:140, 80:140] = [40, 40, 200]    # new blue mark
    d = c.copy(); d[160:190, 160:190] = [10, 10, 10]   # new black
    stall = c.copy()  # same as c — no shift!

    print("GOOD progression:", score([a, b, c, d], return_components=True))
    print("STALLED progression:", score([a, b, c, stall], return_components=True))
