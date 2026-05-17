"""Validator 5: underlayer_reversal_check.

Plates are physical wood. They MUST be mirrored (horizontal flip)
relative to the target image because woodblock printing transfers
the inked plate face onto paper in mirror orientation.

Per docs/v2-design-locked-2026-05-16.md row 5:
    "Validate horizontal flip on Plate.svg; validate no flip on Pull[k].png"

This is a boolean pass/fail validator (not a continuous score).

Two checks:
    1. The exported plate SVG carries the horizontal-flip transform.
       We accept either:
         (a) a <g transform="matrix(-1, 0, 0, 1, W, 0)"> wrapping content
         (b) a <g transform="scale(-1, 1) translate(-W, 0)"> equivalent
         (c) explicit `mirror` metadata attribute
    2. The pull preview is NOT flipped vs the source target orientation.
       We compare via cross-correlation: if the pull aligns better with
       the target when flipped, it's wrong-oriented.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

ArrayLike = Union[np.ndarray, str, "Image.Image"]

# Regex tolerant of whitespace and decimals
_FLIP_PATTERNS = [
    # matrix(-1 0 0 1 W 0) — flip-X
    re.compile(r"matrix\s*\(\s*-1[\s,]+0[\s,]+0[\s,]+1[\s,]+([0-9.eE+-]+)[\s,]+0\s*\)"),
    # scale(-1, 1)
    re.compile(r"scale\s*\(\s*-1\s*[,\s]+1\s*\)"),
    # scale(-1) alone
    re.compile(r"scale\s*\(\s*-1\s*\)"),
]
_MIRROR_META = re.compile(r"\bmirror\s*=\s*[\"']?true[\"']?", re.IGNORECASE)


def _to_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, Image.Image):
        arr = np.asarray(x.convert("RGB"))
    else:
        arr = np.asarray(Image.open(x).convert("RGB"))
    return arr.astype(np.float32) / (255.0 if arr.max() > 1.5 else 1.0)


def _plate_svg_is_flipped(svg_path: str) -> tuple[bool, str]:
    """Inspect an SVG for an X-flip transform OR mirror metadata.

    Returns (is_flipped, evidence_string)
    """
    p = Path(svg_path)
    if not p.exists():
        return False, f"SVG file not found: {svg_path}"
    text = p.read_text(errors="ignore")
    for pat in _FLIP_PATTERNS:
        m = pat.search(text)
        if m:
            return True, f"found transform: {m.group(0)}"
    if _MIRROR_META.search(text):
        return True, "found mirror=true metadata attribute"
    return False, "no X-flip transform or mirror metadata found"


def _orientation_alignment_score(target: np.ndarray, pull: np.ndarray) -> dict:
    """Compare cross-correlation of pull-vs-target both ways.

    Returns dict with:
      score_normal: similarity when pull is left as-is
      score_flipped: similarity when pull is X-flipped
      better_orientation: 'normal' or 'flipped'
    """
    from skimage.transform import resize

    # Downsample for speed
    def small(im):
        h, w = im.shape[:2]
        scale = 128 / max(h, w)
        return resize(im, (max(1, int(h * scale)), max(1, int(w * scale))),
                      anti_aliasing=True, preserve_range=True).astype(np.float32)

    t = small(target.mean(axis=-1) if target.ndim == 3 else target)
    p = small(pull.mean(axis=-1) if pull.ndim == 3 else pull)
    p_flip = p[:, ::-1]

    # Pad to same shape if necessary
    if t.shape != p.shape:
        # Resize p to match t
        from skimage.transform import resize as _r
        p = _r(p, t.shape, anti_aliasing=True, preserve_range=True).astype(np.float32)
        p_flip = _r(p_flip, t.shape, anti_aliasing=True, preserve_range=True).astype(np.float32)

    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    s_normal = cos(t, p)
    s_flip = cos(t, p_flip)
    return {
        "score_normal": s_normal,
        "score_flipped": s_flip,
        "better_orientation": "normal" if s_normal >= s_flip else "flipped",
        "delta": float(abs(s_normal - s_flip)),
    }


def check(
    plate_svg_path: str,
    pull_image_path: Optional[ArrayLike] = None,
    target_image_path: Optional[ArrayLike] = None,
    return_components: bool = False,
):
    """Boolean validator: plate flipped AND pull not flipped.

    Args:
        plate_svg_path: path to exported plate SVG.
        pull_image_path: optional path/array of rendered pull image.
            If provided, must also supply target.
        target_image_path: optional path/array of target image.

    Returns:
        True if all checks pass; False otherwise.
        If return_components, returns dict breakdown.
    """
    plate_flipped, plate_evidence = _plate_svg_is_flipped(plate_svg_path)

    pull_orientation = None
    pull_passes = True
    pull_evidence = "skipped (no pull/target supplied)"
    if pull_image_path is not None and target_image_path is not None:
        pull = _to_array(pull_image_path)
        target = _to_array(target_image_path)
        pull_orientation = _orientation_alignment_score(target, pull)
        # Pull NOT flipped means the "normal" orientation matches target better.
        # Require delta >= 0.02 to call it confidently; otherwise borderline = pass.
        delta = pull_orientation["delta"]
        if pull_orientation["better_orientation"] == "flipped" and delta >= 0.02:
            pull_passes = False
            pull_evidence = (
                f"pull aligns better when FLIPPED (delta={delta:.3f}) — "
                "pull was mistakenly mirrored"
            )
        else:
            pull_evidence = (
                f"pull aligns naturally (better_orientation={pull_orientation['better_orientation']}, "
                f"delta={delta:.3f})"
            )

    overall_pass = plate_flipped and pull_passes

    if return_components:
        return {
            "passes": bool(overall_pass),
            "plate_is_flipped": bool(plate_flipped),
            "plate_evidence": plate_evidence,
            "pull_is_normal_orientation": bool(pull_passes),
            "pull_evidence": pull_evidence,
            "pull_orientation_detail": pull_orientation,
        }
    return bool(overall_pass)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: underlayer_reversal_check.py <plate.svg> [<pull.png> <target.png>]")
        sys.exit(1)
    args = sys.argv[1:]
    if len(args) == 1:
        print(check(args[0], return_components=True))
    else:
        print(check(args[0], args[1], args[2], return_components=True))
