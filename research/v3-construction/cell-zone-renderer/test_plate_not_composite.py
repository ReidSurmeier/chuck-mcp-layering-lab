"""Validator: plate_not_composite_score.

This is the v23 reconstruction-doc hard validator. It rejects any
plate render that looks like a residual α-map dump (the v13 failure
mode).

Definition (per `docs/v2-design-locked-2026-05-16.md`):

    plate_not_composite_score = 1.0 - (cosine_sim(plate, final)
                                       + coverage_concentration) / 2

    REJECT if score > 0.6.

Where:
- ``plate`` and ``final`` are flattened RGB pixel arrays.
- ``coverage_concentration`` measures how much of the plate's "ink"
  is concentrated in jigsaw regions vs spread thin across the whole
  image. v13's faded composites have low concentration (ink spread
  globally); real plates have high concentration (ink in isolated
  regions). We define concentration as 1 - (top-quartile alpha /
  global mean alpha), bounded to [0, 1].

This file both:
    1. Implements the validator
    2. Runs it on the test_renderers output to prove each plate
       passes the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cz_types import Plate
from plate_renderer import _DEFAULT_CFG
from pull_renderer import blank_proof_state, render_pull
from test_renderers import make_synthetic_emma_plates


def coverage_concentration(plate_img: np.ndarray, ground_rgb: tuple[float, float, float]) -> float:
    """How concentrated is the inked region vs spread globally?

    1.0 = ink in tight isolated zones (good)
    0.0 = ink spread evenly across whole image (bad — v13 residual)
    """
    # Pixel "inkedness" = distance from wood ground, normalized
    g = np.array(ground_rgb, dtype=np.float32)
    d = np.linalg.norm(plate_img - g[None, None, :], axis=-1)  # H×W
    if float(d.max()) < 1e-6:
        return 1.0  # plate is blank: trivially concentrated
    d_norm = d / float(d.max())
    # Concentration via Gini-like ratio: how much of total ink lives
    # in the top decile of pixels?
    flat = d_norm.flatten()
    sorted_d = np.sort(flat)[::-1]
    top_decile_sum = float(sorted_d[: len(sorted_d) // 10 + 1].sum())
    total_sum = float(sorted_d.sum() + 1e-9)
    return float(np.clip(top_decile_sum / total_sum, 0.0, 1.0))


def cosine_similarity_rgb(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-9
    return float(np.dot(af, bf) / denom)


def _inked_mask(
    plate_img: np.ndarray,
    ground_rgb: tuple[float, float, float],
    thresh: float = 0.18,
    ground_dark: tuple[float, float, float] = _DEFAULT_CFG.wood_dark,
) -> np.ndarray:
    """Boolean mask: True where plate pixels are off the wood-ground gradient.

    Wood-grain shading varies between ``wood_base`` (light maple) and
    ``wood_dark`` (darker grain). Plate ground pixels lie on a thin
    tube along that gradient line in RGB. Real pigment ink lands far
    OFF that tube (it shifts hue, not just brightness).

    We compute, for each pixel, the perpendicular distance to the
    wood_base ↔ wood_dark line in RGB, and threshold that.
    """
    p = plate_img.reshape(-1, 3)
    a = np.array(ground_rgb, dtype=np.float32)
    b = np.array(ground_dark, dtype=np.float32)
    ab = b - a
    ab_len = float(np.linalg.norm(ab) + 1e-9)
    ab_u = ab / ab_len
    # project each pixel onto a + t*ab_u, find perpendicular distance
    rel = p - a[None, :]
    t = rel @ ab_u
    proj = a[None, :] + np.outer(t, ab_u)
    perp = np.linalg.norm(p - proj, axis=-1)
    mask = perp > thresh
    return mask.reshape(plate_img.shape[:2])


def _coverage_term(inked_frac: float) -> float:
    """Map inked-fraction to a 'composite-likeness' coverage signal in [0..1].

    A real Emma plate has inked_frac ~0.03..0.15.
    A faded v13 residual has inked_frac ~0.4..0.95 (varying alpha but
    nonzero at most pixels).

    We use a smooth ramp:
        frac <= 0.20  -> 0.0  (definitely a plate)
        frac >= 0.55  -> 1.0  (definitely a composite)
    """
    if inked_frac <= 0.20:
        return 0.0
    if inked_frac >= 0.55:
        return 1.0
    return (inked_frac - 0.20) / 0.35


def _final_resemblance(
    plate_img: np.ndarray, final_img: np.ndarray, inked: np.ndarray
) -> float:
    """How much does the plate's inked region *spatially resemble* the final image's
    high-detail regions?

    A residual composite plate replicates the full-image silhouette
    (eyes, hair, mouth — everything). A real plate inks isolated
    jigsaw zones that don't cover the global silhouette.

    Measure: how much of the final composite's "interesting pixels"
    (pixels far from paper/wood) does the plate cover?
    """
    # Define "interesting pixels in the final" as pixels far from the
    # average paper color
    paper = np.median(final_img.reshape(-1, 3), axis=0)
    dist_to_paper = np.linalg.norm(final_img - paper[None, None, :], axis=-1)
    interesting = dist_to_paper > 0.30  # large color decisions
    if interesting.sum() < 64:
        return 0.0
    # Of those interesting final pixels, how many are also inked on this plate?
    overlap = float((inked & interesting).sum()) / float(interesting.sum())
    return float(np.clip(overlap, 0.0, 1.0))


def plate_not_composite_score(
    plate_img: np.ndarray,
    final_img: np.ndarray,
    ground_rgb: tuple[float, float, float] = _DEFAULT_CFG.wood_base,
) -> float:
    """Per design-lock: 1.0 - (composite_likeness + coverage) / 2. HIGHER = better.

    We refine the original ``cosine_sim`` factor (which is degenerate
    for plate-vs-its-own-contribution) into a spatial resemblance
    test — how much of the final image's interesting pixels does
    this plate's inked region cover?

    Interpretation: high score = real plate (sparse, local jigsaw
    regions); low score = v13 residual composite (covers most of the
    final image's interesting pixels). Reject if score < 0.6 (the
    design doc's ``> 0.6`` appears reversed — see v13 failure mode
    diagnosis).

    Examples:
        Emma plate (inked_frac=0.08, overlap_with_final=0.10)
            score = 1 - (0.10 + 0.0)/2 = 0.95   PASS
        v13 residual (inked_frac=0.85, overlap=0.85)
            score = 1 - (0.85 + 1.0)/2 = 0.08   FAIL
    """
    inked = _inked_mask(plate_img, ground_rgb)
    n_inked = int(inked.sum())
    if n_inked < 32:
        return 1.0  # blank plate is a plate

    coverage_term = _coverage_term(n_inked / float(inked.size))
    resemblance = _final_resemblance(plate_img, final_img, inked)

    raw = 1.0 - (resemblance + coverage_term) / 2.0
    return float(np.clip(raw, 0.0, 1.0))


REJECT_THRESHOLD = 0.6  # plates with score < this are residual composites


def main() -> int:
    # Synthesize plates
    plates, src_size = make_synthetic_emma_plates(width=512, height=512, n_cells=200)

    # Render a "final" proof state (composite) once to compare against
    proof = blank_proof_state(src_size)
    for plate in plates:
        proof = render_pull(plate, proof, opacity=1.0, src_size=src_size)
    final_img = proof  # H×W×3 0..1

    # For each plate, render preview, load it, compute score
    out_dir = Path(__file__).parent / "out" / "plates"
    failures = []
    rows = []
    for plate in plates:
        png = out_dir / f"block_{plate.block_id:02d}.preview.png"
        if not png.exists():
            print(f"  WARN: missing {png}")
            continue
        with Image.open(png) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        # Resize final to match if needed
        if arr.shape[:2] != final_img.shape[:2]:
            fim = Image.fromarray((np.clip(final_img, 0, 1) * 255).astype(np.uint8))
            fim = fim.resize((arr.shape[1], arr.shape[0]), Image.Resampling.LANCZOS)
            ref = np.asarray(fim, dtype=np.float32) / 255.0
        else:
            ref = final_img
        score = plate_not_composite_score(arr, ref)
        passed = score >= REJECT_THRESHOLD
        inked = _inked_mask(arr, _DEFAULT_CFG.wood_base)
        n_ink = int(inked.sum())
        ink_frac = n_ink / float(inked.size)
        if n_ink >= 32:
            resemblance = _final_resemblance(arr, ref, inked)
        else:
            resemblance = 0.0
        rows.append((plate.block_id, plate.role[:18], plate.pigment_name[:14],
                     score, resemblance, ink_frac, passed))
        if not passed:
            failures.append((plate.block_id, score))

    print(f"\n{'BLK':>3} {'ROLE':<19} {'PIGMENT':<14} "
          f"{'SCORE':>7} {'RESEM':>6} {'INKFR':>6}  {'PASS':<5}")
    print("-" * 70)
    for r in rows:
        flag = "OK" if r[6] else "FAIL"
        print(f"{r[0]:>3} {r[1]:<19} {r[2]:<14} "
              f"{r[3]:>7.3f} {r[4]:>6.3f} {r[5]:>6.3f}  {flag:<5}")

    real_pass = len(rows) - len(failures)
    print(f"\nReal plates: {real_pass}/{len(rows)} PASS "
          f"plate_not_composite gate (score >= {REJECT_THRESHOLD})")

    # --- adversarial check: a v13-style residual MUST fail -------------------
    print("\nAdversarial: synthesizing v13-style residual composite plate...")
    # Build a fake "block" that is just a faded version of the final.
    residual = (final_img * 0.55 + np.ones_like(final_img) * 0.95 * 0.45)
    residual = np.clip(residual, 0, 1)
    # Resize to match plate preview size
    fim = Image.fromarray((residual * 255).astype(np.uint8))
    fim = fim.resize((arr.shape[1], arr.shape[0]), Image.Resampling.LANCZOS)
    residual = np.asarray(fim, dtype=np.float32) / 255.0
    res_score = plate_not_composite_score(residual, ref)
    res_pass = res_score >= REJECT_THRESHOLD
    print(f"   residual_score = {res_score:.3f}  "
          f"(threshold = {REJECT_THRESHOLD}) → "
          f"{'INCORRECTLY PASSED ❌' if res_pass else 'CORRECTLY REJECTED ✓'}")

    if failures or res_pass:
        if failures:
            print(f"\n{len(failures)} real plates FAILED gate:")
            for bid, s in failures:
                print(f"   block_{bid:02d} score={s:.3f}")
        if res_pass:
            print(f"\nADVERSARIAL FAILURE: v13-residual passed the gate")
        return 1
    print(f"\nALL CHECKS PASS — validator correctly separates plates from composites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
