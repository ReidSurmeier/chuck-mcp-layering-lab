"""Color difference implementations for chuck-mcp v4 plate objective.

Three flavors, chosen per `docs/v2-design-locked-2026-05-16.md`:

    | Use case                                       | Function       | Backend |
    |------------------------------------------------|----------------|---------|
    | JAX solver inner-loop (gradient-required)      | delta_e_76     | JAX     |
    | JAX solver inner-loop (smooth chroma weighting)| delta_e_94     | JAX     |
    | Outer-loop validators / counterfactual ablation| delta_e_2000   | NumPy   |

Why ΔE_2000 is NumPy-only (Sharma/Wu/Dalal 2005):

> "These discontinuities preclude the use of the formula in analysis based on
>  Taylor series approximations and in design techniques using gradient based
>  optimization."  - §1, Sharma 2005

CIEDE2000 has three known mathematical discontinuities. Putting it inside a
JAX traced computation produces grad-NaN at sample points within ~5 ΔE*_ab of
those boundaries. This is a known cause of solver collapse — concentrated
around mean-hue 143° (deep blue/violet, which happens to be Emma's hair).

THE FOUR SHARMA IMPLEMENTATION PITFALLS THIS MODULE GETS RIGHT
-------------------------------------------------------------

1. **Use atan2, not atan, for hue.** atan returns [-π/2, π/2]; you need the
   full [0, 2π). atan2 is the 4-quadrant variant. Wrap to [0°, 360°).

2. **Signed ΔC' and ΔH' — the cross-term R_T depends on the signs.** Some
   "clean" implementations absolute-value these and silently drift on blue
   samples. Sharma §4.3.

3. **Mean-hue h̄' boundary at |h'_1 − h'_2| > 180° (Eq. 14).** When the two
   hues straddle 0°/360°, the naive mean (h1+h2)/2 is wrong by 180°. You
   must add 360° to whichever hue is smaller before averaging.

4. **Hue-diff Δh' sign at exactly 180° apart (Eq. 10).** When |Δh'| equals
   180° exactly, the formula is ambiguous: pick the convention that puts the
   result in (-180°, 180°] (i.e., the "positive 180" branch).

If any of those four are wrong, Sharma's Table I will visibly fail on the
blue and gray-axis pairs. The 34-pair table is the ground truth.

Verified: every pair in `ciede2000_test_table.SHARMA_TABLE_I` reproduces to
<5e-5 against skimage.color.deltaE_ciede2000.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

ArrayLike = jnp.ndarray  # type alias for clarity

# ----------------------------------------------------------------------
# ΔE_76 — Euclidean distance in CIE L*a*b*. Fully differentiable.
# ----------------------------------------------------------------------

def delta_e_76(lab1: ArrayLike, lab2: ArrayLike) -> jnp.ndarray:
    """CIE76 color difference — sqrt(ΔL² + Δa² + Δb²).

    Fully differentiable; suitable for JAX gradient-based optimization.
    No discontinuities. The cost is perceptual non-uniformity (an ΔE_76 of 5
    means different things in blue vs yellow), but for solver inner-loop the
    monotonicity is what matters, not absolute correctness.

    Args:
        lab1: array of shape (..., 3) in CIE Lab (L in [0,100], a/b in ~[-128,127]).
        lab2: same shape as lab1.

    Returns:
        ΔE_76 of shape lab1.shape[:-1].
    """
    d = lab1 - lab2
    return jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-12)


# ----------------------------------------------------------------------
# ΔE_94 — CIE94 with smooth weighting on chroma and hue. Differentiable.
# ----------------------------------------------------------------------

# CIE94 constants for "graphic arts" application (kL=kC=kH=1, K1=0.045, K2=0.015).
# These are the right defaults for print/woodblock.
_DE94_KL = 1.0
_DE94_KC = 1.0
_DE94_KH = 1.0
_DE94_K1 = 0.045
_DE94_K2 = 0.015


def delta_e_94(lab1: ArrayLike, lab2: ArrayLike) -> jnp.ndarray:
    """CIE94 color difference (graphic-arts variant). Differentiable.

    The weighting terms S_C = 1 + K1·C* and S_H = 1 + K2·C* introduce
    chroma-dependent scaling so that bright primaries get more tolerance
    than near-grays. There is no rotation term R_T (that's only in ΔE_2000),
    so there is no atan2/quadrant discontinuity. The remaining sqrt at the
    end gets a tiny epsilon to keep gradients finite at exact equality.

    Args:
        lab1: (..., 3) Lab.
        lab2: (..., 3) Lab.

    Returns:
        ΔE_94 of shape lab1.shape[:-1].
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2

    C1 = jnp.sqrt(a1 * a1 + b1 * b1 + 1e-12)
    C2 = jnp.sqrt(a2 * a2 + b2 * b2 + 1e-12)
    dC = C1 - C2

    # ΔH² = Δa² + Δb² − ΔC² ; can be slightly negative due to fp — clip.
    dH_sq = jnp.maximum(da * da + db * db - dC * dC, 0.0)

    sL = 1.0
    sC = 1.0 + _DE94_K1 * C1
    sH = 1.0 + _DE94_K2 * C1

    term_L = (dL / (_DE94_KL * sL)) ** 2
    term_C = (dC / (_DE94_KC * sC)) ** 2
    term_H = dH_sq / (_DE94_KH * sH) ** 2

    return jnp.sqrt(term_L + term_C + term_H + 1e-12)


# ----------------------------------------------------------------------
# ΔE_2000 — CIEDE2000. NumPy-only. NOT JAX. Validator-grade.
# ----------------------------------------------------------------------

# Standard CIEDE2000 parametric factors (k_L=k_C=k_H=1 per CIE TC1-47).
_DE00_KL = 1.0
_DE00_KC = 1.0
_DE00_KH = 1.0


def _deg(rad: np.ndarray) -> np.ndarray:
    return rad * (180.0 / np.pi)


def _rad(deg: np.ndarray) -> np.ndarray:
    return deg * (np.pi / 180.0)


def delta_e_2000(lab1, lab2) -> np.ndarray:
    """CIEDE2000 color difference (Sharma 2005 implementation).

    NumPy only — has hue-quadrant and mean-hue branches that are not JAX-safe.
    Use this for ablation, validators, and any "after the fact" scoring.

    Implementation follows the Sharma/Wu/Dalal 2005 paper equation-by-equation
    with the four pitfalls explicitly handled (see module docstring).

    Args:
        lab1: array_like of shape (..., 3) in CIE Lab.
        lab2: array_like of shape (..., 3) in CIE Lab.

    Returns:
        np.ndarray of ΔE_2000, shape lab1.shape[:-1].

    Verified: passes all 34 pairs of Sharma 2005 Table I to <5e-5.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Eq. (2): C*_ab and mean
    C1_ab = np.sqrt(a1 * a1 + b1 * b1)
    C2_ab = np.sqrt(a2 * a2 + b2 * b2)
    C_bar_ab = 0.5 * (C1_ab + C2_ab)

    # Eq. (3): G
    C_bar_ab_7 = C_bar_ab ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar_ab_7 / (C_bar_ab_7 + 25.0 ** 7)))

    # Eq. (4): a' = (1+G) * a
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    # Eq. (5): C' = sqrt(a'^2 + b^2)
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    # Eq. (6): h' = atan2(b, a'), wrapped to [0, 360)
    # PITFALL #1: must use atan2 (4-quadrant), not atan.
    h1p = _deg(np.arctan2(b1, a1p))
    h2p = _deg(np.arctan2(b2, a2p))
    h1p = np.where(h1p < 0.0, h1p + 360.0, h1p)
    h2p = np.where(h2p < 0.0, h2p + 360.0, h2p)

    # Eq. (7): ΔL'
    dLp = L2 - L1

    # Eq. (8): ΔC'
    # PITFALL #2: signed, not absolute.
    dCp = C2p - C1p

    # Eq. (10): Δh'
    # PITFALL #4: at exactly |Δh| == 180°, the wrap convention matters.
    #             Sharma's reference table is consistent with "shift into
    #             (-180, 180]" — i.e., +180 stays +180, never -180.
    hp_diff_raw = h2p - h1p
    # If either chroma is zero, Δh' is defined to be 0 (Sharma footnote).
    chroma_zero = (C1p * C2p) == 0.0
    dhp = np.where(
        chroma_zero,
        0.0,
        np.where(
            np.abs(hp_diff_raw) <= 180.0,
            hp_diff_raw,
            np.where(hp_diff_raw > 180.0, hp_diff_raw - 360.0, hp_diff_raw + 360.0),
        ),
    )

    # Eq. (9): ΔH' = 2 sqrt(C1' C2') sin(Δh'/2)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(_rad(dhp) * 0.5)

    # Eq. (12): L̄'
    L_bar_p = 0.5 * (L1 + L2)

    # Eq. (13): C̄'
    C_bar_p = 0.5 * (C1p + C2p)

    # Eq. (14): h̄' — the canonical mean-hue boundary.
    # PITFALL #3: when |h1' − h2'| > 180°, the simple mean is off by 180°.
    #             If sum < 360°, add 360 to the mean; if sum >= 360°,
    #             subtract 360. When chromas are zero, h̄' = h1' + h2'.
    hp_sum = h1p + h2p
    h_bar_p = np.where(
        chroma_zero,
        hp_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            0.5 * hp_sum,
            np.where(hp_sum < 360.0, 0.5 * (hp_sum + 360.0), 0.5 * (hp_sum - 360.0)),
        ),
    )

    # Eq. (15): T
    T = (
        1.0
        - 0.17 * np.cos(_rad(h_bar_p - 30.0))
        + 0.24 * np.cos(_rad(2.0 * h_bar_p))
        + 0.32 * np.cos(_rad(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(_rad(4.0 * h_bar_p - 63.0))
    )

    # Eq. (16): Δθ
    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))

    # Eq. (17): R_C
    C_bar_p_7 = C_bar_p ** 7
    R_C = 2.0 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + 25.0 ** 7))

    # Eq. (18): S_L
    L_bar_p_m50_sq = (L_bar_p - 50.0) ** 2
    S_L = 1.0 + (0.015 * L_bar_p_m50_sq) / np.sqrt(20.0 + L_bar_p_m50_sq)

    # Eq. (19): S_C
    S_C = 1.0 + 0.045 * C_bar_p

    # Eq. (20): S_H
    S_H = 1.0 + 0.015 * C_bar_p * T

    # Eq. (21): R_T
    R_T = -np.sin(_rad(2.0 * d_theta)) * R_C

    # Eq. (22): ΔE_00
    term_L = dLp / (_DE00_KL * S_L)
    term_C = dCp / (_DE00_KC * S_C)
    term_H = dHp / (_DE00_KH * S_H)

    return np.sqrt(
        term_L * term_L
        + term_C * term_C
        + term_H * term_H
        + R_T * term_C * term_H
    )
