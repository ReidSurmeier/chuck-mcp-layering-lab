"""Kubelka-Munk forward renderer using mixbox latent blending.

Stacks N translucent pigment alpha maps on a substrate via mixbox's 7-dim
latent space — physically plausible blending without the Kubelka-Munk
parameter fitting headache. This is the master gate for the decomposition
pipeline: if forward(decompose(img)) ≈ img, the decomposition is valid.

Key insight: mixbox.lerp(a, b, t) == latent_to_rgb((1-t)*la + t*lb).
So stacking pigment i with alpha α over composite C is:
    latent_C := (1 - α) * latent_C + α * latent_pigment_i

The latent blend is fully numpy-vectorizable. Only the final
latent_to_rgb call is per-color — and we deduplicate via quantization
so a 1000x1000 image becomes ~10K unique conversions instead of 1M.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import colour
import mixbox
import numpy as np
from numpy.typing import NDArray

_LATENT_SIZE: int = mixbox.LATENT_SIZE  # 7
_QUANT_BINS: int = 1024  # latent quantization for dedup; ΔE drift < 0.5


def blend_two_pigments(
    rgb_a: tuple[int, int, int],
    rgb_b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Thin wrapper around mixbox.lerp for two-pigment mixing."""
    result = mixbox.lerp(tuple(rgb_a), tuple(rgb_b), float(t))
    return (int(result[0]), int(result[1]), int(result[2]))


def stack_pigments(
    pigment_rgbs: Sequence[tuple[int, int, int]],
    alphas: Sequence[float],
    substrate_rgb: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    """Stack N pigments on substrate via mixbox latent blending. Order matters."""
    if len(pigment_rgbs) != len(alphas):
        raise ValueError(
            f"pigment_rgbs ({len(pigment_rgbs)}) and alphas "
            f"({len(alphas)}) length mismatch"
        )
    composite: tuple[float, ...] = mixbox.rgb_to_latent(tuple(substrate_rgb))
    for rgb, alpha in zip(pigment_rgbs, alphas, strict=True):
        a = max(0.0, min(1.0, float(alpha)))
        if a == 0.0:
            continue
        pigment_lat = mixbox.rgb_to_latent(tuple(rgb))
        composite = tuple(
            (1.0 - a) * composite[k] + a * pigment_lat[k]
            for k in range(_LATENT_SIZE)
        )
    out = mixbox.latent_to_rgb(composite)
    return (int(out[0]), int(out[1]), int(out[2]))


def _palette_latents(
    palette_rgbs: Sequence[tuple[int, int, int]],
) -> NDArray[np.float64]:
    """Convert palette to (P, 7) latent matrix."""
    arr = np.empty((len(palette_rgbs), _LATENT_SIZE), dtype=np.float64)
    for i, rgb in enumerate(palette_rgbs):
        arr[i] = mixbox.rgb_to_latent(tuple(rgb))
    return arr


def _latents_to_rgb_dedup(
    latents_flat: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Vectorized latent -> RGB via quantize-dedup-lookup.

    Quantizes latents to _QUANT_BINS levels, calls mixbox.latent_to_rgb
    once per unique quantized vector, then scatters back. ΔE drift from
    quantization is <0.5 with _QUANT_BINS=1024.
    """
    # latents in mixbox span roughly [-0.5, 1.5]; pick a safe global range
    lo, hi = -1.0, 2.0
    scale = (_QUANT_BINS - 1) / (hi - lo)
    quantized = np.clip(
        np.round((latents_flat - lo) * scale).astype(np.int32),
        0,
        _QUANT_BINS - 1,
    )
    # pack 7 int16 indices into one row -> hashable via void view
    packed = np.ascontiguousarray(quantized.astype(np.int16))
    view = packed.view([("", packed.dtype)] * _LATENT_SIZE).ravel()
    unique_keys, inverse = np.unique(view, return_inverse=True)

    # reconstruct unique latents from their quantized indices
    unique_quant = unique_keys.view(np.int16).reshape(-1, _LATENT_SIZE)
    unique_latents = unique_quant.astype(np.float64) / scale + lo

    unique_rgb = np.empty((unique_latents.shape[0], 3), dtype=np.uint8)
    for i in range(unique_latents.shape[0]):
        rgb = mixbox.latent_to_rgb(tuple(unique_latents[i].tolist()))
        unique_rgb[i] = rgb

    return unique_rgb[inverse]


def forward_render_km(
    per_pigment_alphas: Sequence[NDArray[np.floating]],
    palette_rgbs: Sequence[tuple[int, int, int]],
    print_order: Sequence[int] | None = None,
    substrate_rgb: tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.uint8]:
    """Forward-render N pigment alpha maps via mixbox K-M stack.

    Args:
        per_pigment_alphas: list of (H, W) float arrays in [0, 1], one per palette entry.
        palette_rgbs: P pigment colors as (R, G, B) tuples in [0, 255].
        print_order: stacking order indices into palette; defaults to range(P).
        substrate_rgb: background color (paper).

    Returns:
        (H, W, 3) uint8 rendered composite.
    """
    if not per_pigment_alphas:
        raise ValueError("per_pigment_alphas is empty")
    if len(per_pigment_alphas) != len(palette_rgbs):
        raise ValueError(
            f"alpha count ({len(per_pigment_alphas)}) != palette size "
            f"({len(palette_rgbs)})"
        )
    h, w = per_pigment_alphas[0].shape
    for i, a in enumerate(per_pigment_alphas):
        if a.shape != (h, w):
            raise ValueError(f"alpha[{i}] shape {a.shape} != ({h}, {w})")

    order = list(range(len(palette_rgbs))) if print_order is None else list(print_order)
    palette_lat = _palette_latents(palette_rgbs)  # (P, 7)
    substrate_lat = np.array(mixbox.rgb_to_latent(tuple(substrate_rgb)), dtype=np.float64)

    # composite latent buffer (H, W, 7)
    composite = np.broadcast_to(substrate_lat, (h, w, _LATENT_SIZE)).copy()

    for idx in order:
        alpha = np.clip(per_pigment_alphas[idx].astype(np.float64), 0.0, 1.0)
        if not np.any(alpha):
            continue
        pigment_lat = palette_lat[idx]  # (7,)
        a3 = alpha[..., None]  # (H, W, 1)
        composite = (1.0 - a3) * composite + a3 * pigment_lat

    flat = composite.reshape(-1, _LATENT_SIZE)
    rgb_flat = _latents_to_rgb_dedup(flat)
    return cast(NDArray[np.uint8], rgb_flat.reshape(h, w, 3))


def composite_delta_e2000(
    img_a: NDArray[np.uint8],
    img_b: NDArray[np.uint8],
) -> tuple[float, float]:
    """Mean and p95 CIEDE2000 ΔE between two RGB images.

    Used by orchestrator's master gate to verify forward(decompose(x)) ≈ x.
    """
    if img_a.shape != img_b.shape:
        raise ValueError(f"shape mismatch: {img_a.shape} vs {img_b.shape}")
    if img_a.ndim != 3 or img_a.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3); got {img_a.shape}")

    a_f = img_a.astype(np.float64) / 255.0
    b_f = img_b.astype(np.float64) / 255.0
    # sRGB -> XYZ (D65) -> Lab
    a_xyz = colour.sRGB_to_XYZ(a_f)
    b_xyz = colour.sRGB_to_XYZ(b_f)
    a_lab = colour.XYZ_to_Lab(a_xyz)
    b_lab = colour.XYZ_to_Lab(b_xyz)
    de = colour.delta_E(a_lab, b_lab, method="CIE 2000")
    flat = np.asarray(de, dtype=np.float64).ravel()
    return float(np.mean(flat)), float(np.percentile(flat, 95))
