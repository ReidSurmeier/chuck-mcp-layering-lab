"""D10.c — S4 warm-start bridge.

Wraps Wave A's :func:`backend.algorithms.decomposition.tan_rgb_geometry.decompose_image`
(convex-hull palette + Delaunay barycentric weights summing to 1) and
remaps each hull vertex's RGB anchor onto the Chuck pigment catalog. The lab
warm-start then adds printmaking role seeds for broad underlayers and separate
high-chroma accent plates so L-BFGS does not begin from a pixel-detail-only
palette.

Per interface contract W-2: vertices that snap to the same Mixbox
pigment have their barycentric weights aggregated, NOT duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from backend.algorithms.decomposition import tan_rgb_geometry as _tan
from backend.services.v23.core import color as _color
from backend.services.v23.core import forward_render_jax as _fr

_LIGHT_UNDERLAYER_PIGMENTS = (0, 1, 2, 13, 14, 21, 23)
_SUBTLE_TINT_PIGMENTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("cool", (21, 7, 20, 6)),
    ("pink", (17, 3, 16, 4)),
    ("green", (23, 8, 22)),
    ("warm", (13, 14, 1, 0)),
)
_PIGMENT_GROUPS = (
    (0, 1, 13),          # yellow / ochre
    (2, 10, 14, 15),     # orange / earth
    (3, 16, 17, 18),     # reds
    (4, 5),              # magenta / violet
    (6, 7, 19, 20, 21),  # blues
    (8, 9, 22, 23),      # greens
    (11, 12),            # umber / key
)
_CHROMA_PLATE_PIGMENTS = tuple(
    sorted({pid for group in _PIGMENT_GROUPS[:-1] for pid in group})
)
_KEY_DETAIL_PIGMENTS = (12, 11, 15, 19, 20)
_LAB_WARMSTART_MAX_IMPRESSIONS = 12


@dataclass(frozen=True)
class WarmStartResult:
    """Output of :func:`tan_to_pigment_warmstart`. Pure numpy, no I/O."""

    alpha_stack: NDArray[np.float32]  # (M, H, W), per-pigment alpha seeds
    pigment_idx: tuple[int, ...]      # length M, each indexes the Chuck pigment catalog
    palette_rgb: NDArray[np.float32]  # (M, 3) snapped pigment RGB anchors


def _snap_palette_to_pigment(palette_rgb_01: NDArray[np.float32]) -> NDArray[np.int32]:
    """Snap each palette vertex to its nearest Chuck pigment index by Euclidean RGB distance."""
    pigments = _fr.PIGMENT_TABLE
    # Broadcast: palette (K, 1, 3) vs pigments (1, P, 3) -> (K, P) distances.
    diffs = palette_rgb_01[:, None, :] - pigments[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    return np.argmin(d2, axis=-1).astype(np.int32)


def _aggregate_duplicate_pigments(
    alphas: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> tuple[NDArray[np.float32], tuple[int, ...]]:
    """Collapse vertices that snap to the same pigment by summing alphas."""
    h, w = alphas.shape[:2]
    unique_pigments: list[int] = []
    pigment_to_slot: dict[int, int] = {}
    for pid in pigment_idx.tolist():
        if pid not in pigment_to_slot:
            pigment_to_slot[pid] = len(unique_pigments)
            unique_pigments.append(pid)

    m = len(unique_pigments)
    out = np.zeros((m, h, w), dtype=np.float32)
    for source_slot, pid in enumerate(pigment_idx.tolist()):
        target_slot = pigment_to_slot[pid]
        out[target_slot] += alphas[..., source_slot]
    return out, tuple(unique_pigments)


def tan_to_pigment_warmstart(
    rgb: NDArray[np.uint8],
    *,
    target_palette_size: int = 8,
) -> WarmStartResult:
    """Run Tan RGB-geometry decomposition + remap palette onto Chuck catalog.

    Args:
        rgb: ``(H, W, 3)`` uint8 image.
        target_palette_size: hint to Tan's convex-hull extractor for palette size.

    Returns:
        ``WarmStartResult`` with ``alpha_stack`` shape ``(M, H, W)``,
        ``pigment_idx`` length M (unique pigment indices after de-duplication),
        and ``palette_rgb`` the snapped pigment RGB anchors in [0, 1].
    """
    palette, alphas_hwM = _tan.decompose_image(rgb, target_palette_size=target_palette_size)
    # palette shape (K, 3) float [0,1]; alphas_hwM shape (H, W, K)
    pigment_idx = _snap_palette_to_pigment(palette.astype(np.float32))
    # Convert to (K, H, W)
    alpha_first = np.transpose(alphas_hwM, (2, 0, 1)).astype(np.float32)
    collapsed, unique_pigments = _aggregate_duplicate_pigments(
        np.transpose(alpha_first, (1, 2, 0)), pigment_idx
    )
    snapped_palette = _fr.PIGMENT_TABLE[np.array(unique_pigments)]
    return WarmStartResult(
        alpha_stack=collapsed,
        pigment_idx=unique_pigments,
        palette_rgb=snapped_palette.astype(np.float32),
    )


def _smooth_alpha(alpha: NDArray[np.float32], *, radius: float) -> NDArray[np.float32]:
    arr = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=radius))
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def _broad_underlayer_seed(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Broad low-frequency support seed, independent of the final detail color.

    The chosen pigment is inferred separately. On the current Emma reference the
    weighted mean is warm/light, so the role naturally selects a yellow/ochre.
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    paper = _fr.PAPER_RGB.astype(np.float32)
    paper_distance = np.linalg.norm(arr - paper[None, None, :], axis=-1)
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    low_frequency_presence = np.clip(paper_distance / 0.48, 0.0, 1.0)
    tonal_presence = np.clip((0.96 - lum) / 0.72, 0.0, 1.0)
    color_presence = np.clip(chroma / 0.42, 0.0, 1.0)
    alpha = np.clip(
        (0.52 * low_frequency_presence + 0.28 * tonal_presence + 0.20 * color_presence) * 0.62,
        0.0,
        0.62,
    )
    return _smooth_alpha(alpha.astype(np.float32), radius=max(8.0, min(rgb.shape[:2]) / 40.0))


def _chroma_plate_seed(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Separate regional high-chroma seed for accent/hue-shift blocks."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    # High chroma plus separation from a neutral/value-only explanation.
    channel_gap = np.maximum.reduce([r, g, b]) - np.partition(arr, -2, axis=-1)[..., -2]
    alpha = np.clip((chroma - 0.18) / 0.46, 0.0, 1.0)
    alpha *= np.clip(channel_gap / 0.26, 0.0, 1.0)
    alpha = np.clip(alpha * 0.86, 0.0, 0.86)
    return _smooth_alpha(alpha.astype(np.float32), radius=max(3.0, min(rgb.shape[:2]) / 96.0))


def _dark_key_seed(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Late key/detail seed from dark target structure."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    dx = np.pad(np.abs(lum[:, 1:] - lum[:, :-1]), ((0, 0), (0, 1)))
    dy = np.pad(np.abs(lum[1:, :] - lum[:-1, :]), ((0, 1), (0, 0)))
    edge = np.sqrt(dx * dx + dy * dy)
    edge = edge / max(float(edge.max()), 1e-6)
    dark = np.clip((0.44 - lum) / 0.34, 0.0, 1.0)
    alpha = np.clip((0.72 * dark + 0.28 * dark * edge) * 0.88, 0.0, 0.88)
    return _smooth_alpha(alpha.astype(np.float32), radius=max(1.25, min(rgb.shape[:2]) / 180.0))


def _subtle_tint_seeds(rgb: NDArray[np.uint8]) -> list[tuple[NDArray[np.float32], int]]:
    """Detect pale near-paper hue shifts that should print instead of reading as white.

    Mokuhanga highlights often live as low-alpha blue/grey/pink/yellow washes.
    A plain paper-distance gate misses those because their RGB values are close
    to the substrate. Lab opponent axes make the small hue shifts visible.
    """
    arr = rgb.astype(np.float32) / 255.0
    lab = _color.srgb_to_lab(arr)
    paper_lab = _color.srgb_to_lab(_fr.PAPER_RGB.reshape(1, 1, 3))[0, 0]
    delta = lab - paper_lab[None, None, :]
    d_e = np.sqrt(np.sum(delta * delta, axis=-1))
    luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    light_region = np.clip((luminance - 0.48) / 0.34, 0.0, 1.0)
    subtle = np.clip((d_e - 1.2) / 16.0, 0.0, 1.0) * light_region
    a_axis = delta[..., 1]
    b_axis = delta[..., 2]
    axes = {
        "cool": np.clip((-b_axis + 0.20 * -a_axis) / 9.0, 0.0, 1.0),
        "pink": np.clip((a_axis + 0.20 * -b_axis) / 9.0, 0.0, 1.0),
        "green": np.clip((-a_axis + 0.12 * b_axis) / 9.0, 0.0, 1.0),
        "warm": np.clip((b_axis + 0.12 * a_axis) / 9.0, 0.0, 1.0),
    }

    seeds: list[tuple[NDArray[np.float32], int]] = []
    for name, candidates in _SUBTLE_TINT_PIGMENTS:
        alpha = np.clip(subtle * axes[name] * 0.34, 0.0, 0.34).astype(np.float32)
        alpha = _smooth_alpha(alpha, radius=max(4.0, min(rgb.shape[:2]) / 88.0))
        if float((alpha > 0.025).mean()) < 0.006:
            continue
        tint_color = _weighted_mean_rgb(rgb, alpha)
        pid = _choose_nearest_pigment(tint_color, candidates, prefer_light=True)
        seeds.append((alpha, pid))
    return seeds


def _weighted_mean_rgb(
    rgb: NDArray[np.uint8],
    alpha: NDArray[np.float32],
) -> NDArray[np.float32]:
    arr = rgb.astype(np.float32) / 255.0
    weights = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    total = float(weights.sum())
    if total <= 1e-6:
        return arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return (arr * weights[..., None]).sum(axis=(0, 1)) / total


def _choose_nearest_pigment(
    rgb_01: NDArray[np.float32],
    candidates: tuple[int, ...],
    *,
    prefer_light: bool = False,
    prefer_dark: bool = False,
) -> int:
    pigments = _fr.PIGMENT_TABLE[np.asarray(candidates, dtype=np.int32)]
    d2 = np.sum((pigments - rgb_01[None, :]) ** 2, axis=-1)
    if prefer_light:
        lum = 0.299 * pigments[:, 0] + 0.587 * pigments[:, 1] + 0.114 * pigments[:, 2]
        d2 = d2 + 0.10 * (1.0 - lum)
    if prefer_dark:
        lum = 0.299 * pigments[:, 0] + 0.587 * pigments[:, 1] + 0.114 * pigments[:, 2]
        d2 = d2 + 0.12 * lum
    return int(candidates[int(np.argmin(d2))])


def _same_pigment_group(a: int, b: int) -> bool:
    return any(a in group and b in group for group in _PIGMENT_GROUPS)


def _drop_low_coverage(
    alpha_stack: NDArray[np.float32],
    pigment_idx: tuple[int, ...],
    *,
    keep_first: int,
    max_impressions: int,
) -> tuple[NDArray[np.float32], tuple[int, ...]]:
    """Trim weak optional slots while preserving explicit role seeds first."""
    if alpha_stack.shape[0] <= max_impressions:
        return alpha_stack, pigment_idx

    required = set(range(keep_first))
    optional = list(range(keep_first, alpha_stack.shape[0]))
    optional.sort(key=lambda i: float(alpha_stack[i].mean()), reverse=True)
    keep = sorted(list(required) + optional[: max_impressions - len(required)])
    return alpha_stack[keep], tuple(pigment_idx[i] for i in keep)


def layering_lab_warmstart(
    rgb: NDArray[np.uint8],
    *,
    target_palette_size: int = 8,
    max_impressions: int = _LAB_WARMSTART_MAX_IMPRESSIONS,
) -> WarmStartResult:
    """Role-aware warm-start for the Chuck layering lab.

    The old Tan-only seed can assign light base colors to skinny target-image
    contours. This seed first asks which low-frequency underlayer and which
    high-chroma accent plate the image warrants, then appends the remaining Tan
    palette slots for mid/detail refinement.
    """
    base = tan_to_pigment_warmstart(rgb, target_palette_size=target_palette_size)

    role_alphas: list[NDArray[np.float32]] = []
    role_pigments: list[int] = []
    suppress_tan_groups: list[int] = []
    suppress_tan_exact: set[int] = set()

    underlayer = _broad_underlayer_seed(rgb)
    underlayer_pid: int | None = None
    if float((underlayer > 0.08).mean()) >= 0.02:
        underlayer_color = _weighted_mean_rgb(rgb, underlayer)
        underlayer_pid = _choose_nearest_pigment(
            underlayer_color,
            _LIGHT_UNDERLAYER_PIGMENTS,
            prefer_light=True,
        )
        role_alphas.append(underlayer)
        role_pigments.append(underlayer_pid)
        suppress_tan_groups.append(underlayer_pid)

    for tint_alpha, tint_pid in _subtle_tint_seeds(rgb):
        if any(_same_pigment_group(tint_pid, pid) for pid in suppress_tan_groups):
            continue
        if tint_pid in role_pigments:
            continue
        role_alphas.append(tint_alpha)
        role_pigments.append(tint_pid)
        suppress_tan_exact.add(tint_pid)

    chroma = _chroma_plate_seed(rgb)
    chroma_pid: int | None = None
    if float((chroma > 0.08).mean()) >= 0.002:
        chroma_color = _weighted_mean_rgb(rgb, chroma)
        chroma_pid = _choose_nearest_pigment(chroma_color, _CHROMA_PLATE_PIGMENTS)
        if underlayer_pid is None or not _same_pigment_group(chroma_pid, underlayer_pid):
            role_alphas.append(chroma)
            role_pigments.append(chroma_pid)
            suppress_tan_groups.append(chroma_pid)
        else:
            chroma_pid = None

    key = _dark_key_seed(rgb)
    key_pid: int | None = None
    if float((key > 0.08).mean()) >= 0.004:
        key_color = _weighted_mean_rgb(rgb, key)
        key_lum = float(0.299 * key_color[0] + 0.587 * key_color[1] + 0.114 * key_color[2])
        key_pid = 12 if key_lum < 0.32 else _choose_nearest_pigment(
            key_color,
            _KEY_DETAIL_PIGMENTS,
            prefer_dark=True,
        )
        role_alphas.append(key)
        role_pigments.append(key_pid)
        suppress_tan_groups.append(key_pid)

    explicit_seed_count = len(role_pigments)

    for alpha, pid in zip(base.alpha_stack, base.pigment_idx, strict=True):
        pid = int(pid)
        if any(_same_pigment_group(pid, seed_pid) for seed_pid in suppress_tan_groups):
            # Explicit role seeds own these families; Tan must not add another
            # unlabeled copy that later appears as a duplicate pull.
            continue
        if pid in suppress_tan_exact:
            continue
        role_alphas.append(alpha.astype(np.float32))
        role_pigments.append(pid)

    alpha_stack = np.stack(role_alphas, axis=0).astype(np.float32)
    pigment_tuple = tuple(role_pigments)
    alpha_stack, pigment_tuple = _drop_low_coverage(
        alpha_stack,
        pigment_tuple,
        keep_first=explicit_seed_count,
        max_impressions=min(max_impressions, max(1, target_palette_size + 2)),
    )
    palette = _fr.PIGMENT_TABLE[np.array(pigment_tuple, dtype=np.int32)]
    return WarmStartResult(
        alpha_stack=alpha_stack,
        pigment_idx=pigment_tuple,
        palette_rgb=palette.astype(np.float32),
    )


__all__ = ["WarmStartResult", "layering_lab_warmstart", "tan_to_pigment_warmstart"]
