"""D10.d — S5 inverse stack solver (real, JAX L-BFGS).

Wraps the smoke-mode JAX solver with proper sigmoid_box reparam initialised
from the S4 Tan warm-start. Iter budgets respect ``solve_profile``:
- fast: 60 iters, 1 pyramid level
- default: 180 iters, 1 pyramid level
- thorough: 400 iters, 1 pyramid level

Multi-level pyramid (8x→4x→2x→1x) is the D7.3 follow-up — for day-1 the
single-level run is sufficient to deliver propose_stack with real
impressions (even if reconstruction ΔE doesn't hit Tier-1 mean ≤ 1.5 yet).

Per addendum-v3 fix 1: topology constraints (min-island, mean-island)
do NOT live in this loss. They run post-solve in
:mod:`backend.services.v23.core.topology_repair` (D11).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from backend.services.v23.core import forward_render_jax

_VALID_SOLVE_PROFILES = ("fast", "default", "thorough")
_PROFILE_ITERS = {"fast": 60, "default": 180, "thorough": 400}
_PROFILE_MAX_PIXELS = {"fast": 256_000, "default": 512_000, "thorough": 768_000}
_LOWPASS_WINDOW = 8
_MIDPASS_WINDOW = 12
_UNDERPASS_WINDOW = 32
_SPECKLE_WINDOW = 9
_UNDER_CONTROL_FACTOR = 12
_UNDER_CARVE_FACTOR = 2
_MID_CONTROL_FACTOR = 4
_UNDER_TARGET_STRENGTH = 0.42
_MID_TARGET_STRENGTH = 0.72
_JIGSAW_OVERLAP_WEIGHT = 0.012
_ADAPTIVE_PIGMENT_START = 24
_SUPPORT_PIGMENTS = frozenset({0, 1, 2, 13, 14, 21, 23, 26, 31})
_DARK_KEY_PIGMENTS = frozenset({11, 12, 15, 19, 20})


@dataclass(frozen=True)
class SolverResult:
    """Output of :func:`run_s5_solver`. Pure numpy + a list of impression dicts."""

    alpha_stack: NDArray[np.float32]  # (M, H, W) in [0, 1]
    pigment_idx: tuple[int, ...]
    impressions: list[dict[str, Any]]
    initial_loss: float
    final_loss: float
    iters_used: int
    wall_s: float
    solve_profile: str
    optimized_shape: tuple[int, int]
    original_shape: tuple[int, int]
    downsample_scale: float


@dataclass(frozen=True)
class _RoleLayout:
    under_count: int
    mid_count: int
    detail_count: int

    @property
    def under_end(self) -> int:
        return self.under_count

    @property
    def mid_end(self) -> int:
        return self.under_count + self.mid_count

    @property
    def total(self) -> int:
        return self.under_count + self.mid_count + self.detail_count


def _sigmoid_box(alpha_raw: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(alpha_raw)


def _rgb_luminance(rgb: jnp.ndarray) -> jnp.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _avg_pool_rgb(rgb: jnp.ndarray, window: int = _LOWPASS_WINDOW) -> jnp.ndarray:
    """Average-pool RGB with edge correction so non-multiple sizes stay unbiased."""
    numerator = jax.lax.reduce_window(
        rgb,
        0.0,
        jax.lax.add,
        (window, window, 1),
        (window, window, 1),
        "SAME",
    )
    denominator = jax.lax.reduce_window(
        jnp.ones_like(rgb),
        0.0,
        jax.lax.add,
        (window, window, 1),
        (window, window, 1),
        "SAME",
    )
    return numerator / jnp.maximum(denominator, 1.0)


def _avg_pool_alpha(alpha: jnp.ndarray, window: int = _SPECKLE_WINDOW) -> jnp.ndarray:
    """Average-pool an ``(M, H, W)`` alpha stack per impression."""
    pooled = jax.lax.reduce_window(
        alpha,
        0.0,
        jax.lax.add,
        (1, window, window),
        (1, 1, 1),
        "SAME",
    )
    return pooled / float(window * window)


def _target_edge_weight(target_rgb: jnp.ndarray) -> jnp.ndarray:
    """Detail-aware weights plus near-paper tint sensitivity."""
    lum = _rgb_luminance(target_rgb)
    dx = jnp.pad(jnp.abs(lum[:, 1:] - lum[:, :-1]), ((0, 0), (0, 1)))
    dy = jnp.pad(jnp.abs(lum[1:, :] - lum[:-1, :]), ((0, 1), (0, 0)))
    edge = jnp.sqrt(dx * dx + dy * dy + 1e-8)
    edge = edge / (jnp.max(edge) + 1e-6)
    paper = jnp.asarray(forward_render_jax.PAPER_RGB, dtype=jnp.float32)
    paper_delta = jnp.sqrt(jnp.sum((target_rgb - paper[None, None, :]) ** 2, axis=-1))
    near_paper_tint = (
        jax.nn.sigmoid((lum - 0.50) * 12.0)
        * jax.nn.sigmoid((paper_delta - 0.018) * 48.0)
        * jax.nn.sigmoid((0.26 - paper_delta) * 18.0)
    )
    return (1.0 + 3.0 * edge + 5.0 * near_paper_tint)[..., None]


def _layered_alpha_tv(alpha: jnp.ndarray) -> jnp.ndarray:
    """TV with more pressure on early underlayers than final key impressions."""
    m = alpha.shape[0]
    dx = jnp.abs(alpha[:, :, 1:] - alpha[:, :, :-1])
    dy = jnp.abs(alpha[:, 1:, :] - alpha[:, :-1, :])
    weights = jnp.linspace(1.55, 0.65, m, dtype=jnp.float32)
    tv_x = jnp.mean(dx, axis=(1, 2))
    tv_y = jnp.mean(dy, axis=(1, 2))
    return jnp.mean(weights * (tv_x + tv_y))


def _alpha_speckle(alpha: jnp.ndarray) -> jnp.ndarray:
    """Soft proxy for tiny islands: penalize local peaks above their neighborhood."""
    local = _avg_pool_alpha(alpha)
    active = jax.nn.sigmoid((alpha - 0.25) * 16.0)
    broad_support = jax.nn.relu(0.22 - local)
    return jnp.mean(active * broad_support * broad_support)


def _alpha_highfreq(alpha: jnp.ndarray, window: int) -> jnp.ndarray:
    """Penalize plate detail that survives below a role's intended brush scale."""
    local = _avg_pool_alpha(alpha, window)
    return jnp.mean((alpha - local) ** 2)


def _pairwise_overlap(alpha: jnp.ndarray) -> jnp.ndarray:
    """Encourage middle color plates to jigsaw instead of all fading together."""
    m = alpha.shape[0]
    if m <= 1:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            total = total + jnp.mean(alpha[i] * alpha[j])
            pairs += 1
    return total / float(max(pairs, 1))


def _role_layout(m: int) -> _RoleLayout:
    """Assign printmaking roles after light-to-dark ordering."""
    if m <= 2:
        return _RoleLayout(under_count=0, mid_count=0, detail_count=m)
    if m <= 4:
        return _RoleLayout(under_count=1, mid_count=max(0, m - 2), detail_count=1)
    detail_count = 2
    under_count = min(3, m - detail_count)
    mid_count = max(0, m - under_count - detail_count)
    return _RoleLayout(
        under_count=under_count,
        mid_count=mid_count,
        detail_count=detail_count,
    )


def _role_layout_for_pigments(pigment_idx: NDArray[np.int32]) -> _RoleLayout:
    """Use pigment role hints when adaptive premix/wash slots are present."""
    m = int(len(pigment_idx))
    if m <= 4 or not any(int(pid) >= _ADAPTIVE_PIGMENT_START for pid in pigment_idx.tolist()):
        return _role_layout(m)
    detail_count = 2
    support_count = sum(1 for pid in pigment_idx.tolist() if int(pid) in _SUPPORT_PIGMENTS)
    under_count = min(3, max(1, support_count), m - detail_count)
    mid_count = max(0, m - under_count - detail_count)
    return _RoleLayout(
        under_count=under_count,
        mid_count=mid_count,
        detail_count=detail_count,
    )


def _control_shape(shape: tuple[int, int], factor: int) -> tuple[int, int]:
    h, w = shape
    return max(16, int(round(h / factor))), max(16, int(round(w / factor)))


def _logit_np(alpha: NDArray[np.float32]) -> NDArray[np.float32]:
    clipped = np.clip(alpha, 0.02, 0.98)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _blur_alpha_stack_np(
    alpha_stack: NDArray[np.float32],
    *,
    radius: float,
) -> NDArray[np.float32]:
    out: list[NDArray[np.float32]] = []
    for alpha in alpha_stack:
        arr = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=radius))
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(out, axis=0).astype(np.float32)


def _make_role_params(
    alpha_stack: NDArray[np.float32],
    layout: _RoleLayout,
) -> dict[str, jnp.ndarray]:
    """Build staged control-grid params for under/mid/detail pulls."""
    h, w = alpha_stack.shape[1:]
    params: dict[str, jnp.ndarray] = {}
    if layout.under_count:
        under_h, under_w = _control_shape((h, w), _UNDER_CONTROL_FACTOR)
        under_src = _blur_alpha_stack_np(
            alpha_stack[: layout.under_end],
            radius=max(4.0, min(h, w) / 56.0),
        )
        under = _resize_alpha_stack(under_src, (under_w, under_h))
        params["under"] = jnp.asarray(_logit_np(under), dtype=jnp.float32)
        carve_h, carve_w = _control_shape((h, w), _UNDER_CARVE_FACTOR)
        carve_gate = np.clip(
            alpha_stack[: layout.under_end] / np.maximum(under_src, 0.08),
            0.02,
            0.98,
        ).astype(np.float32)
        carve = _resize_alpha_stack(carve_gate, (carve_w, carve_h))
        params["under_carve"] = jnp.asarray(_logit_np(carve), dtype=jnp.float32)
    if layout.mid_count:
        mid_h, mid_w = _control_shape((h, w), _MID_CONTROL_FACTOR)
        mid = _resize_alpha_stack(
            alpha_stack[layout.under_end : layout.mid_end],
            (mid_w, mid_h),
        )
        params["mid"] = jnp.asarray(_logit_np(mid), dtype=jnp.float32)
    if layout.detail_count:
        detail = alpha_stack[layout.mid_end : layout.total]
        params["detail"] = jnp.asarray(_logit_np(detail), dtype=jnp.float32)
    return params


def _resize_alpha_jax(alpha: jnp.ndarray, shape: tuple[int, int, int]) -> jnp.ndarray:
    if tuple(alpha.shape) == shape:
        return alpha
    return jnp.clip(
        jax.image.resize(alpha, shape, method="linear", antialias=False),
        0.0,
        1.0,
    )


def _expand_role_params(
    params: dict[str, jnp.ndarray],
    layout: _RoleLayout,
    target_shape: tuple[int, int],
) -> jnp.ndarray:
    """Expand role params back to full solve-grid alpha stack."""
    h, w = target_shape
    parts: list[jnp.ndarray] = []
    if layout.under_count:
        envelope = _resize_alpha_jax(
            _sigmoid_box(params["under"]),
            (layout.under_count, h, w),
        )
        if "under_carve" in params:
            carve = _resize_alpha_jax(
                _sigmoid_box(params["under_carve"]),
                (layout.under_count, h, w),
            )
            under = envelope * carve
        else:
            under = envelope
        parts.append(jnp.clip(under, 0.0, 1.0))
    if layout.mid_count:
        mid = _sigmoid_box(params["mid"])
        parts.append(_resize_alpha_jax(mid, (layout.mid_count, h, w)))
    if layout.detail_count:
        detail = _sigmoid_box(params["detail"])
        parts.append(_resize_alpha_jax(detail, (layout.detail_count, h, w)))
    return jnp.concatenate(parts, axis=0)


def _forward_mhw(alpha_mhw: jnp.ndarray, pigment_idx: jnp.ndarray) -> jnp.ndarray:
    alpha_hwm = jnp.transpose(alpha_mhw, (1, 2, 0))
    return forward_render_jax.forward_render(alpha_hwm, pigment_idx)


def _print_order(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> NDArray[np.int64]:
    """Return stable print order: light support, mid chroma, dark key."""
    pigment_rgb = forward_render_jax.PIGMENT_TABLE[pigment_idx]
    luminance = (
        0.299 * pigment_rgb[:, 0]
        + 0.587 * pigment_rgb[:, 1]
        + 0.114 * pigment_rgb[:, 2]
    )
    coverage = alpha_stack.mean(axis=(1, 2))
    priority = np.array([
        0 if int(pid) in _SUPPORT_PIGMENTS
        else 2 if int(pid) in _DARK_KEY_PIGMENTS
        else 1
        for pid in pigment_idx.tolist()
    ], dtype=np.int32)
    # Primary key: support -> mid chroma -> dark key. Secondary keys keep each
    # band light-to-dark and put broad supports before small marks.
    return np.lexsort((-coverage, -luminance, priority))


def _reorder_for_printing(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int64]]:
    order = _print_order(alpha_stack, pigment_idx)
    return alpha_stack[order], pigment_idx[order], order


def _solver_loss(
    params: dict[str, jnp.ndarray],
    target_rgb: jnp.ndarray,
    pigment_idx: jnp.ndarray,
    target_weight: jnp.ndarray,
    target_underpass: jnp.ndarray,
    target_midpass: jnp.ndarray,
    target_lowpass: jnp.ndarray,
    layout: _RoleLayout,
    target_shape: tuple[int, int],
    stage: str = "final",
) -> jnp.ndarray:
    """Print-aware differentiable objective for alpha stack refinement."""
    alpha = _expand_role_params(params, layout, target_shape)
    rgb = _forward_mhw(alpha, pigment_idx)

    under_loss = 0.0
    mid_loss = 0.0
    if layout.under_count:
        rgb_under = _forward_mhw(
            alpha[: layout.under_end],
            pigment_idx[: layout.under_end],
        )
        under_loss = jnp.mean(
            (_avg_pool_rgb(rgb_under, _UNDERPASS_WINDOW) - target_underpass) ** 2
        )
    if layout.mid_end > 0:
        rgb_mid = _forward_mhw(alpha[: layout.mid_end], pigment_idx[: layout.mid_end])
        mid_loss = jnp.mean(
            (_avg_pool_rgb(rgb_mid, _MIDPASS_WINDOW) - target_midpass) ** 2
        )

    if stage == "under":
        alpha_under = alpha[: layout.under_end]
        return (
            under_loss
            + 0.100 * _layered_alpha_tv(alpha_under)
            + 0.160 * _alpha_speckle(alpha_under)
            + 0.025 * _alpha_highfreq(alpha_under, _UNDERPASS_WINDOW)
        )
    if stage == "mid":
        alpha_prefix = alpha[: layout.mid_end]
        alpha_mid = alpha[layout.under_end : layout.mid_end]
        return (
            0.75 * mid_loss
            + 0.25 * under_loss
            + 0.060 * _layered_alpha_tv(alpha_prefix)
            + 0.080 * _alpha_speckle(alpha_prefix)
            + 0.018 * _alpha_highfreq(alpha[: layout.under_end], _UNDERPASS_WINDOW)
            + _JIGSAW_OVERLAP_WEIGHT * _pairwise_overlap(alpha_mid)
        )

    weighted_rgb = jnp.mean(target_weight * (rgb - target_rgb) ** 2)
    lowpass = jnp.mean((_avg_pool_rgb(rgb) - target_lowpass) ** 2)
    tv = _layered_alpha_tv(alpha)
    speckle = _alpha_speckle(alpha)
    under_highfreq = (
        _alpha_highfreq(alpha[: layout.under_end], _UNDERPASS_WINDOW)
        if layout.under_count else 0.0
    )
    mid_overlap = (
        _pairwise_overlap(alpha[layout.under_end : layout.mid_end])
        if layout.mid_count else 0.0
    )

    selected = jnp.asarray(forward_render_jax.PIGMENT_TABLE, dtype=jnp.float32)[pigment_idx]
    pigment_lum = 0.299 * selected[:, 0] + 0.587 * selected[:, 1] + 0.114 * selected[:, 2]
    dark_strength = (1.0 - pigment_lum)[:, None, None]
    target_lum = _rgb_luminance(target_rgb)[None, :, :]
    dark_on_bright = jnp.mean(alpha * dark_strength * target_lum * target_lum)

    return (
        weighted_rgb
        + 0.65 * lowpass
        + 0.040 * mid_loss
        + 0.055 * under_loss
        + 0.030 * tv
        + 0.045 * speckle
        + 0.018 * under_highfreq
        + _JIGSAW_OVERLAP_WEIGHT * mid_overlap
        + 0.015 * dark_on_bright
    )


def _alphas_to_impressions(
    alpha_stack: NDArray[np.float32],
    pigment_idx: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Convert (M, H, W) alpha stack → impressions list (print-order ascending)."""
    h, w = alpha_stack.shape[1:]
    total_px = float(h * w)
    impressions: list[dict[str, Any]] = []
    for i, pid in enumerate(pigment_idx):
        a = alpha_stack[i]
        coverage_pct = float((a > 0.05).sum()) / total_px * 100.0
        mean_alpha = float(a.mean())
        rgb = forward_render_jax.PIGMENT_TABLE[pid]
        luminance_okL = float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        impressions.append({
            "id": f"imp_{i + 1:03d}",
            "order_step": i + 1,
            "pigment_id": int(pid),
            "coverage_pct": round(coverage_pct, 3),
            "mean_alpha": round(mean_alpha, 4),
            "luminance_okL": round(luminance_okL, 4),
        })
    return impressions


def summarise_impressions(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Public impression summaries for post-solve organized alpha stacks."""
    pigment_tuple = tuple(int(p) for p in pigment_idx)
    return _alphas_to_impressions(alpha_stack.astype(np.float32), pigment_tuple)


def _max_solver_pixels(solve_profile: str) -> int:
    raw = os.environ.get("WOODBLOCK_SOLVER_MAX_PIXELS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 4_096:
            return value
    return _PROFILE_MAX_PIXELS[solve_profile]


def _resize_rgb(rgb: NDArray[np.float32], size: tuple[int, int]) -> NDArray[np.float32]:
    """Resize HWC RGB in [0, 1]. ``size`` is (width, height)."""
    arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").resize(size, Image.Resampling.LANCZOS)
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def _resize_alpha_stack(
    alpha_stack: NDArray[np.float32],
    size: tuple[int, int],
) -> NDArray[np.float32]:
    """Resize MHW alpha stack. ``size`` is (width, height)."""
    out: list[NDArray[np.float32]] = []
    for alpha in alpha_stack:
        arr = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "L").resize(size, Image.Resampling.BILINEAR)
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(out, axis=0).astype(np.float32)


def _stage_iter_budget(iters: int, layout: _RoleLayout) -> dict[str, int]:
    """Keep total work bounded while splitting solve into role stages."""
    if os.environ.get("WOODBLOCK_ROLE_WARMUP") != "1":
        return {"joint": iters}
    if layout.under_count and layout.mid_count and layout.detail_count:
        under = min(8, max(1, iters // 10))
        mid = min(8, max(1, iters // 10))
        detail = min(8, max(1, iters // 10))
        joint = max(8, iters - under - mid - detail)
        return {"under": under, "mid": mid, "detail": detail, "joint": joint}
    if layout.mid_count and layout.detail_count:
        mid = min(8, max(1, iters // 10))
        detail = min(8, max(1, iters // 10))
        return {"mid": mid, "detail": detail, "joint": max(8, iters - mid - detail)}
    return {"detail": iters}


def _prepare_solver_grid(
    target_rgb: NDArray[np.float32],
    alpha_init: NDArray[np.float32],
    *,
    solve_profile: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[int, int], float]:
    """Downsample large solves to fit commodity GPUs.

    The persisted plan still carries full-resolution alpha masks. This function
    only bounds the expensive L-BFGS state so native 2K images do not exhaust
    12 GB GPUs before any MCP artifacts can be emitted.
    """
    h, w = target_rgb.shape[:2]
    max_pixels = _max_solver_pixels(solve_profile)
    current_pixels = h * w
    if current_pixels <= max_pixels:
        return target_rgb, alpha_init, (h, w), 1.0

    scale = (max_pixels / float(current_pixels)) ** 0.5
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    target_small = _resize_rgb(target_rgb, (new_w, new_h))
    alpha_small = _resize_alpha_stack(alpha_init, (new_w, new_h))
    return target_small, alpha_small, (new_h, new_w), scale


def run_s5_solver(
    target_rgb: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    alpha_init: NDArray[np.float32],
    *,
    solve_profile: Literal["fast", "default", "thorough"] = "default",
) -> SolverResult:
    """JAX L-BFGS refinement of ``alpha_init`` to minimise forward-render error."""
    if solve_profile not in _VALID_SOLVE_PROFILES:
        raise ValueError(
            f"solve_profile must be one of {_VALID_SOLVE_PROFILES}, got {solve_profile!r}"
        )

    orig_h, orig_w, _ = target_rgb.shape
    m = int(len(pigment_idx))
    if alpha_init.shape != (m, orig_h, orig_w):
        raise ValueError(
            f"alpha_init shape {alpha_init.shape} != expected ({m}, {orig_h}, {orig_w})"
        )

    alpha_ordered, pigment_ordered, _order = _reorder_for_printing(
        alpha_init.astype(np.float32),
        pigment_idx.astype(np.int32),
    )

    iters = _PROFILE_ITERS[solve_profile]
    target_solve, alpha_solve, optimized_shape, downsample_scale = _prepare_solver_grid(
        target_rgb,
        alpha_ordered,
        solve_profile=solve_profile,
    )

    layout = _role_layout_for_pigments(pigment_ordered)
    params = _make_role_params(alpha_solve, layout)
    target_j = jnp.asarray(target_solve, dtype=jnp.float32)
    pigments_j = jnp.asarray(pigment_ordered, dtype=jnp.int32)
    target_weight = _target_edge_weight(target_j)
    paper_j = jnp.asarray(forward_render_jax.PAPER_RGB, dtype=jnp.float32)
    target_under_rgb = paper_j + _UNDER_TARGET_STRENGTH * (target_j - paper_j)
    target_mid_rgb = paper_j + _MID_TARGET_STRENGTH * (target_j - paper_j)
    target_underpass = _avg_pool_rgb(target_under_rgb, _UNDERPASS_WINDOW)
    target_midpass = _avg_pool_rgb(target_mid_rgb, _MIDPASS_WINDOW)
    target_lowpass = _avg_pool_rgb(target_j)
    target_shape = (int(target_solve.shape[0]), int(target_solve.shape[1]))

    def full_loss_fn(p):
        return _solver_loss(
            p,
            target_j,
            pigments_j,
            target_weight,
            target_underpass,
            target_midpass,
            target_lowpass,
            layout,
            target_shape,
            "final",
        )

    initial_loss = float(full_loss_fn(params))
    t0 = time.perf_counter()
    iters_used = 0
    for stage, maxiter in _stage_iter_budget(iters, layout).items():
        if stage == "joint":
            solver = jaxopt.LBFGS(fun=full_loss_fn, maxiter=maxiter, tol=1e-7)
            result = solver.run(params)
            params = result.params
            iters_used += (
                int(getattr(result.state, "iter_num", maxiter))
                if hasattr(result, "state")
                else maxiter
            )
            continue
        if stage not in params:
            continue

        def stage_loss_fn(variable, stage_name=stage, base_params=params):
            stage_params = dict(base_params)
            stage_params[stage_name] = variable
            return _solver_loss(
                stage_params,
                target_j,
                pigments_j,
                target_weight,
                target_underpass,
                target_midpass,
                target_lowpass,
                layout,
                target_shape,
                stage_name,
            )

        solver = jaxopt.LBFGS(fun=stage_loss_fn, maxiter=maxiter, tol=1e-7)
        result = solver.run(params[stage])
        params[stage] = result.params
        iters_used += (
            int(getattr(result.state, "iter_num", maxiter))
            if hasattr(result, "state")
            else maxiter
        )

    wall_s = time.perf_counter() - t0
    alpha_stack_solve = np.asarray(_expand_role_params(params, layout, target_shape))
    alpha_stack = (
        _resize_alpha_stack(alpha_stack_solve, (orig_w, orig_h))
        if optimized_shape != (orig_h, orig_w)
        else alpha_stack_solve
    )
    final_loss = float(full_loss_fn(params))

    pigment_tuple = tuple(int(p) for p in pigment_ordered.tolist())
    impressions = _alphas_to_impressions(alpha_stack, pigment_tuple)

    return SolverResult(
        alpha_stack=alpha_stack.astype(np.float32),
        pigment_idx=pigment_tuple,
        impressions=impressions,
        initial_loss=initial_loss,
        final_loss=final_loss,
        iters_used=iters_used,
        wall_s=wall_s,
        solve_profile=solve_profile,
        optimized_shape=optimized_shape,
        original_shape=(orig_h, orig_w),
        downsample_scale=float(downsample_scale),
    )


__all__ = ["SolverResult", "run_s5_solver", "summarise_impressions"]
