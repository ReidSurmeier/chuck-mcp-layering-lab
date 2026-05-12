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
from PIL import Image

from backend.services.v23.core import forward_render_jax

_VALID_SOLVE_PROFILES = ("fast", "default", "thorough")
_PROFILE_ITERS = {"fast": 60, "default": 180, "thorough": 400}
_PROFILE_MAX_PIXELS = {"fast": 256_000, "default": 512_000, "thorough": 768_000}


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


def _sigmoid_box(alpha_raw: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(alpha_raw)


def _solver_loss(
    alpha_raw: jnp.ndarray,
    target_rgb: jnp.ndarray,
    pigment_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Sum-squared error in RGB space (matches inverse_solver_smoke).

    ``jnp.sum`` keeps gradient magnitude large enough for L-BFGS line search;
    8-term Oklab loss + sparsity + TV land in a follow-up commit when the
    Tier-1 gate tightens the ΔE target.
    """
    alpha = _sigmoid_box(alpha_raw)
    alpha_hwm = jnp.transpose(alpha, (1, 2, 0))  # (M, H, W) -> (H, W, M)
    rgb = forward_render_jax.forward_render(alpha_hwm, pigment_idx)
    return jnp.sum((rgb - target_rgb) ** 2)


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

    iters = _PROFILE_ITERS[solve_profile]
    target_solve, alpha_solve, optimized_shape, downsample_scale = _prepare_solver_grid(
        target_rgb,
        alpha_init,
        solve_profile=solve_profile,
    )
    # Convert α from [0,1] to logit space so unconstrained L-BFGS works
    clipped = np.clip(alpha_solve, 0.02, 0.98)
    init_raw = np.log(clipped / (1.0 - clipped)).astype(np.float32)

    target_j = jnp.asarray(target_solve, dtype=jnp.float32)
    pigments_j = jnp.asarray(pigment_idx, dtype=jnp.int32)

    def loss_fn(a_raw):
        return _solver_loss(a_raw, target_j, pigments_j)

    initial_loss = float(loss_fn(jnp.asarray(init_raw)))
    t0 = time.perf_counter()
    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=iters, tol=1e-7)
    result = solver.run(jnp.asarray(init_raw))
    wall_s = time.perf_counter() - t0
    alpha_raw_final = result.params
    alpha_stack_solve = np.asarray(jax.nn.sigmoid(alpha_raw_final))
    alpha_stack = (
        _resize_alpha_stack(alpha_stack_solve, (orig_w, orig_h))
        if optimized_shape != (orig_h, orig_w)
        else alpha_stack_solve
    )
    final_loss = float(loss_fn(alpha_raw_final))

    pigment_tuple = tuple(int(p) for p in pigment_idx.tolist())
    impressions = _alphas_to_impressions(alpha_stack, pigment_tuple)
    iters_used = (
        int(getattr(result.state, "iter_num", iters))
        if hasattr(result, "state")
        else iters
    )

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


__all__ = ["SolverResult", "run_s5_solver"]
