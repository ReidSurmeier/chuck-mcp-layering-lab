"""S6.c — ΔE-gated printability repair before block packing/vector export."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from skimage import measure

from backend.services.v23.core import color, forward_render_jax, topology_repair


@dataclass(frozen=True)
class PrintabilityRepairResult:
    """Output of :func:`repair_for_printability`."""

    alpha_stack: NDArray[np.float32]
    diagnostics: dict[str, Any]


def _render(alpha_stack: NDArray[np.float32], pigment_idx: NDArray[np.int32]) -> NDArray[np.float32]:
    alpha_hwm = np.transpose(alpha_stack, (1, 2, 0))
    return np.asarray(forward_render_jax.forward_render(
        jnp.asarray(alpha_hwm, dtype=jnp.float32),
        jnp.asarray(pigment_idx, dtype=jnp.int32),
    )).astype(np.float32)


def _component_counts(
    alpha_stack: NDArray[np.float32],
    *,
    threshold: float,
    tiny_px: int,
) -> tuple[int, int]:
    total = 0
    tiny = 0
    for alpha in alpha_stack:
        labels = measure.label(alpha >= threshold, connectivity=2)
        n = int(labels.max())
        total += n
        if n <= 0:
            continue
        areas = [int(r.area) for r in measure.regionprops(labels)]
        tiny += int(sum(1 for area in areas if area < tiny_px))
    return total, tiny


def repair_for_printability(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    *,
    target_rgb: NDArray[np.float32],
    min_island_px: int = 60,
    close_radius: int = 1,
    vis_threshold: float = 0.08,
    de_mean_guard: float = 0.75,
    de_p95_guard: float = 2.25,
) -> PrintabilityRepairResult:
    """Remove tiny islands when visual regression stays within guard rails."""
    original_rgb = _render(alpha_stack, pigment_idx)
    original_de = color.delta_e_summary(original_rgb, target_rgb)
    original_components, original_tiny = _component_counts(
        alpha_stack,
        threshold=vis_threshold,
        tiny_px=min_island_px,
    )

    repaired = topology_repair.morph_repair_stack(
        alpha_stack,
        min_island_px=min_island_px,
        close_radius=close_radius,
        vis_threshold=vis_threshold,
    )
    repaired_rgb = _render(repaired, pigment_idx)
    repaired_de = color.delta_e_summary(repaired_rgb, target_rgb)
    repaired_components, repaired_tiny = _component_counts(
        repaired,
        threshold=vis_threshold,
        tiny_px=min_island_px,
    )

    mean_delta = float(repaired_de["dE_mean"] - original_de["dE_mean"])
    p95_delta = float(repaired_de["dE_p95"] - original_de["dE_p95"])
    accepted = mean_delta <= de_mean_guard and p95_delta <= de_p95_guard
    diagnostics = {
        "applied": bool(accepted),
        "min_island_px": int(min_island_px),
        "close_radius": int(close_radius),
        "vis_threshold": float(vis_threshold),
        "de_mean_guard": float(de_mean_guard),
        "de_p95_guard": float(de_p95_guard),
        "original_dE_mean": round(float(original_de["dE_mean"]), 3),
        "original_dE_p95": round(float(original_de["dE_p95"]), 3),
        "repaired_dE_mean": round(float(repaired_de["dE_mean"]), 3),
        "repaired_dE_p95": round(float(repaired_de["dE_p95"]), 3),
        "dE_mean_delta": round(mean_delta, 3),
        "dE_p95_delta": round(p95_delta, 3),
        "component_count_before": int(original_components),
        "component_count_after": int(repaired_components),
        "tiny_component_count_before": int(original_tiny),
        "tiny_component_count_after": int(repaired_tiny),
        "reason": (
            "repair accepted"
            if accepted
            else "repair rejected by dE regression guard"
        ),
    }
    return PrintabilityRepairResult(
        alpha_stack=(repaired if accepted else alpha_stack).astype(np.float32),
        diagnostics=diagnostics,
    )


__all__ = ["PrintabilityRepairResult", "repair_for_printability"]
