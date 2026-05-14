"""S6.c printability repair tests."""
from __future__ import annotations

import numpy as np


def test_printability_repair_removes_tiny_components_with_guard() -> None:
    from backend.services.v23.core import forward_render_jax
    from backend.services.v23.stages.s6c_printability_repair import repair_for_printability

    alpha = np.zeros((1, 48, 48), dtype=np.float32)
    alpha[0, 10:36, 10:36] = 0.55
    alpha[0, 2, 2] = 0.55
    alpha[0, 44, 44] = 0.55
    pigment_idx = np.array([0], dtype=np.int32)
    target = np.broadcast_to(forward_render_jax.PAPER_RGB, (48, 48, 3)).copy()

    result = repair_for_printability(
        alpha,
        pigment_idx,
        target_rgb=target.astype(np.float32),
        min_island_px=8,
        de_mean_guard=50.0,
        de_p95_guard=100.0,
    )

    assert result.diagnostics["applied"] is True
    assert result.diagnostics["tiny_component_count_after"] < result.diagnostics["tiny_component_count_before"]
    assert result.alpha_stack[0, 2, 2] == 0.0


def test_printability_repair_rejects_large_de_regression() -> None:
    from backend.services.v23.core import forward_render_jax
    from backend.services.v23.stages.s6c_printability_repair import repair_for_printability

    alpha = np.zeros((1, 48, 48), dtype=np.float32)
    alpha[0, 10:36, 10:36] = 0.55
    pigment_idx = np.array([0], dtype=np.int32)
    target = np.asarray(
        forward_render_jax.PAPER_RGB + alpha[0, ..., None] * (
            forward_render_jax.PIGMENT_TABLE[0] - forward_render_jax.PAPER_RGB
        ),
        dtype=np.float32,
    )

    result = repair_for_printability(
        alpha,
        pigment_idx,
        target_rgb=target,
        min_island_px=10_000,
        de_mean_guard=0.01,
        de_p95_guard=0.01,
    )

    assert result.diagnostics["applied"] is False
    np.testing.assert_array_equal(result.alpha_stack, alpha)
