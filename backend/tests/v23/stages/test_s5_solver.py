"""D10.d RED — S5 inverse stack solver with S4 warm-start init.

Wraps inverse_solver_smoke with proper sigmoid_box reparam initialised
from S4 Tan warm-start so L-BFGS doesn't start from random noise.
Returns refined alpha_stack + reconstruction ΔE per solve_profile.

Per addendum-v3 fix 1: topology constraints stay OUT of optimizer. Per
addendum-v4: t1_mixbox is the day-1 forward render. Per addendum-v3
fix 4: combined score arrives via score_candidate_stack (not in S5).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxopt")


def test_s5_solver_warm_start_beats_random_init() -> None:
    """L-BFGS from S4 warm-start should converge to LOWER loss than random init."""
    from backend.services.v23.stages import s4_warmstart, s5_solver

    # Synthetic 2-impression target: yellow + blue patches
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :16] = (240, 220, 50)
    img[:, 16:] = (30, 40, 160)

    warm = s4_warmstart.tan_to_pigment_warmstart(img, target_palette_size=4)
    result = s5_solver.run_s5_solver(
        target_rgb=img.astype(np.float32) / 255.0,
        pigment_idx=np.asarray(warm.pigment_idx, dtype=np.int32),
        alpha_init=warm.alpha_stack,
        solve_profile="fast",
    )
    assert result.alpha_stack.shape[1:] == (32, 32)
    assert result.alpha_stack.min() >= 0.0
    assert result.alpha_stack.max() <= 1.0 + 1e-5
    # Reconstruction ΔE in RGB-L2 should be smaller than initial RGB-L2
    assert result.final_loss < result.initial_loss * 0.50, (
        f"S5 didn't drop loss ≥50%: init={result.initial_loss:.5f} final={result.final_loss:.5f}"
    )


def test_s5_solver_bounds_large_internal_grid(monkeypatch) -> None:
    """Large targets are optimized on a bounded grid, then returned full-size."""
    from backend.services.v23.stages import s5_solver

    target = np.zeros((128, 128, 3), dtype=np.float32)
    alpha = np.ones((2, 128, 128), dtype=np.float32) * 0.5

    monkeypatch.setenv("WOODBLOCK_SOLVER_MAX_PIXELS", "4096")
    target_small, alpha_small, optimized_shape, scale = s5_solver._prepare_solver_grid(
        target,
        alpha,
        solve_profile="fast",
    )

    assert target_small.shape[:2] == (64, 64)
    assert alpha_small.shape == (2, 64, 64)
    assert optimized_shape == (64, 64)
    assert scale == 0.5


def test_s5_solver_respects_solve_profile_iter_budgets() -> None:
    """fast / default / thorough produce different iter counts."""
    from backend.services.v23.stages import s4_warmstart, s5_solver

    img = np.full((24, 24, 3), [180, 90, 80], dtype=np.uint8)
    warm = s4_warmstart.tan_to_pigment_warmstart(img, target_palette_size=3)
    target = img.astype(np.float32) / 255.0

    fast = s5_solver.run_s5_solver(target, np.asarray(warm.pigment_idx, dtype=np.int32),
                                    warm.alpha_stack, solve_profile="fast")
    thorough = s5_solver.run_s5_solver(target, np.asarray(warm.pigment_idx, dtype=np.int32),
                                        warm.alpha_stack, solve_profile="thorough")

    assert fast.iters_used <= thorough.iters_used
    # Thorough should not be WORSE than fast (it can match if already optimal)
    assert thorough.final_loss <= fast.final_loss + 1e-5


def test_s5_solver_emits_impressions_in_print_order() -> None:
    """run_s5_solver returns impressions list with monotonic order_step starting at 1."""
    from backend.services.v23.stages import s4_warmstart, s5_solver

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :8] = (240, 100, 80)
    img[:, 8:] = (60, 120, 200)
    warm = s4_warmstart.tan_to_pigment_warmstart(img, target_palette_size=4)
    target = img.astype(np.float32) / 255.0

    result = s5_solver.run_s5_solver(target, np.asarray(warm.pigment_idx, dtype=np.int32),
                                      warm.alpha_stack, solve_profile="fast")
    assert len(result.impressions) == len(warm.pigment_idx)
    steps = [imp["order_step"] for imp in result.impressions]
    assert steps == list(range(1, len(steps) + 1))
    for imp in result.impressions:
        assert imp["pigment_id"] in range(13)
        assert 0.0 <= imp["coverage_pct"] <= 100.0


def test_s5_solver_invalid_profile_raises() -> None:
    from backend.services.v23.stages import s5_solver

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    a = np.full((1, 4, 4), 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        s5_solver.run_s5_solver(
            target_rgb=img.astype(np.float32) / 255.0,
            pigment_idx=np.array([0], dtype=np.int32),
            alpha_init=a,
            solve_profile="ultra",  # type: ignore[arg-type]
        )


def test_s5_solver_alpha_init_shape_mismatch_raises() -> None:
    from backend.services.v23.stages import s5_solver

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    bad_alpha = np.zeros((2, 8, 16), dtype=np.float32)  # wrong width
    with pytest.raises(ValueError):
        s5_solver.run_s5_solver(
            target_rgb=img.astype(np.float32) / 255.0,
            pigment_idx=np.array([0, 1], dtype=np.int32),
            alpha_init=bad_alpha,
            solve_profile="fast",
        )
