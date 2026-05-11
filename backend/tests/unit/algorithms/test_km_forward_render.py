"""Tests for km_forward_render — the master gate of the decomposition pipeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from algorithms.decomposition.km_forward_render import (
    blend_two_pigments,
    composite_delta_e2000,
    forward_render_km,
    stack_pigments,
)

# Canonical pigment swatches
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CADMIUM_RED = (227, 0, 34)
ULTRAMARINE = (18, 10, 143)
CADMIUM_YELLOW = (255, 246, 0)
PHTHALO_GREEN = (0, 100, 63)


# ----------------- blend_two_pigments -----------------


def test_blend_t0_returns_first() -> None:
    assert blend_two_pigments(CADMIUM_RED, ULTRAMARINE, 0.0) == CADMIUM_RED


def test_blend_t1_returns_second() -> None:
    assert blend_two_pigments(CADMIUM_RED, ULTRAMARINE, 1.0) == ULTRAMARINE


def test_blend_yellow_blue_midpoint_is_greenish() -> None:
    r, g, b = blend_two_pigments(CADMIUM_YELLOW, ULTRAMARINE, 0.5)
    # mixbox famously produces green from yellow+blue (unlike naive RGB lerp)
    assert g > r and g > b, f"expected green-dominant; got ({r}, {g}, {b})"


# ----------------- stack_pigments -----------------


def test_stack_no_pigments_returns_substrate() -> None:
    assert stack_pigments([], [], substrate_rgb=WHITE) == WHITE


def test_stack_all_alpha_zero_returns_substrate() -> None:
    out = stack_pigments(
        [CADMIUM_RED, ULTRAMARINE], [0.0, 0.0], substrate_rgb=WHITE
    )
    assert out == WHITE


def test_stack_full_alpha_last_pigment_wins() -> None:
    out = stack_pigments(
        [CADMIUM_RED, ULTRAMARINE], [1.0, 1.0], substrate_rgb=WHITE
    )
    # last fully opaque pigment should dominate
    assert out == ULTRAMARINE


def test_stack_red_full_alpha_on_white_is_red() -> None:
    out = stack_pigments([CADMIUM_RED], [1.0], substrate_rgb=WHITE)
    assert out == CADMIUM_RED


def test_stack_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        stack_pigments([CADMIUM_RED, ULTRAMARINE], [0.5])


# ----------------- forward_render_km basics -----------------


def _alpha_const(h: int, w: int, v: float) -> np.ndarray:
    return np.full((h, w), float(v), dtype=np.float64)


def test_forward_returns_uint8_hw3() -> None:
    h, w = 32, 24
    alphas = [_alpha_const(h, w, 0.0)]
    out = forward_render_km(alphas, [CADMIUM_RED])
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8


def test_forward_all_alpha_zero_is_substrate() -> None:
    h, w = 16, 16
    alphas = [_alpha_const(h, w, 0.0) for _ in range(3)]
    palette = [CADMIUM_RED, ULTRAMARINE, CADMIUM_YELLOW]
    out = forward_render_km(alphas, palette, substrate_rgb=WHITE)
    assert np.all(out == 255)


def test_forward_single_pigment_full_alpha_matches_pigment() -> None:
    h, w = 16, 16
    out = forward_render_km(
        [_alpha_const(h, w, 1.0)], [CADMIUM_RED], substrate_rgb=WHITE
    )
    # mixbox quantization tolerance
    diff = np.abs(out.astype(int) - np.array(CADMIUM_RED).reshape(1, 1, 3))
    assert diff.max() <= 2


def test_forward_single_pigment_ramp_is_monotone() -> None:
    h, w = 1, 256
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float64).reshape(h, w)
    out = forward_render_km([ramp], [CADMIUM_RED], substrate_rgb=WHITE)
    # red channel: white(255) -> red(227): nearly flat
    # green channel: 255 -> 0: must trend strongly downward
    g = out[0, :, 1].astype(int)
    assert g[0] >= 250 and g[-1] <= 5
    # quantization LUT introduces small non-monotone wiggles;
    # require >=90% of steps non-increasing and zero positive deltas > 5
    diffs = np.diff(g)
    non_increasing_frac = float(np.sum(diffs <= 0)) / float(diffs.size)
    assert non_increasing_frac >= 0.90, f"only {non_increasing_frac:.2%} non-increasing"
    assert int(np.max(diffs)) <= 5, f"max upward step {int(np.max(diffs))} > 5"


def test_forward_print_order_matters() -> None:
    h, w = 32, 32
    a_half = _alpha_const(h, w, 0.5)
    palette = [CADMIUM_RED, ULTRAMARINE]
    fwd = forward_render_km([a_half, a_half], palette, print_order=[0, 1])
    rev = forward_render_km([a_half, a_half], palette, print_order=[1, 0])
    # different last pigment with partial alpha -> different composite
    assert not np.array_equal(fwd, rev)


def test_forward_deterministic() -> None:
    h, w = 24, 24
    a = _alpha_const(h, w, 0.6)
    palette = [CADMIUM_RED, ULTRAMARINE]
    out1 = forward_render_km([a, a], palette)
    out2 = forward_render_km([a, a], palette)
    assert np.array_equal(out1, out2)


def test_forward_shape_mismatch_raises() -> None:
    bad = [_alpha_const(16, 16, 0.5), _alpha_const(16, 8, 0.5)]
    with pytest.raises(ValueError):
        forward_render_km(bad, [CADMIUM_RED, ULTRAMARINE])


def test_forward_empty_alphas_raises() -> None:
    with pytest.raises(ValueError):
        forward_render_km([], [])


def test_forward_palette_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        forward_render_km(
            [_alpha_const(8, 8, 0.5)], [CADMIUM_RED, ULTRAMARINE]
        )


# ----------------- composite_delta_e2000 -----------------


def test_delta_e_identical_is_zero() -> None:
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    mean, p95 = composite_delta_e2000(img, img)
    assert mean < 1e-6
    assert p95 < 1e-6


def test_delta_e_white_vs_black_is_large() -> None:
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    mean, p95 = composite_delta_e2000(white, black)
    assert mean > 50  # max-distance pair in Lab
    assert p95 >= mean


def test_delta_e_shape_mismatch_raises() -> None:
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.zeros((8, 9, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        composite_delta_e2000(a, b)


# ----------------- roundtrip + perf (master gate proxies) -----------------


def test_forward_roundtrip_stub_decompose_under_loose_threshold() -> None:
    """Stub decomposition: synthesize known 3-pigment image, recover same alphas."""
    h, w = 64, 64
    palette = [CADMIUM_RED, ULTRAMARINE, CADMIUM_YELLOW]
    rng = np.random.default_rng(42)
    alphas = [rng.random((h, w)).astype(np.float64) * 0.7 for _ in palette]
    rendered = forward_render_km(alphas, palette, substrate_rgb=WHITE)
    # "decompose stub" = identity: feed alphas straight back
    re_rendered = forward_render_km(alphas, palette, substrate_rgb=WHITE)
    mean, _ = composite_delta_e2000(rendered, re_rendered)
    # identity decompose should be exact within quantization
    assert mean < 30.0  # loose for stub, tight (<5) once real decomp exists


def test_forward_perf_500x500_13_pigments_under_5s() -> None:
    h, w = 500, 500
    rng = np.random.default_rng(0)
    palette = [
        (255, 255, 255), (0, 0, 0), (227, 0, 34), (18, 10, 143),
        (255, 246, 0), (0, 100, 63), (245, 130, 32), (140, 70, 200),
        (255, 192, 203), (60, 30, 20), (100, 180, 230), (200, 200, 50),
        (50, 200, 100),
    ]
    alphas = [rng.random((h, w)).astype(np.float64) * 0.5 for _ in palette]
    t0 = time.perf_counter()
    out = forward_render_km(alphas, palette)
    dt = time.perf_counter() - t0
    assert out.shape == (h, w, 3)
    assert dt < 5.0, f"500x500 x 13 pigments took {dt:.2f}s (budget 5s)"
