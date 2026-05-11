"""Unit tests for the Tan 2017 RGB-geometry decomposition module."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from algorithms.decomposition.tan_rgb_geometry import (
    barycentric_decompose,
    decompose_image,
    extract_palette_hull,
)


# Locate the corpus fixture: env override > walk up to find corpus dir > skip
def _find_corpus_root() -> Path | None:
    env = __import__("os").environ.get("WOODBLOCK_REPO_ROOT")
    if env:
        p = Path(env) / "corpus"
        return p.parent if p.exists() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "corpus").is_dir():
            return parent
    return None

_REPO_ROOT = _find_corpus_root()
_FIXTURE = (
    _REPO_ROOT / "corpus" / "hiroshige_edo_116" / "original.jpeg"
    if _REPO_ROOT else Path("/_missing_fixture")
)


def _random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _is_valid_alpha(alphas: np.ndarray, tol: float = 1e-3) -> bool:
    if not np.all(np.isfinite(alphas)):
        return False
    if alphas.min() < -tol:
        return False
    sums = alphas.sum(axis=-1)
    return bool(np.all(np.abs(sums - 1.0) < tol))


# --- 1. Palette shape ------------------------------------------------------


@pytest.mark.parametrize("k", [4, 7, 13, 20])
def test_palette_shape(k: int) -> None:
    img = _random_image(64, 64, seed=1)
    palette = extract_palette_hull(img, target_palette_size=k)
    assert palette.ndim == 2
    assert palette.shape[1] == 3
    assert palette.shape[0] <= k
    assert palette.min() >= 0.0 - 1e-9
    assert palette.max() <= 1.0 + 1e-9


# --- 2. Single-color image -------------------------------------------------


def test_single_color_image_is_degenerate() -> None:
    img = np.full((32, 32, 3), 180, dtype=np.uint8)
    palette = extract_palette_hull(img, target_palette_size=5)
    # All palette entries collapse to the same color.
    unique = np.unique(np.round(palette * 255).astype(int), axis=0)
    assert unique.shape[0] == 1
    np.testing.assert_allclose(unique[0], [180, 180, 180])


# --- 3. Two-color ramp -----------------------------------------------------


def test_two_color_ramp_linear_progression() -> None:
    # Horizontal ramp from black to white.
    w = 64
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.stack([np.tile(ramp, (8, 1))] * 3, axis=-1)
    palette = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    alphas = barycentric_decompose(img, palette)
    assert alphas.shape == (8, w, 2)
    # Weight of "white" should rise monotonically left -> right.
    white_weight_row = alphas[0, :, 1]
    assert white_weight_row[0] < 0.05
    assert white_weight_row[-1] > 0.95
    diffs = np.diff(white_weight_row)
    assert np.all(diffs >= -1e-5)


# --- 4. Sum-to-one + non-negativity ---------------------------------------


def test_weights_nonneg_and_sum_to_one() -> None:
    img = _random_image(48, 48, seed=2)
    palette, alphas = decompose_image(img, target_palette_size=8)
    assert _is_valid_alpha(alphas)


# --- 5. Out-of-hull pixels still valid ------------------------------------


def test_out_of_hull_pixels_have_valid_weights() -> None:
    # Build a palette that *doesn't* contain the corners of the cube — any
    # very-saturated pixel will be outside the hull.
    palette = np.array(
        [
            [0.2, 0.2, 0.2],
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            [0.8, 0.8, 0.2],
        ],
        dtype=np.float64,
    )
    # Mix of inside + outside pixels.
    img = np.array(
        [
            [
                [255, 255, 255],  # outside (white)
                [0, 0, 0],  # outside (black)
                [128, 128, 128],  # inside-ish
                [200, 50, 50],  # near a vertex
            ]
        ],
        dtype=np.uint8,
    )
    alphas = barycentric_decompose(img, palette)
    assert alphas.shape == (1, 4, 5)
    assert _is_valid_alpha(alphas)
    assert not np.any(np.isnan(alphas))


# --- 6. Performance benchmark ---------------------------------------------


def test_perf_500x500_under_1_5s() -> None:
    img = _random_image(500, 500, seed=3)
    t0 = time.perf_counter()
    palette, alphas = decompose_image(img, target_palette_size=8)
    dt = time.perf_counter() - t0
    assert alphas.shape == (500, 500, palette.shape[0])
    assert dt < 1.5, f"500x500 decomposition took {dt:.3f}s (>1.5s budget)"


# --- 7. Real fixture: Hiroshige Edo plate ---------------------------------


@pytest.mark.skipif(
    not _FIXTURE.exists(), reason=f"fixture missing: {_FIXTURE}"
)
def test_real_fixture_hiroshige() -> None:
    from PIL import Image

    pil = Image.open(_FIXTURE).convert("RGB")
    # Down-scale aggressively so the test stays cheap.
    pil.thumbnail((400, 400))
    img = np.array(pil, dtype=np.uint8)
    palette, alphas = decompose_image(img, target_palette_size=10)
    assert palette.shape[1] == 3
    assert palette.shape[0] <= 10
    assert alphas.shape == (img.shape[0], img.shape[1], palette.shape[0])
    assert _is_valid_alpha(alphas, tol=5e-3)


# --- 8. Decompose accepts custom palette ----------------------------------


def test_decompose_image_accepts_custom_palette() -> None:
    img = _random_image(16, 16, seed=4)
    palette = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    out_palette, alphas = decompose_image(img, palette=palette)
    np.testing.assert_allclose(out_palette, palette)
    assert alphas.shape == (16, 16, 5)
    assert _is_valid_alpha(alphas)
