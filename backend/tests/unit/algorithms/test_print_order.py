"""Unit tests for the mokuhanga print-order solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from algorithms.decomposition.print_order import (
    Impression,
    compute_luminance_oklab,
    order_impressions,
)


# ---------------------------------------------------------------------------
# compute_luminance_oklab
# ---------------------------------------------------------------------------


class TestComputeLuminanceOklab:
    def test_white_is_one(self) -> None:
        L = compute_luminance_oklab((255, 255, 255))
        assert math.isclose(L, 1.0, abs_tol=1e-3)

    def test_black_is_zero(self) -> None:
        L = compute_luminance_oklab((0, 0, 0))
        assert math.isclose(L, 0.0, abs_tol=1e-6)

    def test_red_oklab_l_known_value(self) -> None:
        # Ottosson's OKLab reference: pure sRGB red -> L ~= 0.628.
        L = compute_luminance_oklab((255, 0, 0))
        assert math.isclose(L, 0.628, abs_tol=0.01)

    def test_accepts_float_normalized_input(self) -> None:
        L_int = compute_luminance_oklab((128, 128, 128))
        L_float = compute_luminance_oklab(np.array([128 / 255.0] * 3))
        assert math.isclose(L_int, L_float, abs_tol=1e-9)

    def test_accepts_ndarray(self) -> None:
        L = compute_luminance_oklab(np.array([255, 255, 255]))
        assert math.isclose(L, 1.0, abs_tol=1e-3)

    def test_ordering_white_gray_black(self) -> None:
        L_white = compute_luminance_oklab((255, 255, 255))
        L_gray = compute_luminance_oklab((128, 128, 128))
        L_black = compute_luminance_oklab((0, 0, 0))
        assert L_white > L_gray > L_black

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_luminance_oklab(np.array([1.0, 0.0]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_luminance_oklab(np.array([]))

    def test_2d_array_returns_mean_L(self) -> None:
        arr = np.array([[255, 255, 255], [0, 0, 0]])
        L = compute_luminance_oklab(arr)
        # Mean of L(white)+L(black) over 2 samples ~= 0.5.
        assert 0.4 < L < 0.6


# ---------------------------------------------------------------------------
# order_impressions
# ---------------------------------------------------------------------------


def _palette_sumi_yellow_white() -> dict[str, dict]:
    """Three pigments spanning the L range used by traditional mokuhanga."""
    return {
        "sumi-black": {"hex": "#000000", "rgb": (0, 0, 0), "coverage_pct": 30.0},
        "yellow": {"hex": "#ffe600", "rgb": (255, 230, 0), "coverage_pct": 20.0},
        "white": {"hex": "#ffffff", "rgb": (255, 255, 255), "coverage_pct": 10.0},
    }


class TestOrderImpressions:
    def test_light_to_dark_default_order(self) -> None:
        meta = _palette_sumi_yellow_white()
        assignments = {"sumi-black": 1, "yellow": 2, "white": 3}
        result = order_impressions(assignments, meta)

        assert [imp.pigment_id for imp in result] == ["white", "yellow", "sumi-black"]
        assert [imp.step for imp in result] == [1, 2, 3]
        # mokuhanga light→dark: luminance must be non-increasing
        # (light first → dark last → L decreases across steps).
        Ls = [imp.luminance_okL for imp in result]
        assert all(Ls[i] >= Ls[i + 1] for i in range(len(Ls) - 1))

    def test_direction_toggle_reverses(self) -> None:
        meta = _palette_sumi_yellow_white()
        assignments = {"sumi-black": 1, "yellow": 2, "white": 3}
        light_dark = order_impressions(assignments, meta, direction="light_to_dark")
        dark_light = order_impressions(assignments, meta, direction="dark_to_light")

        assert [imp.pigment_id for imp in dark_light] == [
            imp.pigment_id for imp in reversed(light_dark)
        ]
        # Steps are still 1..M sequentially even after reversal.
        assert [imp.step for imp in dark_light] == [1, 2, 3]

    def test_coverage_tiebreak_smaller_first(self) -> None:
        # Two pigments with identical color (=> identical L) but different coverage.
        meta = {
            "gray-big": {"hex": "#808080", "rgb": (128, 128, 128), "coverage_pct": 60.0},
            "gray-small": {"hex": "#808080", "rgb": (128, 128, 128), "coverage_pct": 5.0},
            "white": {"hex": "#ffffff", "rgb": (255, 255, 255), "coverage_pct": 10.0},
        }
        assignments = {"gray-big": 1, "gray-small": 2, "white": 3}
        result = order_impressions(assignments, meta)

        # White lightest -> first. Then equal-L grays: smaller coverage first.
        assert [imp.pigment_id for imp in result] == ["white", "gray-small", "gray-big"]

    def test_empty_input_returns_empty_list(self) -> None:
        assert order_impressions({}, {}) == []

    def test_steps_numbered_sequentially(self) -> None:
        meta = {
            f"p{i}": {
                "hex": f"#{i:02x}{i:02x}{i:02x}",
                "rgb": (i, i, i),
                "coverage_pct": float(i),
            }
            for i in (10, 50, 90, 130, 200)
        }
        assignments = {pid: idx for idx, pid in enumerate(meta)}
        result = order_impressions(assignments, meta)

        assert len(result) == 5
        assert [imp.step for imp in result] == [1, 2, 3, 4, 5]
        # All input pigments present exactly once.
        assert {imp.pigment_id for imp in result} == set(meta)

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError):
            order_impressions({"x": 1}, {"x": {"hex": "#000000", "rgb": (0, 0, 0), "coverage_pct": 0.0}},
                              direction="sideways")

    def test_missing_pigment_meta_raises(self) -> None:
        with pytest.raises(KeyError):
            order_impressions({"x": 1}, {}, direction="light_to_dark")

    def test_missing_rgb_field_raises(self) -> None:
        with pytest.raises(KeyError):
            order_impressions(
                {"x": 1},
                {"x": {"hex": "#000000", "coverage_pct": 5.0}},
                direction="light_to_dark",
            )

    def test_impression_is_frozen(self) -> None:
        imp = Impression(
            step=1,
            block_id=1,
            pigment_id="x",
            pigment_hex="#000000",
            coverage_pct=1.0,
            luminance_okL=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            imp.step = 2  # type: ignore[misc]

    def test_hex_falls_back_to_rgb_when_missing(self) -> None:
        meta = {
            "redish": {"rgb": (200, 30, 30), "coverage_pct": 12.0},
            "white": {"hex": "#FFFFFF", "rgb": (255, 255, 255), "coverage_pct": 8.0},
        }
        assignments = {"redish": 5, "white": 7}
        result = order_impressions(assignments, meta)

        by_id = {imp.pigment_id: imp for imp in result}
        # Auto-generated hex from rgb (lowercased, zero-padded).
        assert by_id["redish"].pigment_hex == "#c81e1e"
        # Provided hex is normalised to lowercase.
        assert by_id["white"].pigment_hex == "#ffffff"

    def test_block_id_preserved(self) -> None:
        meta = _palette_sumi_yellow_white()
        assignments = {"sumi-black": 42, "yellow": 7, "white": 99}
        result = order_impressions(assignments, meta)
        block_for_pigment = {imp.pigment_id: imp.block_id for imp in result}
        assert block_for_pigment == {"sumi-black": 42, "yellow": 7, "white": 99}
