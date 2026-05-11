"""Unit tests for palette_extract."""

from __future__ import annotations

import time

import numpy as np
import pytest

from algorithms.decomposition.palette_extract import (
    MIXBOX_PIGMENTS,
    Pigment,
    build_palette,
    snap_to_mixbox,
    srgb_to_oklab,
)


class TestMixboxPalette:
    def test_has_exactly_thirteen_pigments(self) -> None:
        assert len(MIXBOX_PIGMENTS) == 13

    def test_pigment_ids_unique(self) -> None:
        ids = [p.id for p in MIXBOX_PIGMENTS]
        assert len(set(ids)) == 13

    def test_pigment_names_unique(self) -> None:
        names = [p.name for p in MIXBOX_PIGMENTS]
        assert len(set(names)) == 13

    def test_hex_matches_rgb(self) -> None:
        for p in MIXBOX_PIGMENTS:
            r, g, b = p.rgb
            expected = f"#{r:02x}{g:02x}{b:02x}"
            assert p.hex.lower() == expected, f"{p.id}: hex {p.hex} != rgb {p.rgb}"

    def test_pigments_are_frozen(self) -> None:
        pig = MIXBOX_PIGMENTS[0]
        with pytest.raises(AttributeError):
            pig.id = "mutated"  # type: ignore[misc]


class TestSrgbToOklab:
    def test_ottosson_reference_red(self) -> None:
        """Pure red (255,0,0) should match the published Ottosson values."""
        red = np.array([255, 0, 0], dtype=np.uint8)
        lab = srgb_to_oklab(red)
        expected = np.array([0.628, 0.225, 0.126])
        assert np.allclose(lab, expected, atol=1e-3)

    def test_accepts_float_input(self) -> None:
        red_u8 = np.array([255, 0, 0], dtype=np.uint8)
        red_f = np.array([1.0, 0.0, 0.0])
        assert np.allclose(srgb_to_oklab(red_u8), srgb_to_oklab(red_f), atol=1e-6)

    def test_batch_shape_preserved(self) -> None:
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        lab = srgb_to_oklab(colors)
        assert lab.shape == (3, 3)

    def test_rejects_bad_trailing_dim(self) -> None:
        with pytest.raises(ValueError):
            srgb_to_oklab(np.zeros((4, 2)))


class TestSnapToMixbox:
    @pytest.mark.parametrize("pigment", MIXBOX_PIGMENTS)
    def test_exact_pigment_snaps_to_itself(self, pigment: Pigment) -> None:
        rgb_01 = np.array(pigment.rgb, dtype=np.float64) / 255.0
        result = snap_to_mixbox(rgb_01[None, :])
        assert result == [pigment]

    def test_halfway_between_yellows_snaps_to_a_yellow(self) -> None:
        """Halfway between cadmium_yellow and hansa_yellow should snap to one of them."""
        cad = np.array(MIXBOX_PIGMENTS[0].rgb, dtype=np.float64)
        han = np.array(MIXBOX_PIGMENTS[1].rgb, dtype=np.float64)
        mid_01 = ((cad + han) / 2.0) / 255.0
        result = snap_to_mixbox(mid_01[None, :])
        assert len(result) == 1
        assert result[0].id in {"cadmium_yellow", "hansa_yellow"}

    def test_returns_one_pigment_per_input(self) -> None:
        palette = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        result = snap_to_mixbox(palette)
        assert len(result) == 4
        assert all(isinstance(p, Pigment) for p in result)

    def test_red_snaps_to_red_family(self) -> None:
        red = np.array([[1.0, 0.0, 0.0]])
        result = snap_to_mixbox(red)
        assert result[0].id in {"cadmium_red", "cadmium_orange", "quinacridone_magenta"}

    def test_accepts_1d_input(self) -> None:
        single = np.array([1.0, 0.0, 0.0])
        result = snap_to_mixbox(single)
        assert len(result) == 1

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            snap_to_mixbox(np.zeros((4, 5)))

    def test_custom_available_pigments(self) -> None:
        only_blues = (MIXBOX_PIGMENTS[6], MIXBOX_PIGMENTS[7], MIXBOX_PIGMENTS[8])
        red = np.array([[1.0, 0.0, 0.0]])
        result = snap_to_mixbox(red, available_pigments=only_blues)
        assert result[0] in only_blues


class TestBuildPalette:
    def test_dedup_preserves_order(self) -> None:
        # Use exact mixbox RGBs so we know the snap targets.
        cad_yellow = np.array(MIXBOX_PIGMENTS[0].rgb) / 255.0
        cad_red = np.array(MIXBOX_PIGMENTS[3].rgb) / 255.0
        cobalt_blue = np.array(MIXBOX_PIGMENTS[7].rgb) / 255.0
        palette = np.stack([cad_yellow, cad_red, cad_yellow, cobalt_blue, cad_red])
        result = build_palette(palette, deduplicate=True)
        ids = [p.id for p in result]
        assert ids == ["cadmium_yellow", "cadmium_red", "cobalt_blue"]

    def test_no_dedup_keeps_duplicates(self) -> None:
        cad_yellow = np.array(MIXBOX_PIGMENTS[0].rgb) / 255.0
        palette = np.stack([cad_yellow, cad_yellow, cad_yellow])
        result = build_palette(palette, deduplicate=False)
        assert len(result) == 3
        assert all(p.id == "cadmium_yellow" for p in result)

    def test_dedup_default_true(self) -> None:
        cad_yellow = np.array(MIXBOX_PIGMENTS[0].rgb) / 255.0
        palette = np.stack([cad_yellow, cad_yellow])
        assert len(build_palette(palette)) == 1


class TestPerformance:
    def test_snap_100_colors_under_50ms(self) -> None:
        rng = np.random.default_rng(42)
        palette = rng.random((100, 3))
        # Warm up the colour-science conversion path.
        snap_to_mixbox(palette[:1])
        start = time.perf_counter()
        result = snap_to_mixbox(palette)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert len(result) == 100
        assert elapsed_ms < 50, f"snap took {elapsed_ms:.2f}ms (>50ms budget)"
