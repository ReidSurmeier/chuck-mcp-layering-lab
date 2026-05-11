"""Unit tests for :mod:`backend.adapters.mokuhanga_zip_emitter`."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

import pytest

from adapters.mokuhanga_zip_emitter import emit_mokuhanga_zip


# --- fixtures ---------------------------------------------------------------

_PNG_STUB = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_SVG_STUB = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'


def _make_plate(rank: int, hex_code: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "hex": hex_code,
        "name": f"plate_{rank}",
        "color": [0, 0, 0],
        "width": 64,
        "height": 48,
        "png_bytes": _PNG_STUB,
        "svg": _SVG_STUB,
    }


def _make_palette(*hexes: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"pig_{i}",
            "name": f"pigment_{i}",
            "rgb": [0, 0, 0],
            "hex": h,
            "alpha_prior": 0.85,
        }
        for i, h in enumerate(hexes)
    ]


def _make_impressions(n: int, block_assignment: dict[str, int]) -> list[dict[str, Any]]:
    pigment_ids = list(block_assignment.keys())
    return [
        {
            "step": i + 1,
            "block_id": block_assignment[pigment_ids[i % len(pigment_ids)]],
            "pigment_id": pigment_ids[i % len(pigment_ids)],
            "hex": f"#{(0x111111 * (i + 1)) & 0xFFFFFF:06x}",
            "coverage_pct": round(10.0 + i * 1.5, 2),
            "luminance": round(0.1 * (i + 1), 3),
        }
        for i in range(n)
    ]


def _open_zip(blob: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(blob), "r")


# --- tests ------------------------------------------------------------------


def test_empty_plates_produces_minimal_zip() -> None:
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=[],
        pigment_palette=[],
        block_assignment={},
        impressions=[],
        impression_previews=[],
        block_svgs={},
    )
    with _open_zip(blob) as zf:
        names = set(zf.namelist())
        assert "composite.png" in names
        assert "manifest.json" in names
        assert not any(n.startswith("png/") for n in names)
        assert not any(n.startswith("svg/") for n in names)
        assert not any(n.startswith("blocks/") for n in names)
        assert not any(n.startswith("impressions/") for n in names)
        assert "print_order.csv" not in names
        assert zf.testzip() is None


def test_three_plates_two_blocks_full_layout() -> None:
    plates = [_make_plate(1, "#ff0000"), _make_plate(2, "#00ff00"), _make_plate(3, "#0000ff")]
    palette = _make_palette("#ff0000", "#00ff00", "#0000ff")
    block_assignment = {"pig_0": 0, "pig_1": 0, "pig_2": 1}
    impressions = _make_impressions(4, block_assignment)
    previews = [_PNG_STUB] * 4
    block_svgs = {0: _SVG_STUB, 1: _SVG_STUB}

    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=plates,
        pigment_palette=palette,
        block_assignment=block_assignment,
        impressions=impressions,
        impression_previews=previews,
        block_svgs=block_svgs,
    )
    with _open_zip(blob) as zf:
        names = set(zf.namelist())
        png_entries = [n for n in names if n.startswith("png/")]
        svg_entries = [n for n in names if n.startswith("svg/")]
        block_entries = [n for n in names if n.startswith("blocks/")]
        impression_entries = [n for n in names if n.startswith("impressions/")]

        assert len(png_entries) == 3
        assert len(svg_entries) == 3
        assert len(block_entries) == 2
        assert len(impression_entries) == 4
        assert "print_order.csv" in names
        assert "manifest.json" in names
        assert "composite.png" in names

        assert "png/01_ff0000.png" in names
        assert "svg/01_ff0000.svg" in names
        assert "blocks/block_00.svg" in names
        assert "blocks/block_01.svg" in names
        assert "impressions/step_001.png" in names
        assert "impressions/step_004.png" in names


def test_manifest_editor_extension_shape() -> None:
    plates = [_make_plate(1, "#abcdef"), _make_plate(2, "#123456")]
    palette = _make_palette("#abcdef", "#123456")
    block_assignment = {"pig_0": 0, "pig_1": 1}
    impressions = _make_impressions(2, block_assignment)

    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=plates,
        pigment_palette=palette,
        block_assignment=block_assignment,
        impressions=impressions,
        impression_previews=[_PNG_STUB, _PNG_STUB],
        block_svgs={0: _SVG_STUB, 1: _SVG_STUB},
    )
    with _open_zip(blob) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["num_plates"] == 2
    assert manifest["width"] == 64
    assert manifest["height"] == 48
    assert "editor" in manifest
    editor = manifest["editor"]
    assert editor["schemaVersion"] == "1.0.0"
    assert editor["mode"] == "v21_mokuhanga"
    assert editor["print_order_mode"] == "light_to_dark"
    assert editor["block_count"] == 2
    assert editor["palette"] == palette
    assert editor["block_assignment"] == block_assignment
    assert editor["impressions"] == impressions


def test_editor_block_count_matches_unique_assignments() -> None:
    # 4 pigments, but only 2 unique blocks
    block_assignment = {"pig_0": 5, "pig_1": 5, "pig_2": 9, "pig_3": 9}
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=[],
        pigment_palette=[],
        block_assignment=block_assignment,
        impressions=[],
        impression_previews=[],
        block_svgs={},
    )
    with _open_zip(blob) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["editor"]["block_count"] == len(set(block_assignment.values())) == 2


def test_print_order_csv_columns_and_rows() -> None:
    block_assignment = {"pig_0": 0, "pig_1": 1}
    impressions = _make_impressions(3, block_assignment)

    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=[],
        pigment_palette=[],
        block_assignment=block_assignment,
        impressions=impressions,
        impression_previews=[],
        block_svgs={},
    )
    with _open_zip(blob) as zf:
        csv_bytes = zf.read("print_order.csv")

    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    header = next(reader)
    assert header == ["step", "block_id", "pigment_id", "hex", "coverage_pct", "luminance"]
    rows = list(reader)
    assert len(rows) == 3
    # spot-check row content matches impressions
    assert rows[0][0] == str(impressions[0]["step"])
    assert rows[0][2] == impressions[0]["pigment_id"]


def test_zipfile_testzip_returns_none() -> None:
    plates = [_make_plate(1, "#ff0000")]
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=plates,
        pigment_palette=_make_palette("#ff0000"),
        block_assignment={"pig_0": 0},
        impressions=_make_impressions(1, {"pig_0": 0}),
        impression_previews=[_PNG_STUB],
        block_svgs={0: _SVG_STUB},
    )
    with _open_zip(blob) as zf:
        assert zf.testzip() is None


def test_cnc_backwards_compat_glob() -> None:
    """cnc tool reads v20 svg entries via glob; emitter must keep that layout."""
    plates = [_make_plate(1, "#ff0000"), _make_plate(2, "#00ff00"), _make_plate(3, "#0000ff")]
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=plates,
        pigment_palette=_make_palette("#ff0000", "#00ff00", "#0000ff"),
        block_assignment={"pig_0": 0, "pig_1": 1, "pig_2": 2},
        impressions=[],
        impression_previews=[],
        block_svgs={0: _SVG_STUB, 1: _SVG_STUB, 2: _SVG_STUB},
    )
    with _open_zip(blob) as zf:
        names = zf.namelist()
        legacy_svgs = sorted(n for n in names if n.startswith("svg/") and n.endswith(".svg"))
        legacy_pngs = sorted(n for n in names if n.startswith("png/") and n.endswith(".png"))
    assert legacy_svgs == ["svg/01_ff0000.svg", "svg/02_00ff00.svg", "svg/03_0000ff.svg"]
    assert legacy_pngs == ["png/01_ff0000.png", "png/02_00ff00.png", "png/03_0000ff.png"]


def test_source_manifest_extra_merged_but_editor_protected() -> None:
    extra = {"upscaled": True, "editor": {"mode": "hijacked"}}
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=[],
        pigment_palette=[],
        block_assignment={},
        impressions=[],
        impression_previews=[],
        block_svgs={},
        source_manifest_extra=extra,
    )
    with _open_zip(blob) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest.get("upscaled") is True
    # editor block must be the emitter's, not the extras' override
    assert manifest["editor"]["mode"] == "v21_mokuhanga"


def test_block_svgs_zero_padded_filenames() -> None:
    block_svgs = {0: _SVG_STUB, 1: _SVG_STUB, 12: _SVG_STUB}
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=[],
        pigment_palette=[],
        block_assignment={},
        impressions=[],
        impression_previews=[],
        block_svgs=block_svgs,
    )
    with _open_zip(blob) as zf:
        names = set(zf.namelist())
    assert "blocks/block_00.svg" in names
    assert "blocks/block_01.svg" in names
    assert "blocks/block_12.svg" in names


def test_custom_schema_version() -> None:
    blob = emit_mokuhanga_zip(
        composite_png_bytes=_PNG_STUB,
        plates=[],
        pigment_palette=[],
        block_assignment={},
        impressions=[],
        impression_previews=[],
        block_svgs={},
        schema_version="2.5.1",
    )
    with _open_zip(blob) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["editor"]["schemaVersion"] == "2.5.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
