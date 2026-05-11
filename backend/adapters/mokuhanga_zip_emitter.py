"""v21 mokuhanga ZIP emitter.

Serializes v21 output into a ZIP container that:
  * preserves the v20 ``png/`` + ``svg/`` legacy layout (cnc tool reads via glob)
  * adds v21 physical block SVGs under ``blocks/``
  * adds cumulative impression previews under ``impressions/``
  * adds a ``print_order.csv`` index for the editor UI
  * embeds an ``editor.*`` extension block inside ``manifest.json``

Layout::

    composite.png
    png/<rank>_<hex>.png
    svg/<rank>_<hex>.svg
    blocks/block_<NN>.svg
    impressions/step_<NNN>.png
    print_order.csv
    manifest.json

Public API: :func:`emit_mokuhanga_zip`.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

_PRINT_ORDER_COLUMNS: tuple[str, ...] = (
    "step",
    "block_id",
    "pigment_id",
    "hex",
    "coverage_pct",
    "luminance",
)


def _strip_blob_fields(plate: dict[str, Any]) -> dict[str, Any]:
    """Return a manifest-safe copy of a v20 plate (no inline png_bytes/svg)."""
    return {k: v for k, v in plate.items() if k not in ("png_bytes", "svg", "mask", "image")}


def _hex_to_filename(hex_str: str) -> str:
    """Normalize a hex color to a filesystem-safe stem (no leading ``#``)."""
    return hex_str.lstrip("#").lower()


def _build_manifest(
    *,
    plates: list[dict[str, Any]],
    width: int,
    height: int,
    pigment_palette: list[dict[str, Any]],
    block_assignment: dict[str, int],
    impressions: list[dict[str, Any]],
    schema_version: str,
    source_manifest_extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the manifest dict with v20 base + v21 editor extension."""
    sanitized_plates = [_strip_blob_fields(p) for p in plates]
    block_count = len(set(block_assignment.values())) if block_assignment else 0

    manifest: dict[str, Any] = {
        "num_plates": len(plates),
        "plates": sanitized_plates,
        "width": width,
        "height": height,
        "editor": {
            "schemaVersion": schema_version,
            "mode": "v21_mokuhanga",
            "palette": pigment_palette,
            "block_count": block_count,
            "block_assignment": block_assignment,
            "impressions": impressions,
            "print_order_mode": "light_to_dark",
        },
    }
    if source_manifest_extra:
        for k, v in source_manifest_extra.items():
            if k == "editor":
                # don't let extras overwrite the v21 editor block
                continue
            manifest[k] = v
    return manifest


def _build_print_order_csv(impressions: list[dict[str, Any]]) -> bytes:
    """Render the impressions list as a CSV index."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(_PRINT_ORDER_COLUMNS))
    writer.writeheader()
    for imp in impressions:
        writer.writerow(
            {
                "step": imp.get("step", ""),
                "block_id": imp.get("block_id", ""),
                "pigment_id": imp.get("pigment_id", ""),
                "hex": imp.get("hex", ""),
                "coverage_pct": imp.get("coverage_pct", ""),
                "luminance": imp.get("luminance", ""),
            }
        )
    return buf.getvalue().encode("utf-8")


def _infer_dimensions(plates: list[dict[str, Any]]) -> tuple[int, int]:
    """Best-effort width/height inference from plate metadata."""
    for plate in plates:
        w = plate.get("width")
        h = plate.get("height")
        if isinstance(w, int) and isinstance(h, int):
            return w, h
    return 0, 0


def emit_mokuhanga_zip(
    *,
    composite_png_bytes: bytes,
    plates: list[dict[str, Any]],
    pigment_palette: list[dict[str, Any]],
    block_assignment: dict[str, int],
    impressions: list[dict[str, Any]],
    impression_previews: list[bytes],
    block_svgs: dict[int, str],
    source_manifest_extra: dict[str, Any] | None = None,
    schema_version: str = "1.0.0",
) -> bytes:
    """Serialize a v21 mokuhanga separation to a ZIP archive.

    The output is backwards-compatible with the v20 cnc tool:
    legacy ``png/<rank>_<hex>.png`` and ``svg/<rank>_<hex>.svg`` entries
    are present alongside the new v21 ``blocks/`` and ``impressions/`` entries.

    Args:
        composite_png_bytes: Reconstruction PNG bytes.
        plates: v20-style plate dicts. Each must carry ``rank`` and ``hex``
            (or ``color``) plus pre-rendered ``png_bytes`` and ``svg`` blobs.
        pigment_palette: Editor palette entries (``id``, ``name``, ``rgb``,
            ``hex``, ``alpha_prior``).
        block_assignment: Map from pigment id (string) to block id (int).
        impressions: Ordered list of impression records.
        impression_previews: Cumulative composite PNGs, one per impression.
        block_svgs: Map from block id to multi-region composite SVG string.
        source_manifest_extra: Optional extra fields merged into the top-level
            manifest (``editor`` key is reserved).
        schema_version: Editor extension schema version.

    Returns:
        Raw ZIP bytes ready to stream to a client.
    """
    width, height = _infer_dimensions(plates)

    manifest = _build_manifest(
        plates=plates,
        width=width,
        height=height,
        pigment_palette=pigment_palette,
        block_assignment=block_assignment,
        impressions=impressions,
        schema_version=schema_version,
        source_manifest_extra=source_manifest_extra,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            zipfile.ZipInfo("composite.png"),
            composite_png_bytes,
            compress_type=zipfile.ZIP_STORED,
        )

        # v20 legacy per-pigment artifacts (cnc tool reads via glob)
        for plate in plates:
            rank = plate.get("rank")
            hex_str = plate.get("hex") or plate.get("color_hex") or ""
            if rank is None or not hex_str:
                continue
            stem = f"{int(rank):02d}_{_hex_to_filename(str(hex_str))}"

            png_blob = plate.get("png_bytes")
            if isinstance(png_blob, (bytes, bytearray)):
                zf.writestr(
                    zipfile.ZipInfo(f"png/{stem}.png"),
                    bytes(png_blob),
                    compress_type=zipfile.ZIP_STORED,
                )

            svg_blob = plate.get("svg")
            if isinstance(svg_blob, str):
                zf.writestr(
                    zipfile.ZipInfo(f"svg/{stem}.svg"),
                    svg_blob.encode("utf-8"),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            elif isinstance(svg_blob, (bytes, bytearray)):
                zf.writestr(
                    zipfile.ZipInfo(f"svg/{stem}.svg"),
                    bytes(svg_blob),
                    compress_type=zipfile.ZIP_DEFLATED,
                )

        # v21 physical blocks (multi-region SVG per cuttable woodblock)
        for block_id, svg_str in sorted(block_svgs.items()):
            zf.writestr(
                zipfile.ZipInfo(f"blocks/block_{int(block_id):02d}.svg"),
                svg_str.encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )

        # v21 cumulative impression previews
        for idx, preview_bytes in enumerate(impression_previews):
            step_num = idx + 1
            zf.writestr(
                zipfile.ZipInfo(f"impressions/step_{step_num:03d}.png"),
                bytes(preview_bytes),
                compress_type=zipfile.ZIP_STORED,
            )

        # v21 print order index
        if impressions:
            zf.writestr(
                zipfile.ZipInfo("print_order.csv"),
                _build_print_order_csv(impressions),
                compress_type=zipfile.ZIP_DEFLATED,
            )

        zf.writestr(
            zipfile.ZipInfo("manifest.json"),
            json.dumps(manifest, indent=2).encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    return buf.getvalue()
