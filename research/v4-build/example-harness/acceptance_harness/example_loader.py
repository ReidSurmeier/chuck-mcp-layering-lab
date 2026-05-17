"""Load reference example sub-images from /srv/woodblock-share/Examples.

The reference set is heterogeneous:

- Woodblock-print-process.png (770x580 RGBA, webp-encoded): the canonical
  Hokusai "Great Wave" 8-cell progression we anchor row 1 against. Cells are
  laid out as a 4-column x 2-row grid. We crop deterministically.
- Screenshot 2026-05-10 at 5.49.23 PM.png (3358x2144): a Chuck Close-style
  proof grid screenshot. Used as a tertiary visual reference.
- Screenshot 2026-05-14 at 6.13.05 PM.png (2472x1570): a methodology proof
  sheet screenshot.
- Untitled-1.png (2078x1290) and 12341243.png (4239x2105): supplementary
  reference photos.

Only the Hokusai 8-cell file is structurally split. The others are loaded as
single thumbnails on demand.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REFERENCE_EXAMPLES_DIR = Path("/srv/woodblock-share/Examples")
# Chuck Close self-portrait 2015 progressive proof from Pace Editions —
# 8 cumulative proof states laid out 4 cols x 2 rows. Anchors row 1 of the
# acceptance sheet. Closer reference for chuck-mcp's Chuck-Close-portrait
# scope than Hokusai's Great Wave. (Per user feedback 2026-05-17.)
CHUCK_CLOSE_PROGRESSIVE_FILENAME = "Screenshot 2026-05-14 at 6.13.05 PM.png"
HOKUSAI_FILENAME = CHUCK_CLOSE_PROGRESSIVE_FILENAME  # alias kept for back-compat
HOKUSAI_GRID_COLS = 4
HOKUSAI_GRID_ROWS = 2
HOKUSAI_CELL_COUNT = HOKUSAI_GRID_COLS * HOKUSAI_GRID_ROWS  # 8

# Tile slot defaults used by all rows when source material is missing. Single
# centralized place so the contact sheet stays visually consistent.
DEFAULT_TILE_W = 256
DEFAULT_TILE_H = 256


def _normalized_filename(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).split())


def _resolve_example_file(examples_dir: Path, filename: str) -> Path:
    exact = examples_dir / filename
    if exact.exists():
        return exact
    target = _normalized_filename(filename)
    if examples_dir.exists():
        for path in examples_dir.iterdir():
            if path.is_file() and _normalized_filename(path.name) == target:
                return path
    return exact


def _make_placeholder_tile(
    width: int, height: int, label: str, bg: tuple[int, int, int] = (40, 40, 40)
) -> Image.Image:
    """Render a labeled placeholder tile.

    Used when a slot is missing (e.g. plan has only 20 pulls but row asks for
    8 evenly-spaced checkpoints; or reference examples folder is empty in a
    test harness).
    """
    tile = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover — load_default never raises in tested PIL
        font = None
    text_color = (200, 200, 200)
    if font is not None:
        # PIL's textbbox is the modern path (textsize was removed in PIL 10).
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            ((width - tw) // 2, (height - th) // 2),
            label,
            fill=text_color,
            font=font,
        )
    # Subtle border so missing tiles read as "intentional placeholder".
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(90, 90, 90))
    return tile


def load_woodblock_print_process(
    examples_dir: Path = REFERENCE_EXAMPLES_DIR,
) -> list[Image.Image]:
    """Split the Hokusai 8-cell PNG into 8 individual proof tiles.

    Returns RGB tiles in reading order (top-left → bottom-right). If the file
    is missing, returns 8 placeholder tiles so the harness can still produce a
    readable contact sheet for tests.
    """
    src = _resolve_example_file(examples_dir, HOKUSAI_FILENAME)
    if not src.exists():
        return [
            _make_placeholder_tile(
                DEFAULT_TILE_W,
                DEFAULT_TILE_H,
                f"ref proof {i + 1}\n(missing)",
                bg=(30, 30, 50),
            )
            for i in range(HOKUSAI_CELL_COUNT)
        ]

    img = Image.open(src).convert("RGB")
    w, h = img.size
    cell_w = w // HOKUSAI_GRID_COLS
    cell_h = h // HOKUSAI_GRID_ROWS
    tiles: list[Image.Image] = []
    for r in range(HOKUSAI_GRID_ROWS):
        for c in range(HOKUSAI_GRID_COLS):
            left = c * cell_w
            upper = r * cell_h
            right = left + cell_w
            lower = upper + cell_h
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles


def load_reference_proofs(
    examples_dir: Path = REFERENCE_EXAMPLES_DIR,
    target_count: int = 8,
) -> list[Image.Image]:
    """Public entry point: return `target_count` reference proof tiles.

    Primary source is the Hokusai 8-cell sheet. If `target_count` > 8, the
    extras are filled with placeholder tiles (caller's responsibility to ask
    for sane counts; design doc fixes 8 across the board).
    """
    tiles = load_woodblock_print_process(examples_dir)
    if len(tiles) >= target_count:
        return tiles[:target_count]
    short = target_count - len(tiles)
    pad = [
        _make_placeholder_tile(
            DEFAULT_TILE_W,
            DEFAULT_TILE_H,
            f"ref pad {i + 1}",
            bg=(30, 30, 50),
        )
        for i in range(short)
    ]
    return tiles + pad


def list_available_examples(examples_dir: Path = REFERENCE_EXAMPLES_DIR) -> list[str]:
    """List filenames present in the examples directory (for provenance)."""
    if not examples_dir.exists():
        return []
    return sorted(p.name for p in examples_dir.iterdir() if p.is_file())
