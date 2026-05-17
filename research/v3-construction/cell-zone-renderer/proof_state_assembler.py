"""Proof-state 8-up sheet assembler.

Picks checkpoint pulls from the full pulls list and lays them out in
an 8-up grid — exactly the layout in
`/srv/woodblock-share/Examples/Screenshot 2026-05-14 at 6.13.05 PM.png`.

The locked design's 7 checkpoints are after pulls 4, 8, 12, 16, 20,
24, 132. We pack them into a 4×2 grid so the 8th cell is the final
proof. (4×2 = 8 cells; we always emit 8 to match the Chuck Close
reference proof sheet.)

Each cell is a thumbnail of the proof state at that checkpoint, with
a thin border, a small caption (pull index), and an optional outer
margin to keep printing nice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CHECKPOINTS = [4, 8, 12, 16, 20, 24, 64, 132]


def _to_pil(img: NDArray[np.float32]) -> Image.Image:
    arr = np.clip(img, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def _try_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def assemble_proof_sheet(
    pulls: list[NDArray[np.float32]],
    *,
    checkpoint_indices: list[int] | None = None,
    cols: int = 4,
    rows: int = 2,
    cell_size: tuple[int, int] = (520, 720),
    bg_rgb: tuple[int, int, int] = (250, 246, 232),
    cell_border: int = 2,
    cell_margin: int = 24,
    label_height: int = 26,
) -> Image.Image:
    """Lay out checkpoint pulls into a cols×rows sheet.

    Parameters
    ----------
    pulls:
        List of length N — one cumulative proof-state image per pull,
        index ``i`` is the state after pull ``i+1``. Each array is
        H×W×3 float32 0..1.
    checkpoint_indices:
        1-based pull indices to emit. Defaults to the locked-design
        checkpoint list (truncated to fit cols*rows). Indices past
        ``len(pulls)`` are clamped to the final pull.
    cols, rows:
        Sheet grid. cols*rows must equal len(checkpoint_indices).
    cell_size:
        Pixel size of each thumbnail.
    bg_rgb, cell_border, cell_margin, label_height:
        Visual settings.
    """
    n_cells = cols * rows
    if checkpoint_indices is None:
        checkpoint_indices = DEFAULT_CHECKPOINTS[:n_cells]
    if len(checkpoint_indices) != n_cells:
        raise ValueError(
            f"checkpoint_indices length {len(checkpoint_indices)} != cols*rows={n_cells}"
        )
    if not pulls:
        raise ValueError("pulls must be non-empty")

    cw, ch = cell_size
    total_w = cols * (cw + cell_margin) + cell_margin
    total_h = rows * (ch + label_height + cell_margin) + cell_margin
    sheet = Image.new("RGB", (total_w, total_h), bg_rgb)
    draw = ImageDraw.Draw(sheet)
    font = _try_font(16)

    for i, ckpt in enumerate(checkpoint_indices):
        clamped_idx = max(0, min(len(pulls) - 1, ckpt - 1))
        thumb = _to_pil(pulls[clamped_idx]).resize(
            (cw, ch), Image.Resampling.LANCZOS
        )
        r, c = divmod(i, cols)
        x = cell_margin + c * (cw + cell_margin)
        y = cell_margin + r * (ch + label_height + cell_margin)
        sheet.paste(thumb, (x, y))
        if cell_border > 0:
            draw.rectangle(
                [x - cell_border, y - cell_border,
                 x + cw + cell_border - 1, y + ch + cell_border - 1],
                outline=(60, 50, 30), width=cell_border,
            )
        label = f"after pull {ckpt:>3d}"
        draw.text(
            (x + 4, y + ch + 4),
            label,
            fill=(60, 50, 30),
            font=font,
        )
    return sheet


def assemble_plate_and_pull_sheet(
    plate_previews: list[Image.Image],
    pull_previews: list[NDArray[np.float32]],
    *,
    cols: int = 4,
    cell_size: tuple[int, int] = (320, 440),
    bg_rgb: tuple[int, int, int] = (250, 246, 232),
    cell_margin: int = 16,
) -> Image.Image:
    """Each cell is plate-above / pull-below, like Woodblock-print-process.png.

    plate_previews[i] is paired with pull_previews[i]. Both lists
    must have the same length.
    """
    if len(plate_previews) != len(pull_previews):
        raise ValueError(
            f"plate/pull length mismatch: {len(plate_previews)} vs {len(pull_previews)}"
        )
    n = len(plate_previews)
    if n == 0:
        raise ValueError("need at least one pair")
    rows = (n + cols - 1) // cols
    cw, ch = cell_size
    pair_h = 2 * ch + cell_margin // 2  # plate stacked above pull
    label_h = 22

    total_w = cols * (cw + cell_margin) + cell_margin
    total_h = rows * (pair_h + label_h + cell_margin) + cell_margin
    sheet = Image.new("RGB", (total_w, total_h), bg_rgb)
    draw = ImageDraw.Draw(sheet)
    font = _try_font(14)

    for i in range(n):
        r, c = divmod(i, cols)
        x = cell_margin + c * (cw + cell_margin)
        y = cell_margin + r * (pair_h + label_h + cell_margin)
        plate_thumb = plate_previews[i].resize((cw, ch), Image.Resampling.LANCZOS)
        pull_thumb = _to_pil(pull_previews[i]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(plate_thumb, (x, y))
        sheet.paste(pull_thumb, (x, y + ch + cell_margin // 2))
        draw.text(
            (x + 2, y + pair_h + 2),
            f"step {i + 1:>3d}",
            fill=(60, 50, 30),
            font=font,
        )
    return sheet
