"""All-blocks contact-sheet renderer — the artifact that replaces
``LATEST_CHUCK_MCP_ALL_BLOCKS_CONTACT_SHEET.png``.

The v13 sheet shows faded full-face residuals labelled `block_NN`.
That is the bug. This module renders the 27-plate grid using
``plate_renderer.render_plate_preview`` so every cell is an isolated
jigsaw region on wood-grain ground, mirrored, with the plate's own
pigment color. Plate-not-composite by construction.

Layout: 7 columns × 4 rows = 28 cells (27 plates + 1 blank), matching
the v13 sheet's aspect so an A/B comparison is intuitive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cz_types import Plate
from plate_renderer import (
    PlateRenderConfig,
    _DEFAULT_CFG,
    render_plate_preview,
)


def _try_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_all_blocks_contact_sheet(
    plates: list[Plate],
    *,
    cols: int = 7,
    rows: int | None = None,
    cell_size: tuple[int, int] = (320, 420),
    cell_margin: int = 14,
    label_height: int = 22,
    bg_rgb: tuple[int, int, int] = (250, 246, 232),
    src_size: tuple[int, int] = (1024, 1024),
    cfg: PlateRenderConfig | None = None,
    tmp_dir: str | Path | None = None,
) -> Image.Image:
    """Return the all-blocks contact sheet as a single PIL Image.

    Each cell is a mirrored plate preview. Cells past the plate count
    are blank. The output PIL Image can be saved with .save(path).
    """
    cfg = cfg or _DEFAULT_CFG
    n = len(plates)
    if rows is None:
        rows = (n + cols - 1) // cols
    n_cells = cols * rows

    cw, ch = cell_size
    total_w = cols * (cw + cell_margin) + cell_margin
    total_h = rows * (ch + label_height + cell_margin) + cell_margin
    sheet = Image.new("RGB", (total_w, total_h), bg_rgb)
    draw = ImageDraw.Draw(sheet)
    font = _try_font(13)

    # Render each plate to a tmp PNG and paste it. We render to PNG
    # rather than calling the underlying image function directly so
    # this module composes the documented public API.
    tmp_root = Path(tmp_dir) if tmp_dir is not None else Path("/tmp/cz_contact_sheet")
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Use a smaller cfg.preview_size_px so we don't render full-res
    # 2048 plates 27 times for the contact sheet.
    cs_cfg = PlateRenderConfig(
        preview_size_px=cell_size,
        margin_frac=cfg.margin_frac,
        mill_radius_px=cfg.mill_radius_px,
        kento_kagi_frac=cfg.kento_kagi_frac,
        kento_hikitsuke_frac=cfg.kento_hikitsuke_frac,
        wood_base=cfg.wood_base,
        wood_dark=cfg.wood_dark,
        pigment_alpha=cfg.pigment_alpha,
        pigment_noise_strength=cfg.pigment_noise_strength,
    )

    for i in range(n_cells):
        r, c = divmod(i, cols)
        x = cell_margin + c * (cw + cell_margin)
        y = cell_margin + r * (ch + label_height + cell_margin)
        if i < n:
            plate = plates[i]
            out = tmp_root / f"block_{plate.block_id:02d}.png"
            render_plate_preview(
                plate, out, src_size=src_size, mirror=True, cfg=cs_cfg
            )
            with Image.open(out) as img:
                sheet.paste(img, (x, y))
            label = f"block_{plate.block_id:02d} • {plate.role[:14]} • {plate.pigment_name[:18]}"
        else:
            # blank cell: still draw a faint placeholder rectangle so the
            # grid reads visually
            draw.rectangle(
                [x, y, x + cw, y + ch], outline=(220, 215, 200), width=1
            )
            label = ""

        if label:
            draw.text(
                (x + 4, y + ch + 4),
                label,
                fill=(60, 50, 30),
                font=font,
            )

    return sheet


def save_contact_sheet(
    plates: list[Plate], output_path: str | Path, **kwargs
) -> Path:
    """Convenience wrapper that saves the contact sheet to ``output_path``."""
    img = render_all_blocks_contact_sheet(plates, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", optimize=True)
    return output_path
