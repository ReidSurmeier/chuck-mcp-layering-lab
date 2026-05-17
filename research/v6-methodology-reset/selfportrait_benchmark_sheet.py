#!/usr/bin/env python3
"""Render a white-background self-portrait methodology benchmark sheet.

The sheet is deliberately visual-first:

1. Chuck Close/Pace reference progressive proofs.
2. Current Chuck MCP cumulative proofs, aligned to the same pull numbers.
3. Chuck Close/Pace reference block previews.
4. Current Chuck MCP generated block previews.

This is not an acceptance metric by itself. It is the artifact Reid can inspect
to decide whether the algorithm is developing a print like the reference method
or merely satisfying numerical plumbing validators.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_REFERENCE_DIR = Path(
    "/srv/woodblock-share/plotter-separation/close-self-portrait-2001"
)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

PAGE_BG = (250, 250, 247)
INK = (32, 32, 29)
MUTED = (99, 99, 91)
RULE = (216, 214, 205)
CARD_BG = (255, 255, 255)


@dataclass(frozen=True)
class ReferencePlate:
    index: int
    label: str
    progressive_path: Path
    preview_path: Path


def load_reference_plates(reference_dir: Path) -> list[ReferencePlate]:
    manifest_path = reference_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        plates = []
        for raw in manifest.get("plates", []):
            plates.append(
                ReferencePlate(
                    index=int(raw["index"]),
                    label=str(raw["label"]),
                    progressive_path=Path(raw["progressive_path"]),
                    preview_path=Path(raw["preview_path"]),
                )
            )
        return sorted(plates, key=lambda p: p.index)

    progressives = sorted((reference_dir / "progressives").glob("progressive_*.png"))
    previews = sorted((reference_dir / "previews").glob("plate_*.png"))
    return [
        ReferencePlate(
            index=i,
            label=f"plate {i:02d}",
            progressive_path=prog,
            preview_path=previews[i - 1] if i - 1 < len(previews) else prog,
        )
        for i, prog in enumerate(progressives, start=1)
    ]


def build_sheet(
    *,
    reference_dir: Path,
    job_dir: Path,
    output: Path,
    title: str = "Chuck MCP Self-Portrait Methodology Benchmark",
) -> Path:
    reference_plates = load_reference_plates(reference_dir)
    if not reference_plates:
        raise ValueError(f"no reference plates found in {reference_dir}")

    column_count = len(reference_plates)
    artifacts_dir = job_dir / "artifacts"
    current_proofs = _evenly_spaced(
        sorted((artifacts_dir / "pulls").glob("pull_*.png")),
        column_count,
    )
    if not current_proofs:
        current_proofs = _evenly_spaced(
            sorted(artifacts_dir.glob("cumulative_pull_*.png")),
            column_count,
        )
    current_blocks = _evenly_spaced(
        sorted((artifacts_dir / "plates").glob("block_*.preview.png")),
        column_count,
    )
    if not current_blocks:
        current_blocks = _evenly_spaced(
            sorted((artifacts_dir / "plates").glob("block_*.png")),
            column_count,
        )

    rows = [
        (
            "Reference proofs",
            "Pace / Close progressive proof sequence",
            [p.progressive_path for p in reference_plates],
            [f"{p.index:02d} {p.label}" for p in reference_plates],
        ),
        (
            "Model proofs",
            "Chuck MCP cumulative pulls from this run",
            current_proofs,
            [_numbered_caption(path, "pull") for path in current_proofs],
        ),
        (
            "Reference blocks",
            "Known block contribution previews",
            [p.preview_path for p in reference_plates],
            [f"{p.index:02d} {p.label}" for p in reference_plates],
        ),
        (
            "Model blocks",
            "Generated block previews in pull order",
            current_blocks,
            [_numbered_caption(path, "block") for path in current_blocks],
        ),
    ]

    tile_w = 148
    tile_h = 210
    label_w = 190
    gutter = 16
    side = 34
    top = 112
    row_gap = 54
    caption_h = 35
    row_h = tile_h + caption_h
    width = side * 2 + label_w + gutter + column_count * tile_w + (column_count - 1) * gutter
    height = top + len(rows) * row_h + (len(rows) - 1) * row_gap + 42

    sheet = Image.new("RGB", (width, height), PAGE_BG)
    draw = ImageDraw.Draw(sheet)
    font_title = _font(FONT_BOLD, 27)
    font_subtitle = _font(FONT_REGULAR, 14)
    font_row = _font(FONT_BOLD, 15)
    font_small = _font(FONT_REGULAR, 10)
    font_col = _font(FONT_BOLD, 11)

    run_label = job_dir.name
    draw.text((side, 28), title, fill=INK, font=font_title)
    draw.text(
        (side, 65),
        f"Reference: {reference_dir.name}    Run: {run_label}",
        fill=MUTED,
        font=font_subtitle,
    )
    draw.line((side, top - 24, width - side, top - 24), fill=RULE, width=1)

    x0 = side + label_w + gutter
    for idx in range(column_count):
        x = x0 + idx * (tile_w + gutter)
        draw.text((x + 4, top - 18), f"{idx + 1:02d}", fill=MUTED, font=font_col)

    y = top
    for row_title, row_note, paths, captions in rows:
        draw.text((side, y + 4), row_title, fill=INK, font=font_row)
        draw.text(
            (side, y + 26),
            _fit_text(row_note, font_small, label_w - 8),
            fill=MUTED,
            font=font_small,
        )
        draw.line((side, y - 12, width - side, y - 12), fill=RULE, width=1)

        for idx in range(column_count):
            x = x0 + idx * (tile_w + gutter)
            path = paths[idx] if idx < len(paths) else None
            _draw_tile(
                sheet,
                draw,
                path,
                box=(x, y, x + tile_w, y + tile_h),
                caption=captions[idx] if idx < len(captions) else "missing",
                caption_y=y + tile_h + 7,
                font=font_small,
            )
        y += row_h + row_gap

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)
    return output


def _draw_tile(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    path: Path | None,
    *,
    box: tuple[int, int, int, int],
    caption: str,
    caption_y: int,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=6, fill=CARD_BG, outline=RULE, width=1)
    pad = 7
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    if path is not None and path.exists():
        img = Image.open(path).convert("RGB")
        fitted = _fit_image(img, inner[2] - inner[0], inner[3] - inner[1])
        px = inner[0] + ((inner[2] - inner[0]) - fitted.width) // 2
        py = inner[1] + ((inner[3] - inner[1]) - fitted.height) // 2
        sheet.paste(fitted, (px, py))
    else:
        draw.rectangle(inner, fill=(244, 244, 240))
        draw.text((inner[0] + 10, inner[1] + 10), "missing", fill=MUTED, font=font)

    caption = _fit_text(caption, font, x1 - x0)
    tw = _text_width(caption, font)
    draw.text((x0 + ((x1 - x0) - tw) / 2, caption_y), caption, fill=MUTED, font=font)


def _fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return out


def _evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    idxs = np.linspace(0, len(paths) - 1, num=count).round().astype(int).tolist()
    out: list[Path] = []
    seen: set[int] = set()
    for idx in idxs:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(paths[idx])
    return out


def _font(path: Path, size: int) -> ImageFont.ImageFont:
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _fit_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if _text_width(text, font) <= max_width:
        return text
    ellipsis = "..."
    words = text.split()
    if len(words) <= 1:
        base = text
        while base and _text_width(base + ellipsis, font) > max_width:
            base = base[:-1]
        return base + ellipsis
    out = ""
    for word in words:
        candidate = f"{out} {word}".strip()
        if _text_width(candidate + ellipsis, font) > max_width:
            return out + ellipsis
        out = candidate
    return out


def _text_width(text: str, font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0])


def _numbered_caption(path: Path, prefix: str) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        return prefix
    return f"{prefix} {int(digits):02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Chuck MCP Self-Portrait Methodology Benchmark")
    args = parser.parse_args()

    output = build_sheet(
        reference_dir=args.reference_dir,
        job_dir=args.job_dir,
        output=args.output,
        title=args.title,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
