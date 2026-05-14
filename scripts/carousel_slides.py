"""Visual review carousel builder for full-image validation runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from backend.services.v23.core import forward_render_jax


def _as_u8(rgb: NDArray[np.float32]) -> NDArray[np.uint8]:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _fit_image(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, (245, 241, 230))
    x = (box[0] - img.width) // 2
    y = (box[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _slide(
    title: str,
    image: Image.Image,
    out_path: Path,
    *,
    subtitle: str = "",
    notes: list[str] | None = None,
) -> None:
    w, h = 1600, 1040
    margin = 42
    header_h = 118
    note_w = 420
    canvas = Image.new("RGB", (w, h), (245, 241, 230))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=34)
    body_font = ImageFont.load_default(size=22)
    small_font = ImageFont.load_default(size=18)
    draw.text((margin, 30), title, fill=(20, 20, 20), font=title_font)
    if subtitle:
        draw.text((margin, 74), subtitle, fill=(55, 55, 55), font=body_font)

    image_box = (w - note_w - margin * 3, h - header_h - margin)
    fitted = _fit_image(image, image_box)
    canvas.paste(fitted, (margin, header_h))

    x = margin * 2 + image_box[0]
    y = header_h + 8
    draw.text((x, y), "Review Notes", fill=(20, 20, 20), font=body_font)
    y += 42
    for note in notes or []:
        wrapped = _wrap(note, 39)
        for line in wrapped:
            draw.text((x, y), line, fill=(45, 45, 45), font=small_font)
            y += 25
        y += 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if len(trial) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _plate_preview(alpha: NDArray[np.float32], pigment_id: int) -> NDArray[np.float32]:
    paper = forward_render_jax.PAPER_RGB.astype(np.float32)
    pigment = forward_render_jax.PIGMENT_TABLE[int(pigment_id)].astype(np.float32)
    a = np.clip(alpha[..., None], 0.0, 1.0)
    return paper[None, None, :] * (1.0 - a) + pigment[None, None, :] * a


def _apply_plate(
    composite: NDArray[np.float32],
    mask: NDArray[np.bool_],
    pigment_id: int,
    alpha: float,
) -> NDArray[np.float32]:
    pigment = forward_render_jax.PIGMENT_TABLE[int(pigment_id)].astype(np.float32)
    a = np.zeros(mask.shape + (1,), dtype=np.float32)
    a[mask] = float(np.clip(alpha, 0.0, 1.0))
    return composite * (1.0 - a) + pigment[None, None, :] * a


def _cell_mask(labels: NDArray[np.int32], cell_ids: list[int]) -> NDArray[np.bool_]:
    if not cell_ids:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, np.asarray(cell_ids, dtype=np.int32))


def _production_plate_preview(
    labels: NDArray[np.int32],
    plate: dict[str, Any],
) -> tuple[NDArray[np.float32], int, float, NDArray[np.bool_]]:
    suggested = (plate.get("suggested_pigments") or [{}])[0]
    pigment_id = int(suggested.get("pigment_id", 0))
    alpha = float(plate.get("suggested_alpha", 0.4))
    mask = _cell_mask(labels, list(plate.get("cell_ids", [])))
    paper = np.broadcast_to(
        forward_render_jax.PAPER_RGB.astype(np.float32),
        labels.shape + (3,),
    ).copy()
    preview = _apply_plate(paper, mask, pigment_id, alpha)
    return preview.astype(np.float32), pigment_id, alpha, mask


def _contact_sheet(slides: list[Path], out_path: Path) -> None:
    if not slides:
        return
    cols = 4
    thumb = (360, 234)
    margin = 18
    label_h = 26
    rows = int(np.ceil(len(slides) / cols))
    sheet = Image.new(
        "RGB",
        (cols * thumb[0] + (cols + 1) * margin,
         rows * (thumb[1] + label_h) + (rows + 1) * margin),
        (245, 241, 230),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, path in enumerate(slides):
        col = idx % cols
        row = idx // cols
        x = margin + col * (thumb[0] + margin)
        y = margin + row * (thumb[1] + label_h + margin)
        draw.text((x, y), path.stem[:52], fill=(20, 20, 20), font=font)
        img = Image.open(path).convert("RGB").resize(thumb, Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _batch_notes(batch_plan: dict[str, Any] | None) -> list[str]:
    if not batch_plan:
        return ["No production batch plan was available for this run."]
    diag = batch_plan.get("diagnostics", {})
    notes = [
        f"Template: {diag.get('template', 'unknown')}",
        f"Plate proposal: {diag.get('plate_count', 0)} plates across {diag.get('batch_count', 0)} batches.",
        "Batch 1 should read as light pink, blue, orange, and green support roles.",
        "Batch 2 should add stronger color/depth before the regional detail batch.",
    ]
    return notes


def _add_production_sequence(
    *,
    out_dir: Path,
    slide_paths: list[Path],
    batch_plan: dict[str, Any] | None,
) -> None:
    if not batch_plan:
        return
    labels_path = batch_plan.get("cell_labels_path")
    if not labels_path or not Path(labels_path).is_file():
        return
    labels = np.load(labels_path).astype(np.int32)
    composite = np.broadcast_to(
        forward_render_jax.PAPER_RGB.astype(np.float32),
        labels.shape + (3,),
    ).copy()
    pull_no = 0
    for batch in batch_plan.get("batches", []):
        for plate in batch.get("plates", []):
            pull_no += 1
            preview, pigment_id, alpha, mask = _production_plate_preview(labels, plate)
            name = forward_render_jax.PIGMENT_NAMES[int(pigment_id)]
            role = str(plate.get("role", f"plate_{pull_no:02d}"))
            slide = out_dir / f"slide_prod_{pull_no:03d}_plate_{role}_{name}.png"
            _slide(
                f"Production Pull {pull_no:02d} Plate",
                Image.fromarray(_as_u8(preview), "RGB"),
                slide,
                subtitle=f"{batch.get('batch_id')} / {role} / {name}",
                notes=[
                    f"Suggested alpha: {alpha:.2f}; coverage: {float(mask.mean() * 100.0):.2f} percent.",
                    "This is a cell-group production proposal, not the flat solver mask.",
                ],
            )
            slide_paths.append(slide)

            composite = _apply_plate(composite, mask, pigment_id, alpha)
            slide = out_dir / f"slide_prod_{pull_no:03d}_after_{role}_{name}.png"
            _slide(
                f"Production Composite After Pull {pull_no:02d}",
                Image.fromarray(_as_u8(composite), "RGB"),
                slide,
                subtitle=f"{batch.get('name')} / {role}",
                notes=[
                    "This shows the proposed production print developing by batch.",
                    "Use this to judge the Pace-style 4 + 4 + detail logic.",
                ],
            )
            slide_paths.append(slide)


def build_carousel_slides(
    *,
    run_name: str,
    carousel_root: Path,
    target_rgb: NDArray[np.float32],
    final_rgb: NDArray[np.float32],
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    cumulative_frames: list[NDArray[np.float32]],
    batch_plan: dict[str, Any] | None = None,
) -> Path:
    """Write numbered review slides and return the carousel directory."""
    out_dir = carousel_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_paths: list[Path] = []

    target_img = Image.fromarray(_as_u8(target_rgb), "RGB")
    final_img = Image.fromarray(_as_u8(final_rgb), "RGB")
    slide = out_dir / "slide_000_target.png"
    _slide(
        "Target Input",
        target_img,
        slide,
        subtitle=run_name,
        notes=["Use this as the visual reference for color density, hue separation, and facial structure."],
    )
    slide_paths.append(slide)

    batch_img = _batch_summary_image(batch_plan)
    slide = out_dir / "slide_001_batch_plan.png"
    _slide("Production Batch Plan", batch_img, slide, notes=_batch_notes(batch_plan))
    slide_paths.append(slide)
    _add_production_sequence(out_dir=out_dir, slide_paths=slide_paths, batch_plan=batch_plan)

    for i, (alpha, pid, frame) in enumerate(
        zip(alpha_stack, pigment_idx, cumulative_frames, strict=True),
        start=1,
    ):
        name = forward_render_jax.PIGMENT_NAMES[int(pid)]
        plate = Image.fromarray(_as_u8(_plate_preview(alpha, int(pid))), "RGB")
        slide = out_dir / f"slide_{i * 2:03d}_pull_{i:02d}_plate_{name}.png"
        _slide(
            f"Pull {i:02d} Plate",
            plate,
            slide,
            subtitle=name,
            notes=[
                f"Coverage above 0.08 alpha: {float((alpha >= 0.08).mean() * 100.0):.2f} percent.",
                "This is the plate color preview, not the cumulative print.",
            ],
        )
        slide_paths.append(slide)

        cumulative = Image.fromarray(_as_u8(frame), "RGB")
        slide = out_dir / f"slide_{i * 2 + 1:03d}_after_pull_{i:02d}_{name}.png"
        _slide(
            f"After Pull {i:02d}",
            cumulative,
            slide,
            subtitle=name,
            notes=[
                "This shows the print developing after the current pull.",
                "Review whether this stage reads like a plausible Pace-style batch progression.",
            ],
        )
        slide_paths.append(slide)

    slide = out_dir / f"slide_{len(slide_paths):03d}_final_composite.png"
    _slide(
        "Final Composite",
        final_img,
        slide,
        notes=["Compare density, chroma, and local hue separation against the target input."],
    )
    slide_paths.append(slide)

    _contact_sheet(slide_paths, out_dir / "carousel_contact_sheet.png")
    manifest = {
        "run_name": run_name,
        "slide_count": len(slide_paths),
        "slides": [str(path) for path in slide_paths],
        "contact_sheet": str(out_dir / "carousel_contact_sheet.png"),
    }
    (out_dir / "carousel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def _batch_summary_image(batch_plan: dict[str, Any] | None) -> Image.Image:
    img = Image.new("RGB", (1100, 820), (245, 241, 230))
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=20)
    small_font = ImageFont.load_default(size=17)
    draw.text((36, 28), "4 + 4 + Detail Batch Expansion", fill=(20, 20, 20), font=title_font)
    if not batch_plan:
        draw.text((36, 86), "No batch plan available.", fill=(45, 45, 45), font=body_font)
        return img
    y = 86
    for batch in batch_plan.get("batches", []):
        draw.text((36, y), str(batch.get("name", batch.get("batch_id"))), fill=(20, 20, 20), font=body_font)
        y += 32
        for plate in batch.get("plates", [])[:8]:
            text = (
                f"{plate.get('role')}: {plate.get('area_pct')}% area, "
                f"{plate.get('cell_count')} cells, {plate.get('mean_hex')}"
            )
            draw.text((58, y), text[:96], fill=(50, 50, 50), font=small_font)
            y += 24
        y += 16
        if y > 760:
            break
    return img


__all__ = ["build_carousel_slides"]
