from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from selfportrait_benchmark_sheet import build_sheet


def _save(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 120), color).save(path)


def test_selfportrait_sheet_uses_white_background_and_four_rows(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    job = tmp_path / "job"
    plates = []
    for idx, label in enumerate(("yellow", "red", "black"), start=1):
        progressive = reference / "progressives" / f"progressive_{idx:02d}_{label}.png"
        preview = reference / "previews" / f"plate_{idx:02d}_{label}.png"
        _save(progressive, (230 - idx * 10, 210, 130))
        _save(preview, (160, 140, 100 + idx * 10))
        plates.append(
            {
                "index": idx,
                "label": label,
                "progressive_path": str(progressive),
                "preview_path": str(preview),
            }
        )
        _save(job / "artifacts" / "pulls" / f"pull_{idx:03d}.png", (210, 170, 120))
        _save(
            job / "artifacts" / "plates" / f"block_{idx:02d}.preview.png",
            (130, 115, 95),
        )
    (reference / "manifest.json").write_text(json.dumps({"plates": plates}))

    output = build_sheet(reference_dir=reference, job_dir=job, output=tmp_path / "sheet.png")

    assert output.exists()
    img = Image.open(output).convert("RGB")
    assert img.width > 500
    assert img.height > 900
    assert img.getpixel((5, 5)) == (250, 250, 247)
