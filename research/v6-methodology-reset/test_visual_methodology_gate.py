from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visual_methodology_gate import DEFAULT_REFERENCE, run_gate


def _save_rgb(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)


def _make_reference_sheet(path: Path) -> None:
    sheet = Image.new("RGB", (800, 400), (245, 242, 232))
    for idx in range(8):
        tile = Image.new("RGB", (200, 200), (245, 242, 232))
        draw = ImageDraw.Draw(tile)
        # Increasing portrait-like mass with irregular interior linework.
        margin = max(12, 70 - idx * 7)
        draw.ellipse((margin, 15, 200 - margin, 185), outline=(150, 80, 60), width=4)
        for k in range(6 + idx * 2):
            x0 = margin + (k * 17) % max(1, 200 - 2 * margin)
            y0 = 25 + (k * 23) % 145
            draw.line((x0, y0, min(190, x0 + 35), min(180, y0 + 22)), fill=(180, 90, 45), width=3)
        sheet.paste(tile, ((idx % 4) * 200, (idx // 4) * 200))
    _save_rgb(path, sheet)


def _make_dot_artifacts(artifacts_dir: Path) -> None:
    pulls = artifacts_dir / "pulls"
    pulls.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 9):
        img = Image.new("RGB", (256, 256), (245, 242, 232))
        draw = ImageDraw.Draw(img)
        for k in range(idx * 5):
            x = 35 + (k % 8) * 24
            y = 45 + (k // 8) * 24
            draw.ellipse((x, y, x + 12, y + 12), fill=(230, 110, 95))
        _save_rgb(pulls / f"pull_{idx:03d}.png", img)


def test_dot_cell_progression_fails_methodology_gate(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    artifacts = tmp_path / "artifacts"
    _make_reference_sheet(reference)
    _make_dot_artifacts(artifacts)

    report = run_gate(artifacts, reference)

    assert not report.passes
    assert report.verdict == "FAIL"
    assert not report.gates["dot_cell_artifact_rejection"]["passes"]


def test_real_iter13_baseline_fails_when_available() -> None:
    artifacts = Path("/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/artifacts")
    if not artifacts.exists() or not DEFAULT_REFERENCE.exists():
        return

    report = run_gate(artifacts, DEFAULT_REFERENCE)

    assert not report.passes
    assert report.verdict == "FAIL"
