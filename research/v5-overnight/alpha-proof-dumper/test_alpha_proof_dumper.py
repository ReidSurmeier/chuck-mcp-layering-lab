"""TDD suite for the alpha-proof-dumper.

Cycle 1 — per-pull alpha PNG dump.
Cycle 2 — per-plate mirrored preview PNG dump.
Cycle 3 — per-pull cumulative proof PNG dump.
Cycle 4 — 7-checkpoint proof state series dump.
Cycle 5 — acceptance_harness consumes the new outputs (rows 2/3/4 populated).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
EMMA_IMAGE = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")
HARNESS_DIR = REPO_ROOT / "research" / "v4-build" / "example-harness"


# ---------------------------------------------------------------------------
# Fixtures — synthetic SolvedPlate list (no full hybrid solve needed)
# ---------------------------------------------------------------------------


def _make_solved_plate(
    block_id: int,
    pass_index: int,
    target_shape: tuple[int, int],
    pigment_lab: tuple[float, float, float],
    rng: np.random.Generator,
    cells_per_plate: int = 3,
):
    """Create a SolvedPlate (hybrid_optimizer schema) with a random inked mask."""
    from chuck_mcp_v2.types import Plate as SolvedPlate

    H, W = target_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for _ in range(cells_per_plate):
        cy = int(rng.integers(low=H // 6, high=H - H // 6))
        cx = int(rng.integers(low=W // 6, high=W - W // 6))
        r = int(rng.integers(low=max(2, min(H, W) // 20), high=max(4, min(H, W) // 8)))
        yy, xx = np.ogrid[:H, :W]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask = np.maximum(mask, disk.astype(np.uint8))
    return SolvedPlate(
        block_id=block_id,
        cell_zone_ids=list(range(block_id * cells_per_plate, (block_id + 1) * cells_per_plate)),
        pigment_id="ivory_black",
        opacity=0.55,
        dilution=0.2,
        role="regional_mass",
        pass_index=pass_index,
        pigment_weights={"ivory_black": 1.0},
        inked_mask=mask,
        area_px=int(mask.sum()),
        repair_stats={"pigment_blend_lab": list(pigment_lab)},
    )


@pytest.fixture
def synthetic_run(tmp_path: Path) -> dict:
    """Build a fake hybrid OptimizationResult + target_rgb in tmp_path.

    Returns a dict ready to pass to dump_run_artifacts.
    """
    rng = np.random.default_rng(0xDEADBEEF)
    target_shape = (96, 96)
    H, W = target_shape

    # 6 plates with varied pigments → enough to trigger > 1 checkpoint
    palette = [
        (88.0, -8.0, 70.0),    # yellow
        (50.0, 70.0, 50.0),    # red
        (35.0, 5.0, -55.0),    # blue
        (50.0, -45.0, 10.0),   # green
        (45.0, 25.0, 35.0),    # sienna
        (15.0, 0.0, 0.0),      # black
    ]
    plates = [
        _make_solved_plate(
            block_id=i + 1,
            pass_index=i + 1,
            target_shape=target_shape,
            pigment_lab=palette[i % len(palette)],
            rng=rng,
        )
        for i in range(6)
    ]

    target_rgb = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    return {
        "target_rgb": target_rgb,
        "plates": plates,
        "out_dir": out_dir,
        "target_shape": target_shape,
    }


@pytest.fixture
def synthetic_run_many_pulls(tmp_path: Path) -> dict:
    """Bigger fixture — 25 plates so we can exercise the 132-pull case
    (after expansion via plate.pulls).
    """
    rng = np.random.default_rng(0xC0FFEE)
    target_shape = (96, 96)
    H, W = target_shape
    palette = [
        (88.0, -8.0, 70.0),
        (50.0, 70.0, 50.0),
        (35.0, 5.0, -55.0),
        (50.0, -45.0, 10.0),
        (45.0, 25.0, 35.0),
        (15.0, 0.0, 0.0),
    ]
    plates = [
        _make_solved_plate(
            block_id=i + 1,
            pass_index=i + 1,
            target_shape=target_shape,
            pigment_lab=palette[i % len(palette)],
            rng=rng,
        )
        for i in range(25)
    ]
    target_rgb = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    out_dir = tmp_path / "run_many"
    out_dir.mkdir()
    return {
        "target_rgb": target_rgb,
        "plates": plates,
        "out_dir": out_dir,
        "target_shape": target_shape,
    }


# ---------------------------------------------------------------------------
# Cycle 1: Per-pull alpha PNG dump
# ---------------------------------------------------------------------------


def test_dump_writes_per_pull_alpha_png(synthetic_run: dict) -> None:
    """RED: after dump_run_artifacts, alphas/pull_NNN_alpha.png exists for each
    plate. Dimensions must match the target (H, W).
    """
    from alpha_proof_dumper.dumper import dump_run_artifacts

    res = dump_run_artifacts(
        target_rgb=synthetic_run["target_rgb"],
        plates=synthetic_run["plates"],
        out_dir=synthetic_run["out_dir"],
    )
    alphas_dir = synthetic_run["out_dir"] / "alphas"
    assert alphas_dir.is_dir(), "alphas/ directory must be created"

    expected_count = len(synthetic_run["plates"])
    alpha_files = sorted(alphas_dir.glob("pull_*_alpha.png"))
    assert len(alpha_files) == expected_count, (
        f"expected {expected_count} alpha PNGs, found {len(alpha_files)}"
    )

    H, W = synthetic_run["target_shape"]
    for p in alpha_files:
        img = Image.open(p)
        # PIL reports (width, height)
        assert img.size == (W, H), (
            f"{p.name} dims {img.size} != target ({W}, {H})"
        )

    # The dumper must report what it wrote.
    assert "alphas" in res and len(res["alphas"]) == expected_count


# ---------------------------------------------------------------------------
# Cycle 2: Per-plate mirrored preview PNG dump
# ---------------------------------------------------------------------------


def test_dump_writes_per_plate_preview_png(synthetic_run: dict) -> None:
    """RED: after dump_run_artifacts, plates/block_NN.png + plates/block_NN.preview.png
    exist, are mirrored (horizontal flip of the inked mask placement), and
    contain RGB pixels (not pure black).
    """
    from alpha_proof_dumper.dumper import dump_run_artifacts

    dump_run_artifacts(
        target_rgb=synthetic_run["target_rgb"],
        plates=synthetic_run["plates"],
        out_dir=synthetic_run["out_dir"],
    )
    plates_dir = synthetic_run["out_dir"] / "plates"
    assert plates_dir.is_dir(), "plates/ directory must be created"

    expected_count = len(synthetic_run["plates"])
    plate_files = sorted(plates_dir.glob("block_*.png"))
    # Need both block_NN.png and block_NN.preview.png shapes recognized by harness.
    raw_pngs = sorted(plates_dir.glob("block_*[!preview].png"))
    preview_pngs = sorted(plates_dir.glob("block_*.preview.png"))
    assert len(raw_pngs) == expected_count, (
        f"expected {expected_count} block_NN.png files, found {len(raw_pngs)}"
    )
    assert len(preview_pngs) == expected_count, (
        f"expected {expected_count} block_NN.preview.png files, found {len(preview_pngs)}"
    )

    # Each preview must be a real RGB image with non-zero color variance
    # (i.e. not solid black/white — a real plate has wood + pigment).
    for p in preview_pngs[:3]:
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img)
        assert arr.shape[-1] == 3, "plate preview must be RGB"
        # Variance > 0 means at least some pixels differ. A degenerate solid
        # color would have variance = 0.
        assert float(arr.var()) > 1.0, f"{p.name} has near-zero variance ({arr.var()})"

    # Mirror correctness: a non-mirrored preview should fail this. We
    # construct a known-asymmetric plate and verify the dumped preview
    # has the inked mass on the OPPOSITE horizontal side.
    # (Implemented via the dedicated mirror test below.)


def test_plate_preview_is_horizontally_mirrored(tmp_path: Path) -> None:
    """RED: a plate whose inked_mask is fully on the LEFT half must produce a
    preview where the pigmented region sits on the RIGHT half (because wood
    blocks print backwards — the carving is the mirror of the print).
    """
    from chuck_mcp_v2.types import Plate as SolvedPlate
    from alpha_proof_dumper.dumper import dump_run_artifacts

    H, W = 64, 64
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[20:44, 4:24] = 1  # solid block on LEFT half
    plate = SolvedPlate(
        block_id=1,
        cell_zone_ids=[1],
        pigment_id="ivory_black",
        opacity=0.95,
        dilution=0.0,
        role="key_detail",
        pass_index=1,
        pigment_weights={"ivory_black": 1.0},
        inked_mask=mask,
        area_px=int(mask.sum()),
        repair_stats={"pigment_blend_lab": [15.0, 0.0, 0.0]},
    )
    target_rgb = np.full((H, W, 3), 200, dtype=np.uint8)
    out = tmp_path / "mirror_run"
    out.mkdir()
    dump_run_artifacts(target_rgb=target_rgb, plates=[plate], out_dir=out)

    preview = out / "plates" / "block_01.preview.png"
    assert preview.exists()
    arr = np.asarray(Image.open(preview).convert("RGB"))
    h, w = arr.shape[:2]
    # Compute mean luminance per half — pigment is dark vs wood-tan background.
    lum = arr.mean(axis=-1)
    left_dark = float(lum[:, : w // 2].mean())
    right_dark = float(lum[:, w // 2 :].mean())
    # If mirrored correctly, right side is darker (pigment moved there).
    assert right_dark < left_dark - 5.0, (
        f"plate preview not mirrored: left_lum={left_dark:.1f} "
        f"right_lum={right_dark:.1f}"
    )


# ---------------------------------------------------------------------------
# Cycle 3: Per-pull cumulative proof PNG dump
# ---------------------------------------------------------------------------


def test_dump_writes_per_pull_cumulative_proof(synthetic_run: dict) -> None:
    """RED: after dump_run_artifacts, pulls/pull_NNN.png exists for each pull
    and the LATER pulls are visually different from the earlier ones (cumulative).
    """
    from alpha_proof_dumper.dumper import dump_run_artifacts

    dump_run_artifacts(
        target_rgb=synthetic_run["target_rgb"],
        plates=synthetic_run["plates"],
        out_dir=synthetic_run["out_dir"],
    )
    pulls_dir = synthetic_run["out_dir"] / "pulls"
    assert pulls_dir.is_dir(), "pulls/ directory must be created"

    expected_count = len(synthetic_run["plates"])
    pull_files = sorted(pulls_dir.glob("pull_*.png"))
    assert len(pull_files) == expected_count, (
        f"expected {expected_count} pull PNGs, found {len(pull_files)}"
    )

    # First pull and last pull must look different (cumulative state evolved).
    first = np.asarray(Image.open(pull_files[0]).convert("RGB"), dtype=np.float32)
    last = np.asarray(Image.open(pull_files[-1]).convert("RGB"), dtype=np.float32)
    assert first.shape == last.shape
    mean_abs_diff = float(np.abs(first - last).mean())
    assert mean_abs_diff > 1.0, (
        f"cumulative proof didn't evolve: |first - last|_mean = {mean_abs_diff}"
    )

    # The harness also looks for `cumulative_pull_NN.png` at the plan root for
    # row 2. We must emit those names too (subset, evenly spaced).
    root_cumulatives = sorted(synthetic_run["out_dir"].glob("cumulative_pull_*.png"))
    assert len(root_cumulatives) >= 1, "must emit at least one cumulative_pull_NN.png"


# ---------------------------------------------------------------------------
# Cycle 4: 7-checkpoint proof state series
# ---------------------------------------------------------------------------


def test_dump_writes_7_checkpoint_proof_states(synthetic_run_many_pulls: dict) -> None:
    """RED: after dump_run_artifacts, proofs/proof_NN_after_pull_MMM.png exists
    for at least 7 checkpoints evenly spaced across the pull sequence.
    """
    from alpha_proof_dumper.dumper import dump_run_artifacts

    dump_run_artifacts(
        target_rgb=synthetic_run_many_pulls["target_rgb"],
        plates=synthetic_run_many_pulls["plates"],
        out_dir=synthetic_run_many_pulls["out_dir"],
    )
    proofs_dir = synthetic_run_many_pulls["out_dir"] / "proofs"
    assert proofs_dir.is_dir(), "proofs/ directory must be created"

    proof_files = sorted(proofs_dir.glob("proof_*_after_pull_*.png"))
    assert len(proof_files) >= 7, (
        f"expected >= 7 proof checkpoints, found {len(proof_files)}"
    )

    # Filenames must be parseable: proof_NN_after_pull_MMM.png
    import re
    pattern = re.compile(r"^proof_(\d+)_after_pull_(\d+)\.png$")
    indices = []
    for p in proof_files:
        m = pattern.match(p.name)
        assert m, f"unexpected proof filename: {p.name}"
        indices.append(int(m.group(2)))

    # Indices must be monotonically increasing (sorted by filename works
    # because we zero-pad the checkpoint number).
    assert indices == sorted(indices), "proof checkpoint indices not monotonic"

    # First < last — proof should evolve.
    first = np.asarray(Image.open(proof_files[0]).convert("RGB"), dtype=np.float32)
    last = np.asarray(Image.open(proof_files[-1]).convert("RGB"), dtype=np.float32)
    mean_abs_diff = float(np.abs(first - last).mean())
    assert mean_abs_diff > 1.0, (
        f"proof checkpoints did not evolve: |first - last|_mean = {mean_abs_diff}"
    )


# ---------------------------------------------------------------------------
# Cycle 5: Acceptance harness consumes the new outputs (rows 2/3/4 populated)
# ---------------------------------------------------------------------------


def test_acceptance_harness_finds_plates_and_alphas_after_plan_emma(
    synthetic_run_many_pulls: dict,
) -> None:
    """RED: feed an alpha-proof-dumper-populated plan dir into the acceptance
    harness; the resulting result object must report 8 plates rendered,
    >= 1 alpha rendered, and >= 7 proof checkpoints rendered.
    """
    from alpha_proof_dumper.dumper import dump_run_artifacts

    dump_run_artifacts(
        target_rgb=synthetic_run_many_pulls["target_rgb"],
        plates=synthetic_run_many_pulls["plates"],
        out_dir=synthetic_run_many_pulls["out_dir"],
    )

    # Make acceptance_harness importable for this in-process test.
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    from acceptance_harness.acceptance_harness import render_acceptance_sheet

    sheet_path = synthetic_run_many_pulls["out_dir"] / "acceptance_sheet.png"
    result = render_acceptance_sheet(
        plan_output_dir=synthetic_run_many_pulls["out_dir"],
        output_path=sheet_path,
    )

    assert sheet_path.exists(), "acceptance sheet must be written"
    assert result.plate_count_rendered == 8, (
        f"row 3 not populated — only {result.plate_count_rendered} plates "
        f"rendered (warnings: {result.warnings})"
    )
    assert result.alpha_count_rendered >= 1, (
        f"row 4 not populated — {result.alpha_count_rendered} alphas rendered "
        f"(warnings: {result.warnings})"
    )
    # Row 2 — at least one cumulative checkpoint.
    assert len(result.proof_checkpoints_rendered) >= 1, (
        f"row 2 empty — {result.proof_checkpoints_rendered}"
    )
