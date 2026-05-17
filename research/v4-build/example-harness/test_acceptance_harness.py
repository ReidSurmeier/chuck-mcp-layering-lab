"""pytest suite for the chuck-mcp v4 acceptance harness.

8 required tests (audit Phase 1 deliverable):
  1. loads reference examples returns 8 proofs
  2. renders 4-row sheet correct dimensions
  3. handles missing proof checkpoints gracefully
  4. handles fewer than 27 plates
  5. normalizes alpha maps to visible range
  6. cli invocation smoke
  7. runs on synthetic plan (27 dummy PNGs)
  8. runs on v13 failing baseline (real Phase 0 baseline if accessible)

Plus a handful of focused helpers, kept in this file to keep the module
self-contained.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# When the test module sits inside the package directory we want to import via
# the package name. The conftest.py inserts the parent of this dir onto
# sys.path so `import acceptance_harness` works.
from acceptance_harness.acceptance_harness import (
    TILE_H,
    TILE_W,
    TILES_PER_ROW,
    _evenly_spaced_indices,
    _normalize_alpha_tile,
    render_acceptance_sheet,
)
from acceptance_harness.acceptance_result import AcceptanceSheetResult
from acceptance_harness.example_loader import (
    HOKUSAI_CELL_COUNT,
    REFERENCE_EXAMPLES_DIR,
    load_reference_proofs,
)


# ---------------------------------------------------------------------------
# Synthetic plan fixtures
# ---------------------------------------------------------------------------


def _solid_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def _gradient_png(path: Path, size: tuple[int, int], hue_seed: int) -> None:
    """Make a tile distinguishable from its neighbors (for proof_progression test)."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (hue_seed * 17) % 255
    arr[..., 1] = (hue_seed * 53) % 255
    arr[..., 2] = (hue_seed * 89) % 255
    # Add a per-row gradient so consecutive checkpoints differ in structure too.
    arr[..., 0] = np.clip(arr[..., 0].astype(int) + np.arange(h)[:, None], 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _noisy_alpha_png(path: Path, size: tuple[int, int], peak: int) -> None:
    """Generate a low-range grayscale png to exercise _normalize_alpha_tile."""
    w, h = size
    rng = np.random.default_rng(peak)
    arr = rng.integers(low=0, high=max(peak, 1), size=(h, w), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def synthetic_plan_dir(tmp_path: Path) -> Path:
    """Create a synthetic plan with 27 pulls + 27 plates + 27 alpha maps."""
    plan = tmp_path / "synthetic_plan"
    plan.mkdir()
    (plan / "plates").mkdir()
    (plan / "alpha_masks").mkdir()

    for i in range(1, 28):
        _gradient_png(plan / f"cumulative_pull_{i:02d}.png", (320, 320), hue_seed=i)
        _gradient_png(plan / "plates" / f"block_{i:02d}.preview.png", (320, 320), hue_seed=i * 7)
        _noisy_alpha_png(plan / "alpha_masks" / f"alpha_{i:02d}.png", (160, 160), peak=8)

    _gradient_png(plan / "final_composite.png", (512, 512), hue_seed=999)
    _gradient_png(plan / "target.png", (512, 512), hue_seed=998)
    return plan


@pytest.fixture
def sparse_plan_dir(tmp_path: Path) -> Path:
    """Plan with only 3 pulls, 0 plates, 0 alpha maps — exercises graceful fallback."""
    plan = tmp_path / "sparse_plan"
    plan.mkdir()
    for i in range(1, 4):
        _gradient_png(plan / f"cumulative_pull_{i:02d}.png", (200, 200), hue_seed=i)
    return plan


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loads_reference_examples_returns_8_proofs() -> None:
    tiles = load_reference_proofs(REFERENCE_EXAMPLES_DIR, target_count=8)
    assert len(tiles) == 8
    # Each tile must be a PIL image with sane dimensions; the Hokusai cells are
    # 192x290 so just assert > 0.
    for t in tiles:
        assert t.size[0] > 0 and t.size[1] > 0
    # The Hokusai cell-count contract must stay at 8.
    assert HOKUSAI_CELL_COUNT == 8


def test_renders_4_row_sheet_correct_dimensions(synthetic_plan_dir: Path) -> None:
    out = synthetic_plan_dir / "sheet.png"
    result = render_acceptance_sheet(
        plan_output_dir=synthetic_plan_dir,
        reference_examples_dir=REFERENCE_EXAMPLES_DIR,
        output_path=out,
    )
    assert isinstance(result, AcceptanceSheetResult)
    assert result.sheet_path == out
    assert out.exists()

    sheet = Image.open(out)
    # 4 rows × (TILE_H + label) + 3 gaps + 2*pad + header — check height is
    # in the expected ballpark (allow ±20 px for label font fluctuations).
    expected_inner_h = 4 * (TILE_H + 28) + 3 * 12  # ROW_LABEL_H=28, ROW_GAP=12
    expected_h = expected_inner_h + 2 * 24 + 56  # SHEET_PAD=24, HEADER_H=56
    assert abs(sheet.height - expected_h) <= 8

    # Width = 8 tiles + 7 inter-tile gaps + 2*pad
    expected_w = TILES_PER_ROW * TILE_W + (TILES_PER_ROW - 1) * 4 + 2 * 24
    assert sheet.width == expected_w

    # Plate metrics: 8 rendered = 8 metrics.
    assert result.plate_count_rendered == 8
    assert len(result.plate_metrics) == 8

    # Proof progression score must be > 0 because gradient PNGs differ.
    assert result.proof_progression_score > 0.0


def test_handles_missing_proof_checkpoints_gracefully(sparse_plan_dir: Path) -> None:
    out = sparse_plan_dir / "sheet.png"
    result = render_acceptance_sheet(
        plan_output_dir=sparse_plan_dir,
        reference_examples_dir=REFERENCE_EXAMPLES_DIR,
        output_path=out,
    )
    assert out.exists()
    # Only 3 pulls available → at most 3 checkpoints rendered.
    assert len(result.proof_checkpoints_rendered) <= 3
    # Sheet still has full geometry (placeholders fill the rest).
    sheet = Image.open(out)
    assert sheet.height > 0 and sheet.width > 0
    # Should warn about missing plates and alphas.
    msgs = " ".join(result.warnings)
    assert "block_NN" in msgs or "plate_NN" in msgs
    assert "alpha" in msgs.lower()


def test_handles_fewer_than_27_plates(tmp_path: Path) -> None:
    plan = tmp_path / "small_plan"
    plan.mkdir()
    (plan / "plates").mkdir()
    for i in range(1, 6):  # only 5 plates
        _gradient_png(plan / f"cumulative_pull_{i:02d}.png", (200, 200), hue_seed=i)
        _gradient_png(plan / "plates" / f"block_{i:02d}.preview.png", (200, 200), hue_seed=i)
    _gradient_png(plan / "final_composite.png", (200, 200), hue_seed=999)

    out = plan / "sheet.png"
    result = render_acceptance_sheet(plan, output_path=out)
    assert result.plate_count_rendered == 5
    assert len(result.plate_metrics) == 5
    # Each metric must report a real coverage_fraction in [0, 1].
    for pm in result.plate_metrics:
        assert 0.0 <= pm.coverage_fraction <= 1.0
        assert 0.0 <= pm.plate_not_composite_score <= 1.0


def test_normalizes_alpha_maps_to_visible_range(tmp_path: Path) -> None:
    # Make a near-black alpha (max ~5) — without normalization it's invisible.
    src = tmp_path / "alpha_dark.png"
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:20, 10:20] = 4
    arr[30:40, 30:40] = 5
    Image.fromarray(arr, mode="L").save(src)

    tile = _normalize_alpha_tile(src, TILE_W, TILE_H)
    assert tile.size == (TILE_W, TILE_H)
    np_tile = np.asarray(tile)
    # After normalization, max pixel must be 255 (or near it) within the inset
    # region — i.e. the contrast has been STRETCHED.
    assert np_tile.max() >= 240, (
        f"alpha map not normalized — max pixel = {np_tile.max()} (expected ~255)"
    )

    # Flat-zero alpha → should produce a flat mid-gray (no NaN explosion).
    flat = tmp_path / "alpha_flat.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(flat)
    flat_tile = _normalize_alpha_tile(flat, TILE_W, TILE_H)
    assert flat_tile.size == (TILE_W, TILE_H)
    # No black holes, no white blowout — should be the mid-gray placeholder.
    fa = np.asarray(flat_tile)
    # Center pixel sits in the placeholder paste region.
    cy, cx = fa.shape[0] // 2, fa.shape[1] // 2
    assert 100 <= int(fa[cy, cx, 0]) <= 160


def test_cli_invocation_smoke(synthetic_plan_dir: Path) -> None:
    out = synthetic_plan_dir / "cli_sheet.png"
    # Invoke the package as a module — exercises __main__.py too.
    pkg_parent = str(Path(__file__).resolve().parent)
    env_overlay = f"import sys; sys.path.insert(0, {pkg_parent!r}); "
    cmd = [
        sys.executable,
        "-c",
        env_overlay
        + "from acceptance_harness.cli import main; "
        + "raise SystemExit(main())",
        str(synthetic_plan_dir),
        "--output",
        str(out),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    payload = json.loads(proc.stdout)
    assert "sheet_path" in payload
    assert payload["human_eyeball_required"] is True
    assert out.exists()


def test_runs_on_synthetic_plan(synthetic_plan_dir: Path) -> None:
    # End-to-end: full plan with 27 PNGs, evenly-spaced selection picks 8.
    result = render_acceptance_sheet(synthetic_plan_dir)
    assert len(result.proof_checkpoints_rendered) == TILES_PER_ROW
    # 27 cumulative_pull files indexed 1..27 — evenly-spaced picks should hit
    # both endpoints (1 and 27 in 1-based, i.e. our indices include 1 and 27).
    assert min(result.proof_checkpoints_rendered) == 1
    assert max(result.proof_checkpoints_rendered) == 27


V13_BASELINE = Path(
    "/srv/woodblock-share/chuck-clean-outputs/2026-05-17_v3-audit-thorough-main"
)


@pytest.mark.skipif(
    not V13_BASELINE.exists(),
    reason="v13 failing baseline not present on this host",
)
def test_runs_on_v13_failing_baseline(tmp_path: Path) -> None:
    out = tmp_path / "v13_acceptance_sheet.png"
    result = render_acceptance_sheet(
        plan_output_dir=V13_BASELINE,
        reference_examples_dir=REFERENCE_EXAMPLES_DIR,
        output_path=out,
    )
    assert out.exists()
    # Baseline has 12 cumulative pulls → 8 checkpoints picked from 12.
    assert len(result.proof_checkpoints_rendered) == TILES_PER_ROW
    # No plates/ subdir in this baseline → plates row goes to placeholders, but
    # the sheet must still render and metrics list must be empty or all-zero.
    assert isinstance(result.plate_count_rendered, int)
    # Sheet renders in under 10 seconds per the spec's performance budget.
    elapsed = next(
        (float(w.split("=")[1]) for w in result.warnings if w.startswith("render_seconds=")),
        None,
    )
    assert elapsed is not None, "render_seconds warning missing"
    assert elapsed < 10.0, f"render took {elapsed}s — exceeds 10s budget"


# ---------------------------------------------------------------------------
# Focused unit tests for helpers (lift coverage above 80%)
# ---------------------------------------------------------------------------


def test_evenly_spaced_indices_small_n() -> None:
    assert _evenly_spaced_indices(0, 8) == []
    assert _evenly_spaced_indices(3, 8) == [0, 1, 2]
    assert _evenly_spaced_indices(8, 8) == [0, 1, 2, 3, 4, 5, 6, 7]
    picked = _evenly_spaced_indices(27, 8)
    assert picked[0] == 0
    assert picked[-1] == 26
    # Must be strictly increasing.
    assert all(b > a for a, b in zip(picked[:-1], picked[1:], strict=False))


def test_render_raises_for_missing_plan_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_acceptance_sheet(tmp_path / "does_not_exist")


def test_result_is_serializable(synthetic_plan_dir: Path) -> None:
    result = render_acceptance_sheet(synthetic_plan_dir)
    d = result.to_dict()
    payload = json.dumps(d, default=str)
    parsed = json.loads(payload)
    assert parsed["human_eyeball_required"] is True
    assert "plate_metrics" in parsed


def test_cli_in_process(synthetic_plan_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise cli.main() directly (in-process) for coverage + human-text path."""
    from acceptance_harness.cli import main

    out = synthetic_plan_dir / "in_proc_sheet.png"
    rc = main([str(synthetic_plan_dir), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr()
    assert "acceptance sheet:" in captured.out
    assert "HUMAN EYEBALL REQUIRED" in captured.out

    # JSON mode.
    rc = main([str(synthetic_plan_dir), "--output", str(out), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plate_count_rendered"] == 8


def test_cli_reports_missing_plan_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from acceptance_harness.cli import main

    rc = main([str(tmp_path / "no_such_dir")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ERROR" in err


def test_performance_under_10s(synthetic_plan_dir: Path) -> None:
    """Performance budget: full render < 10s per audit spec."""
    import time as _t

    t0 = _t.perf_counter()
    result = render_acceptance_sheet(synthetic_plan_dir)
    elapsed = _t.perf_counter() - t0
    assert elapsed < 10.0, f"render took {elapsed:.2f}s — exceeds 10s budget"
    # The result itself also reports timing in warnings.
    timing_warnings = [w for w in result.warnings if w.startswith("render_seconds=")]
    assert len(timing_warnings) == 1
