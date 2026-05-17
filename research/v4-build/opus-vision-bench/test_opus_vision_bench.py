"""
test_opus_vision_bench.py — required tests for the bench harness.

The 7 required tests:
  1. test_ground_truth_generator_produces_10_overlays
  2. test_opus_extractor_returns_dict_with_19_regions
  3. test_jaccard_evaluator_perfect_match_returns_1
  4. test_jaccard_evaluator_disjoint_returns_0
  5. test_fallback_router_routes_opus_above_threshold
  6. test_fallback_router_routes_mediapipe_below_threshold
  7. test_end_to_end_benchmark_runs_without_error

The "Opus extractor returns 19 regions" test is checked structurally —
we monkeypatch the underlying translate_intent_prompt call so we don't
incur subscription cost during pytest. The end-to-end test runs with
dry_run=True, which short-circuits Opus and uses ground truth as the
prediction (so we exercise every line of run_benchmark.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fallback_router import RoutingDecision, route_to_opus_or_mediapipe
from generate_ground_truth import (
    GroundTruthEntry,
    generate_dataset,
    generate_overlay,
)
from jaccard_evaluator import BenchResult, ImageScore, aggregate, compare
from opus_cell_id_extractor import (
    ExtractionResult,
    OpusExtractionError,
    extract_cell_ids_from_overlay,
)
from run_benchmark import run_benchmark

# Region vocabulary
sys.path.insert(0, str(_HERE.parent.parent / "v3-construction"
                       / "mediapipe-face-spatial"))
from region_vocabulary import list_supported_regions  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures — generate ground truth ONCE per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gt_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gt")
    generate_dataset(out, force=False)
    return out


@pytest.fixture(scope="session")
def gt_entries(gt_dir) -> list[GroundTruthEntry]:
    return generate_dataset(gt_dir, force=False)


# ---------------------------------------------------------------------------
# 1. ground truth produces 10 overlays
# ---------------------------------------------------------------------------


def test_ground_truth_generator_produces_10_overlays(gt_entries):
    assert len(gt_entries) == 10
    for e in gt_entries:
        assert e.overlay_path.exists(), f"missing overlay: {e.overlay_path}"
        assert e.regions_path.exists(), f"missing regions: {e.regions_path}"
        payload = json.loads(e.regions_path.read_text())
        assert "regions" in payload
        # at least one region should have non-empty cell list (face is
        # almost-guaranteed; everything else may be missing on bald/no-bg
        # synthetic inputs).
        assert any(len(v) > 0 for v in payload["regions"].values()), \
            f"{e.image_id} produced empty regions"


# ---------------------------------------------------------------------------
# 2. Opus extractor returns dict with 19 regions
# ---------------------------------------------------------------------------


def test_opus_extractor_returns_dict_with_19_regions(tmp_path):
    """Structurally validate the extractor returns the canonical 19 regions.

    We mock translate_intent_prompt so this test does NOT call Opus.
    """
    region_names = list_supported_regions()
    assert len(region_names) == 19, "vocab drifted; expected 19 regions"

    # Need a real overlay PNG on disk (extractor reads + base64-encodes it).
    fake_overlay = tmp_path / "fake.png"
    # 1x1 PNG header
    import base64
    fake_overlay.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIA"
        "AAUAAYIWXgwAAAAASUVORK5CYII="
    ))

    fake_structured = {name: [0, 1, 2] for name in region_names}
    fake_envelope = {
        "total_cost_usd": 0.0,
        "duration_ms": 100,
        "session_id": "test",
        "is_error": False,
        "subtype": "success",
        "structured_output": fake_structured,
    }
    from claude_p import ClaudeResult
    fake_result = ClaudeResult(structured_output=fake_structured,
                               raw_envelope=fake_envelope)

    with patch("opus_cell_id_extractor.translate_intent_prompt",
               return_value=fake_result):
        out = extract_cell_ids_from_overlay(
            fake_overlay, region_names,
            image_id="mock", max_cell_id=10,
        )

    assert isinstance(out, ExtractionResult)
    assert isinstance(out.predictions, dict)
    assert set(out.predictions.keys()) == set(region_names)
    assert len(out.predictions) == 19
    for region, ids in out.predictions.items():
        assert ids == [0, 1, 2], f"{region}: clamping/dedup damaged output"


# ---------------------------------------------------------------------------
# 3+4. Jaccard arithmetic
# ---------------------------------------------------------------------------


def test_jaccard_evaluator_perfect_match_returns_1():
    pred = {"face": [1, 2, 3], "hair": [4, 5]}
    truth = {"face": [1, 2, 3], "hair": [4, 5]}
    score = compare(pred, truth, image_id="perfect")
    assert score.mean_jaccard() == 1.0
    assert score.mean_f1() == 1.0
    assert score.min_jaccard() == 1.0
    for r in score.region_scores:
        assert r.jaccard == 1.0
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0


def test_jaccard_evaluator_disjoint_returns_0():
    pred = {"face": [10, 11, 12], "hair": [20, 21]}
    truth = {"face": [1, 2, 3], "hair": [4, 5]}
    score = compare(pred, truth, image_id="disjoint")
    assert score.mean_jaccard() == 0.0
    assert score.mean_f1() == 0.0
    for r in score.region_scores:
        assert r.jaccard == 0.0
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0
        assert not r.is_trivial


# ---------------------------------------------------------------------------
# 5+6. Fallback router
# ---------------------------------------------------------------------------


def _bench_with_region_jaccards(per_region_means: dict[str, float]) -> BenchResult:
    """Build a BenchResult where every image has exactly the supplied
    per-region Jaccards (one image per call)."""
    image_scores: list[ImageScore] = []
    for image_idx in range(3):  # >=3 images so per_region_summary has data
        scores = []
        from jaccard_evaluator import RegionScore
        for region, j in per_region_means.items():
            scores.append(RegionScore(
                region=region, jaccard=j, precision=j, recall=j, f1=j,
                n_pred=10, n_truth=10, n_intersect=int(10 * j),
                is_trivial=False,
            ))
        image_scores.append(
            ImageScore(image_id=f"img{image_idx}", region_scores=scores)
        )
    return aggregate(image_scores)


def test_fallback_router_routes_opus_above_threshold():
    bench = _bench_with_region_jaccards({
        "face": 0.98, "hair": 0.96, "left_cheek": 0.97, "right_cheek": 0.95,
    })
    decision = route_to_opus_or_mediapipe(bench)
    assert decision.global_route == "opus"
    assert decision.is_go() is True
    # All regions also above the per-region floor (0.85).
    for region, route in decision.per_region_route.items():
        assert route == "opus", f"{region}: expected opus, got {route}"


def test_fallback_router_routes_mediapipe_below_threshold():
    bench = _bench_with_region_jaccards({
        "face": 0.55, "hair": 0.40, "left_cheek": 0.30, "right_cheek": 0.20,
    })
    decision = route_to_opus_or_mediapipe(bench)
    assert decision.global_route == "mediapipe"
    assert decision.is_go() is False
    # All per-region scores are below the 0.85 floor -> all mediapipe.
    for region, route in decision.per_region_route.items():
        assert route == "mediapipe", f"{region}: expected mediapipe, got {route}"


# ---------------------------------------------------------------------------
# 7. End-to-end (dry-run = no Opus calls; uses ground truth as prediction)
# ---------------------------------------------------------------------------


def test_end_to_end_benchmark_runs_without_error(tmp_path):
    bench, decision, payload = run_benchmark(
        out_dir=tmp_path,
        timeout_s=60,
        max_retries=0,
        force_ground_truth=False,
        dry_run=True,
    )
    # Dry-run uses ground truth as prediction, so Jaccard should be ~1.0
    # (trivially-empty regions are excluded from the mean, so any non-empty
    # region scores 1.0).
    assert payload["n_images_scored"] == 10
    assert payload["bench"]["overall_mean_jaccard"] >= 0.99, (
        f"dry-run should score ~1.0; got "
        f"{payload['bench']['overall_mean_jaccard']:.3f}"
    )
    assert decision.global_route == "opus"

    # Artifacts written
    assert (tmp_path / "bench_results.json").exists()
    assert (tmp_path / "bench_report.md").exists()

    report_text = (tmp_path / "bench_report.md").read_text()
    assert "Routing decision" in report_text
    assert "overall mean Jaccard" in report_text
