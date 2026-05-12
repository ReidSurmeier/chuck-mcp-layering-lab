"""D14.d — compare_plans + merge_impressions real wiring."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    from backend.mcp import paths
    importlib.reload(paths)
    from backend.services.v23 import session as _sess
    importlib.reload(_sess)
    from backend.services.v23 import orchestrator as _orch
    importlib.reload(_orch)


def _write_test_image(p: Path, h: int = 16, w: int = 16, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    img_path = p / f"tiny_{seed}.png"
    Image.fromarray(arr, "RGB").save(img_path)
    return img_path


@pytest.fixture
def two_plans(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    from backend.services.v23 import orchestrator as _orch
    p1 = _orch.run_pipeline_partial(str(_write_test_image(tmp_path, seed=1)), solve_profile="fast")
    p2 = _orch.run_pipeline_partial(str(_write_test_image(tmp_path, seed=2)), solve_profile="fast")
    return p1, p2


def test_compare_plans_returns_real_dE_delta(two_plans) -> None:
    p1, p2 = two_plans
    from backend.mcp.tools import hitl
    r = hitl.compare_plans(p1.plan_id, p2.plan_id)
    assert r.ok is True, r.errors
    assert "dE_delta_mean" in r.data
    assert "dE_delta_p95" in r.data
    # Real values (not the all-zero mock)
    assert r.data["dE_delta_mean"] is not None
    assert r.data["impression_count_a"] == len(p1.impressions)
    assert r.data["impression_count_b"] == len(p2.impressions)


def test_compare_plans_unknown_refuses(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl
    r = hitl.compare_plans("nope_a", "nope_b")
    assert r.ok is False


def test_merge_impressions_real_creates_new_plan(two_plans) -> None:
    p1, _ = two_plans
    if len(p1.impressions) < 2:
        pytest.skip("solver produced <2 impressions on this seed")
    ids = [imp["id"] for imp in p1.impressions[:2]]
    from backend.mcp.tools import hitl
    r = hitl.merge_impressions(p1.plan_id, ids)
    assert r.ok is True, r.errors
    assert r.data["new_plan_id"] != p1.plan_id
    assert r.data["merged_count"] == 2
    # New plan must load
    from backend.services.v23 import orchestrator as _orch
    merged = _orch.load_plan(r.data["new_plan_id"])
    assert len(merged.impressions) == len(p1.impressions) - 1
