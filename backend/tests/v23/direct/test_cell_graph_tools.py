"""Cell graph MCP tool surface tests."""
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


def _write_test_image(p: Path) -> Path:
    arr = np.full((24, 24, 3), [246, 241, 227], dtype=np.uint8)
    arr[:12, :12] = [238, 190, 125]
    arr[:12, 12:] = [92, 148, 190]
    arr[12:, :12] = [225, 68, 58]
    arr[12:, 12:] = [34, 48, 62]
    img_path = p / "cell_tool.png"
    Image.fromarray(arr, "RGB").save(img_path)
    return img_path


@pytest.fixture
def real_plan(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    from backend.services.v23 import orchestrator as _orch
    return _orch.run_pipeline_partial(str(_write_test_image(tmp_path)), solve_profile="fast")


def test_cell_at_returns_persisted_cell(real_plan) -> None:
    from backend.mcp.tools import introspection

    r = introspection.cell_at(real_plan.plan_id, 4, 4)
    assert r.ok is True, r.errors
    assert r.data["cell_id"] >= 0
    assert r.data["cell"]["area_px"] > 0
    assert r.data["graph_summary"]["cell_count"] > 0


def test_inspect_cell_returns_active_impressions(real_plan) -> None:
    from backend.mcp.tools import introspection

    cell = introspection.cell_at(real_plan.plan_id, 4, 4).data["cell_id"]
    r = introspection.inspect_cell(real_plan.plan_id, cell)
    assert r.ok is True, r.errors
    assert r.data["cell"]["cell_id"] == cell
    assert "active_impressions" in r.data
    assert "cell_delta_e76" in r.data


def test_score_printability_returns_components(real_plan) -> None:
    from backend.mcp.tools import introspection

    r = introspection.score_printability(real_plan.plan_id)
    assert r.ok is True, r.errors
    assert 0.0 <= r.data["score_0_100"] <= 100.0
    assert r.data["cell_graph_present"] is True
    assert len(r.data["per_impression"]) == len(real_plan.impressions)


def test_propose_plate_reorganization_writes_candidate(real_plan) -> None:
    from backend.mcp.tools import hitl

    r = hitl.propose_plate_reorganization(real_plan.plan_id, max_actions=6)
    assert r.ok is True, r.errors
    assert r.data["mutation_applied"] is False
    assert Path(r.data["candidate_path"]).is_file()
    assert "recommended_actions" in r.data
