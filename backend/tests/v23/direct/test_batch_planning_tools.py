"""Production batch planning MCP tests."""
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
    arr = np.full((28, 28, 3), [246, 240, 226], dtype=np.uint8)
    arr[:14, :14] = [246, 180, 190]
    arr[:14, 14:] = [170, 220, 236]
    arr[14:, :14] = [226, 108, 52]
    arr[14:, 14:] = [34, 48, 62]
    img_path = p / "batch_plan.png"
    Image.fromarray(arr, "RGB").save(img_path)
    return img_path


@pytest.fixture
def real_plan(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    from backend.services.v23 import orchestrator as _orch
    return _orch.run_pipeline_partial(str(_write_test_image(tmp_path)), solve_profile="fast")


def test_plan_production_batches_writes_four_four_detail_plan(real_plan) -> None:
    from backend.mcp.tools import planning

    r = planning.plan_production_batches(real_plan.plan_id, detail_slots=6)

    assert r.ok is True, r.errors
    assert Path(r.data["batch_plan_path"]).is_file()
    assert Path(r.data["cell_labels_path"]).is_file()
    assert r.data["diagnostics"]["batch_count"] == 3
    assert r.data["diagnostics"]["target_first_batch_plates"] == 4
    assert r.data["diagnostics"]["target_second_batch_plates"] == 4
    assert len(r.data["batches"][0]["plates"]) == 4
    assert len(r.data["batches"][1]["plates"]) == 4
    assert len(r.data["batches"][2]["plates"]) <= 6
    first_plate = r.data["batches"][0]["plates"][0]
    assert "cell_ids" in first_plate
    assert "suggested_alpha" in first_plate


def test_plan_adaptive_ink_stack_solves_with_adaptive_colors(real_plan) -> None:
    from backend.mcp.tools import planning

    r = planning.plan_adaptive_ink_stack(real_plan.plan_id, max_plates=12)

    assert r.ok is True, r.errors
    assert Path(r.data["adaptive_plan_path"]).is_file()
    assert r.data["diagnostics"]["template"] == "adaptive_ink_batches"
    assert r.data["diagnostics"]["plate_count"] <= 12
    plates = [p for batch in r.data["batches"] for p in batch["plates"]]
    assert any("ink_rgb" in plate for plate in plates)
    assert all("suggested_pigments" not in plate for plate in plates)
