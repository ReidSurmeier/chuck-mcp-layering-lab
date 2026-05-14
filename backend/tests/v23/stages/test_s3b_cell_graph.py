"""S3.b cell graph tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _grid_target() -> np.ndarray:
    img = np.full((72, 72, 3), [246, 241, 227], dtype=np.uint8)
    img[8:34, 8:34] = [236, 186, 124]
    img[8:34, 38:64] = [114, 170, 204]
    img[38:64, 8:34] = [232, 98, 74]
    img[38:64, 38:64] = [44, 58, 74]
    return img


def test_cell_graph_builds_cells_and_adjacency() -> None:
    from backend.services.v23.stages.s3b_cell_graph import build_cell_graph

    result = build_cell_graph(_grid_target(), n_cells=36)

    assert result.labels.shape == (72, 72)
    assert result.labels.dtype == np.int32
    assert result.diagnostics["cell_count"] >= 12
    assert result.diagnostics["adjacency_count"] > 0
    assert len(result.cells) == result.diagnostics["cell_count"]
    assert {"cell_id", "mean_hex", "role_hint", "paper_delta_e"} <= set(result.cells[0])


def test_cell_graph_detects_subtle_tint_and_key_hints() -> None:
    from backend.services.v23.stages.s3b_cell_graph import build_cell_graph

    result = build_cell_graph(_grid_target(), n_cells=36)
    hints = {cell["role_hint"] for cell in result.cells}

    assert "key_or_shadow" in hints
    assert "local_chroma" in hints or "regional_mass" in hints


def test_cell_graph_persists_json_and_labels(tmp_path: Path) -> None:
    import json

    from backend.services.v23.stages.s3b_cell_graph import build_cell_graph, persist_cell_graph

    result = build_cell_graph(_grid_target(), n_cells=28)
    paths = persist_cell_graph(result, tmp_path)

    labels = np.load(paths["cell_labels_path"])
    payload = json.loads(Path(paths["cell_graph_path"]).read_text())
    assert labels.shape == (72, 72)
    assert payload["diagnostics"]["cell_count"] == len(payload["cells"])
    assert Path(payload["labels_path"]).is_file()
