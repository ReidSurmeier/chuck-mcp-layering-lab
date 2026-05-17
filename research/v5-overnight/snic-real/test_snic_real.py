"""TDD test suite for SNIC-real cell proposer.

Cycle 1: real cell extraction (>=1500 cells on Emma 2048).
Cycle 2: hue cluster diversity (>=5 distinct hue clusters).
Cycle 3: per-cell properties (mean_lab, chroma, role_hint, adjacency_list).
Cycle 4: integration patch — chuck_mcp_v2.plan_emma uses snic, not grid.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EMMA = (
    REPO_ROOT
    / "research"
    / "v3-construction"
    / "mediapipe-face-spatial"
    / "close_emma_2002_2048.jpg"
)


# ---------------------------------------------------------------------------
# Cycle 1 — real cell extraction
# ---------------------------------------------------------------------------
def test_snic_on_real_emma_produces_at_least_1500_cells(tmp_path):
    """SNIC on Emma 2048 must produce >= 1500 superpixel cells.

    The test EXERCISES real SNIC/SLIC on the actual Emma JPG — no mocks at the
    segmentation level. File I/O is the only thing that may be redirected.
    """
    assert EMMA.exists(), f"Emma image missing: {EMMA}"

    from snic_proposer import propose_cells  # noqa: E402

    cell_graph = propose_cells(str(EMMA), target_cells=2000, max_dim=2048)

    assert isinstance(cell_graph, dict)
    assert "cells" in cell_graph
    assert (
        len(cell_graph["cells"]) >= 1500
    ), f"got {len(cell_graph['cells'])} cells, expected >= 1500"


# ---------------------------------------------------------------------------
# Cycle 2 — hue cluster diversity
# ---------------------------------------------------------------------------
def test_snic_emma_cells_have_at_least_5_hue_clusters():
    """Emma has skin, hair, eyes, lips, background — at minimum 5 hue clusters.

    Group cells by Lab hue (atan2(b*, a*) binned every 36deg = 10 bins),
    require >= 5 bins to be populated by >=30 cells each (so a single
    speckle isn't counted).
    """
    from snic_proposer import propose_cells, hue_cluster_count

    cell_graph = propose_cells(str(EMMA), target_cells=2000, max_dim=2048)
    n_clusters = hue_cluster_count(cell_graph, min_cells_per_cluster=30, n_bins=10)

    assert (
        n_clusters >= 5
    ), f"got {n_clusters} hue clusters (>=30 cells each), expected >= 5"


# ---------------------------------------------------------------------------
# Cycle 3 — per-cell properties
# ---------------------------------------------------------------------------
def test_cells_have_lab_chroma_role_hint_adjacency():
    """Each cell must carry mean_lab, chroma, role_hint, adjacency_list."""
    from snic_proposer import propose_cells

    cell_graph = propose_cells(str(EMMA), target_cells=2000, max_dim=2048)

    required_keys = {
        "mean_rgb",
        "mean_lab",
        "chroma",
        "role_hint",
        "adjacency_list",
        "pixels",
        "centroid_yx",
    }
    allowed_roles = {"skin", "hair", "eyes", "lip", "background", "other"}

    cells = cell_graph["cells"]
    seen_roles: set[str] = set()
    for cid, cell in cells.items():
        missing = required_keys - set(cell.keys())
        assert not missing, f"cell {cid} missing keys: {missing}"
        assert cell["role_hint"] in allowed_roles, (
            f"cell {cid} bad role_hint {cell['role_hint']!r}"
        )
        assert isinstance(cell["adjacency_list"], list)
        seen_roles.add(cell["role_hint"])

    # Emma has at minimum 5 role hints (skin/hair/eyes/lip/background)
    assert len(seen_roles) >= 5, (
        f"only {len(seen_roles)} distinct role_hints: {sorted(seen_roles)}"
    )

    # Adjacency must be reciprocal and reference real cell ids
    cell_ids = set(cells.keys())
    for cid, cell in cells.items():
        for neigh in cell["adjacency_list"]:
            assert neigh in cell_ids, f"cell {cid} -> unknown neighbor {neigh}"
            assert (
                cid in cells[neigh]["adjacency_list"]
            ), f"adjacency not reciprocal: {cid} <-/-> {neigh}"


# ---------------------------------------------------------------------------
# Cycle 4 — integration patch
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_plan_emma_uses_snic_not_grid(tmp_path):
    """Running plan_emma on the real image must emit a production plan whose
    cell_graph metadata reports `cell_proposal_source = "snic"`.

    The grid placeholder produced `cell_proposal_source = "grid"`. We assert
    SNIC pathway is wired into the canonical chuck_mcp_v2.plan_emma module.
    """
    plan_path = tmp_path / "production_plan.json"
    result_path = tmp_path / "hybrid_result.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    # Keep the run fast: tiny solve space, low iter counts. We only care about
    # the production plan structure here, not the solver convergence.
    cmd = [
        sys.executable,
        "-m",
        "chuck_mcp_v2.plan_emma",
        str(EMMA),
        "--size",
        "128",
        "--cells",
        "2000",
        "--plate-count",
        "26",
        "--max-outer-iters",
        "1",
        "--max-inner-iters",
        "1",
        "--mill-radius-px",
        "1",
        # Skip the v5 mediapipe constraint step — orthogonal to SNIC integration
        # and gated on optional cv2 / mediapipe deps that may not be present.
        "--no-face-regions",
        # Skip the v5 mokuhanga-pigments adapter — also orthogonal and gated
        # on yaml + the rule classifier dependency tree.
        "--no-mokuhanga-pigments",
        "--output",
        str(result_path),
        "--plan-output",
        str(plan_path),
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (
        f"plan_emma failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    assert plan_path.exists(), "ProductionPlan JSON not written"
    plan = json.loads(plan_path.read_text())

    meta = plan.get("meta", {})
    assert meta.get("cell_proposal_source") == "snic", (
        f"cell_proposal_source={meta.get('cell_proposal_source')!r} "
        f"(expected 'snic'); meta={meta}"
    )
    assert plan.get("cell_count", 0) >= 1500, (
        f"cell_count={plan.get('cell_count')}, expected >= 1500"
    )
    assert meta.get("hue_cluster_count", 0) >= 5, (
        f"hue_cluster_count={meta.get('hue_cluster_count')}, expected >= 5"
    )
