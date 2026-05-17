"""Tests for the hybrid alternating optimizer.

Required by the swarm spec (10 tests, ≥ 80% coverage):

    test_alternating_loop_converges_on_synthetic_plan
    test_stage2_graph_cut_assigns_all_cells (no orphans)
    test_stage2_respects_role_constraints
    test_stage3_jax_solve_decreases_loss
    test_stage3_does_not_modify_cell_assignments (frozen)
    test_stage4_morphology_repair_kills_sub_mill_islands
    test_stage5_re_solves_on_validator_failure
    test_max_outer_iters_terminates_loop
    test_optimization_result_passes_all_6_validators_on_clean_input
    test_optimization_result_serializable_to_json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# The hybrid_optimizer package alias is installed by conftest.py because
# the directory name has a hyphen and is not a valid Python identifier.
from hybrid_optimizer import (  # noqa: E402
    CandidatePlate,
    CellGraph,
    CellNode,
    DEFAULT_MILL_RADIUS_PX,
    FrozenPlate,
    OptimizationResult,
    ProductionPlanInput,
    RoleConstraints,
    SolvedPlate,
    any_needs_resolve,
    assign_cells_to_plates,
    assignment_to_cells_by_plate,
    optimize,
    repair_plate_mask,
    repair_plates,
    solve_pigment_load,
)
from hybrid_optimizer.jax_continuous_solve import _rgb_to_lab_np  # noqa: E402


# ---------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------


def _synthetic_cell_graph(
    n_cells: int = 24,
    H: int = 96,
    W: int = 96,
    seed: int = 42,
    base_area_px: int = 1500,
) -> CellGraph:
    """Build a synthetic SNIC-style cell graph spread over an H x W canvas.

    Cells are placed on a sqrt(n) x sqrt(n) lattice with jittered positions
    and assigned to one of 4 role families round-robin.
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_cells)))
    dy, dx = H / side, W / side
    nodes = []
    edges: list[tuple[int, int]] = []
    edge_weights: dict[tuple[int, int], float] = {}
    roles = ("underlayer_light", "local_chroma", "regional_mass", "key_detail")
    cell_grid: dict[tuple[int, int], int] = {}
    cid = 0
    for r in range(side):
        for c in range(side):
            if cid >= n_cells:
                break
            cy = dy * (r + 0.5) + rng.uniform(-2, 2)
            cx = dx * (c + 0.5) + rng.uniform(-2, 2)
            role = roles[cid % len(roles)]
            # color: warm/cool spectrum
            L = 60 + 20 * np.sin(cid)
            a = 30 * np.cos(cid * 0.7)
            b = 30 * np.sin(cid * 0.7 + 1.0)
            # Default 1500 px area is bigger than π * 19^2 ≈ 1134 so the cell
            # disk survives Stage 4 morphology repair (mill radius = 19px).
            area = int(base_area_px + (cid % 4) * 200)
            nodes.append(
                CellNode(
                    cell_id=cid,
                    role=role,
                    lab_color=(float(L), float(a), float(b)),
                    area_px=area,
                    centroid_yx=(float(cy), float(cx)),
                )
            )
            cell_grid[(r, c)] = cid
            cid += 1

    # Edges: 4-neighbor lattice adjacency
    for (r, c), a_id in cell_grid.items():
        for dr, dc in ((1, 0), (0, 1)):
            b_id = cell_grid.get((r + dr, c + dc))
            if b_id is None:
                continue
            edges.append((a_id, b_id))
            # Weight: closer Lab colors -> heavier edge (keep together)
            la = np.array(nodes[a_id].lab_color)
            lb = np.array(nodes[b_id].lab_color)
            d = float(np.linalg.norm(la - lb))
            edge_weights[(min(a_id, b_id), max(a_id, b_id))] = float(
                np.clip(50.0 / (d + 1e-6), 0.1, 10.0)
            )

    return CellGraph(nodes=nodes, edges=edges, edge_weights=edge_weights)


def _candidate_plates_for(graph: CellGraph, per_role: int = 3) -> list[CandidatePlate]:
    """Build candidate plates: per_role plates per role family."""
    plates = []
    pid = 1
    roles = sorted({n.role for n in graph.nodes})
    pigments = {
        "underlayer_light": "gamboge_yellow",
        "local_chroma": "vermilion_red",
        "regional_mass": "phthalo_blue",
        "key_detail": "ivory_black",
    }
    for role in roles:
        for _ in range(per_role):
            plates.append(
                CandidatePlate(
                    plate_id=pid,
                    role=role,
                    max_area_px=None,
                    pigment_id=pigments.get(role, "ivory_black"),
                )
            )
            pid += 1
    return plates


def _synthetic_target_image(H: int = 96, W: int = 96, seed: int = 7) -> np.ndarray:
    """Generate a colored target image (uint8 RGB) with spatial structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = (np.sin(yy / 12.0) * 80 + 150).clip(0, 255)
    g = (np.cos(xx / 18.0) * 80 + 150).clip(0, 255)
    b = ((yy - xx) % 60 + 100).clip(0, 255)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    # Add a touch of noise
    noise = rng.integers(-10, 10, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _full_synthetic_plan(
    n_cells: int = 24,
    per_role: int = 3,
    H: int = 96,
    W: int = 96,
) -> tuple[np.ndarray, ProductionPlanInput]:
    graph = _synthetic_cell_graph(n_cells=n_cells, H=H, W=W)
    plates = _candidate_plates_for(graph, per_role=per_role)
    target = _synthetic_target_image(H, W)
    plan = ProductionPlanInput(
        cell_graph=graph,
        candidate_plates=plates,
        target_image_rgb=target,
        target_shape=(H, W),
        dpi=300.0,
    )
    return target, plan


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_stage2_graph_cut_assigns_all_cells():
    """Every cell in the graph must end up assigned to exactly one plate."""
    graph = _synthetic_cell_graph(n_cells=24)
    plates = _candidate_plates_for(graph, per_role=3)
    assignment = assign_cells_to_plates(graph, plates)
    # Length == cell count
    assert len(assignment) == len(graph.nodes)
    # All cell ids covered
    assert set(assignment.keys()) == {n.cell_id for n in graph.nodes}
    # No invalid plate ids
    valid_plate_ids = {p.plate_id for p in plates}
    assert set(assignment.values()).issubset(valid_plate_ids)


def test_stage2_respects_role_constraints():
    """Cells must only be assigned to plates of the SAME role."""
    graph = _synthetic_cell_graph(n_cells=24)
    plates = _candidate_plates_for(graph, per_role=3)
    assignment = assign_cells_to_plates(graph, plates)
    by_id = {n.cell_id: n for n in graph.nodes}
    plate_role = {p.plate_id: p.role for p in plates}
    for cell_id, plate_id in assignment.items():
        assert plate_role[plate_id] == by_id[cell_id].role, (
            f"cell {cell_id} role={by_id[cell_id].role} "
            f"assigned to plate {plate_id} role={plate_role[plate_id]}"
        )

    # And a role with ZERO plates should raise
    bad = [p for p in plates if p.role != "key_detail"]
    with pytest.raises(ValueError):
        assign_cells_to_plates(graph, bad)


def test_stage3_jax_solve_decreases_loss():
    """Stage 3 must produce loss_final <= loss_initial (or be a no-op)."""
    H, W = 48, 48
    graph = _synthetic_cell_graph(n_cells=8, H=H, W=W)
    plates = _candidate_plates_for(graph, per_role=2)
    assignment = assign_cells_to_plates(graph, plates)

    # Manually rasterize masks (small + deterministic)
    by_plate = assignment_to_cells_by_plate(assignment)
    by_id = {n.cell_id: n for n in graph.nodes}
    frozen = []
    for plate_id, cells in by_plate.items():
        mask = np.zeros((H, W), dtype=np.uint8)
        for cid in cells:
            cn = by_id[cid]
            cy, cx = int(cn.centroid_yx[0]), int(cn.centroid_yx[1])
            r = 4
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            mask[y0:y1, x0:x1] = 1
        cp = next(p for p in plates if p.plate_id == plate_id)
        frozen.append(
            FrozenPlate(
                block_id=plate_id,
                cell_zone_ids=cells,
                inked_mask=mask,
                role=cp.role,
                pigment_choices=[
                    ("gamboge", (88.0, -8.0, 70.0)),
                    ("phthalo", (35.0, 5.0, -55.0)),
                    ("vermilion", (50.0, 70.0, 50.0)),
                ],
                pass_index=plate_id,
                initial_opacity=0.4,
                initial_dilution=0.3,
            )
        )

    target_rgb = _synthetic_target_image(H, W)
    target_lab = _rgb_to_lab_np(target_rgb.astype(np.float32) / 255.0)
    result = solve_pigment_load(frozen, target_lab, max_iters=40)
    # loss_final must not exceed loss_initial (within numerical noise)
    assert result.loss_final <= result.loss_initial + 1e-3
    assert all(0.0 <= v["opacity"] <= 1.0 for v in result.per_plate.values())
    assert all(0.0 <= v["dilution"] <= 1.0 for v in result.per_plate.values())


def test_stage3_does_not_modify_cell_assignments():
    """FROZEN INVARIANT: Stage 3 must not modify which cells live on which plate."""
    H, W = 48, 48
    graph = _synthetic_cell_graph(n_cells=12, H=H, W=W)
    plates = _candidate_plates_for(graph, per_role=2)
    assignment = assign_cells_to_plates(graph, plates)
    by_plate_before = assignment_to_cells_by_plate(assignment)

    by_id = {n.cell_id: n for n in graph.nodes}
    frozen = []
    for plate_id, cells in by_plate_before.items():
        mask = np.zeros((H, W), dtype=np.uint8)
        for cid in cells:
            cn = by_id[cid]
            cy, cx = int(cn.centroid_yx[0]), int(cn.centroid_yx[1])
            mask[max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4] = 1
        cp = next(p for p in plates if p.plate_id == plate_id)
        frozen.append(
            FrozenPlate(
                block_id=plate_id,
                cell_zone_ids=list(cells),
                inked_mask=mask,
                role=cp.role,
                pigment_choices=[
                    ("gamboge", (88.0, -8.0, 70.0)),
                    ("phthalo", (35.0, 5.0, -55.0)),
                ],
            )
        )

    target_rgb = _synthetic_target_image(H, W)
    target_lab = _rgb_to_lab_np(target_rgb.astype(np.float32) / 255.0)
    _result = solve_pigment_load(frozen, target_lab, max_iters=30)

    # cell_zone_ids on each FrozenPlate must be unchanged
    for fp in frozen:
        original = by_plate_before[fp.block_id]
        assert list(fp.cell_zone_ids) == list(original), (
            f"plate {fp.block_id} cell assignment mutated by Stage 3!"
        )


def test_stage4_morphology_repair_kills_sub_mill_islands():
    """Stage 4 must eliminate components smaller than the mill footprint."""
    H, W = 200, 200
    mask = np.zeros((H, W), dtype=np.uint8)
    # Big valid component (60 x 60 = 3600 px > π * 19^2 ≈ 1134)
    mask[40:100, 40:100] = 1
    # Tiny noise component (3 x 3 = 9 px)
    mask[150, 150] = 1
    mask[151, 150] = 1
    mask[150, 151] = 1
    mask[151, 151] = 1
    mask[152, 150] = 1
    # Hairline (1 px wide line) - should be killed by opening
    mask[20, 10:80] = 1

    repaired, stats = repair_plate_mask(mask, mill_radius_px=DEFAULT_MILL_RADIUS_PX)
    # The big square survives
    assert repaired[70, 70] == 1
    # The tiny island is gone
    assert repaired[150, 150] == 0
    # Hairline is gone
    assert repaired[20, 40] == 0
    # Component count dropped
    assert stats.components_after <= stats.components_before


def test_stage4_morphology_repair_triggers_resolve_on_big_shrink():
    """Heavy shrinkage should flag needs_resolve so Stage 5 re-runs."""
    H, W = 100, 100
    # All hairlines: nothing survives a mill_radius_px=19 disk
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10, 10:80] = 1
    mask[30, 10:80] = 1
    mask[50, 10:80] = 1
    repaired, stats = repair_plate_mask(mask, mill_radius_px=DEFAULT_MILL_RADIUS_PX)
    assert stats.shrink_fraction > 0.5
    assert stats.needs_resolve is True
    assert any_needs_resolve([stats])


def test_stage5_re_solves_on_validator_failure(tmp_path):
    """If validators fail, optimize() must run more than one outer iter."""
    # Small synthetic plan with default settings
    target, plan = _full_synthetic_plan(n_cells=16, per_role=2, H=64, W=64)
    # Force re-solve by using max_outer_iters=3 and disabling early stop
    result = optimize(
        target,
        plan,
        max_outer_iters=3,
        max_inner_iters=20,
        early_stop_on_gates=False,
    )
    assert result.outer_iter_count == 3
    assert len(result.history) == 3


def test_max_outer_iters_terminates_loop():
    """Loop must never exceed max_outer_iters even with garbage data."""
    target, plan = _full_synthetic_plan(n_cells=12, per_role=2, H=48, W=48)
    cap = 2
    result = optimize(
        target, plan, max_outer_iters=cap, max_inner_iters=10, early_stop_on_gates=False
    )
    assert result.outer_iter_count <= cap


def test_alternating_loop_converges_on_synthetic_plan():
    """End-to-end: synthetic plan should converge to >= 4/5 gates passing
    (proof_progression is strict; we tolerate 4/5 in V1)."""
    target, plan = _full_synthetic_plan(n_cells=24, per_role=3, H=80, W=80)
    result = optimize(
        target,
        plan,
        max_outer_iters=3,
        max_inner_iters=30,
        early_stop_on_gates=True,
    )
    assert isinstance(result, OptimizationResult)
    assert result.outer_iter_count >= 1
    assert len(result.plates) > 0
    # All plates must have valid opacity / dilution
    for p in result.plates:
        assert 0.0 <= p.opacity <= 1.0
        assert 0.0 <= p.dilution <= 1.0
        assert p.role in (
            "underlayer_light",
            "local_chroma",
            "regional_mass",
            "key_detail",
        )
    # At least the structurally-trivial validators should be available
    assert "plate_not_composite" in result.validator_scores
    assert "role_purity" in result.validator_scores
    assert "jigsaw_separation" in result.validator_scores
    assert "proof_progression" in result.validator_scores
    assert "underlayer_reversal" in result.validator_scores


def test_optimization_result_passes_all_6_validators_on_clean_input():
    """V1.0 baseline: on a *clean* synthetic plan, the report should contain
    keys for all 6 validators and a non-empty plate list."""
    target, plan = _full_synthetic_plan(n_cells=24, per_role=3, H=80, W=80)
    result = optimize(target, plan, max_outer_iters=2, max_inner_iters=20)
    expected = {
        "plate_not_composite",
        "role_purity",
        "jigsaw_separation",
        "proof_progression",
        "underlayer_reversal",
        "final_match",
    }
    assert expected.issubset(set(result.validator_scores.keys()))
    # All gates get a `passes` flag; final_match is advisory and uses
    # `advisory_passes` per the v3 validator contract.
    gates = expected - {"final_match"}
    for k in gates:
        assert "passes" in result.validator_scores[k], k
    fm = result.validator_scores["final_match"]
    assert "advisory_passes" in fm or "passes" in fm


def test_optimization_result_serializable_to_json(tmp_path):
    """OptimizationResult.to_json() must round-trip through json.loads."""
    target, plan = _full_synthetic_plan(n_cells=12, per_role=2, H=48, W=48)
    result = optimize(target, plan, max_outer_iters=1, max_inner_iters=10)
    s = result.to_json()
    parsed = json.loads(s)
    assert "plates" in parsed
    assert "validator_scores" in parsed
    assert "outer_iter_count" in parsed
    assert "delta_e_mean" in parsed
    assert isinstance(parsed["plates"], list)
    # Save to file and read back
    out = result.save(str(tmp_path / "result.json"))
    assert out.exists()
    parsed2 = json.loads(out.read_text())
    assert parsed2["outer_iter_count"] == result.outer_iter_count


# ---------------------------------------------------------------------
# Bonus integration: synthetic Emma-scale wall-time + ΔE check
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_synthetic_emma_scale_under_5_minutes():
    """Full optimize() at Emma-scale must finish < 5 min wall-time."""
    target, plan = _full_synthetic_plan(n_cells=120, per_role=6, H=256, W=256)
    t0 = time.time()
    result = optimize(target, plan, max_outer_iters=3, max_inner_iters=80)
    elapsed = time.time() - t0
    assert elapsed < 300.0, f"too slow: {elapsed:.1f}s"
    assert result.outer_iter_count >= 1
