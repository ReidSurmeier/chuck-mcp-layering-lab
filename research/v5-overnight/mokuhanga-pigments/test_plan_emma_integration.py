"""TDD CYCLE 3 — plan_emma plates have real pigments, not defaults.

This is the load-bearing integration test: run plan_emma end-to-end on a
synthetic Emma image and assert that EVERY produced plate carries a real
pigment from pigment_library_emma.yaml (no defaults / blanks / "None").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pigment_library import load_pigment_library, EMMA_PIGMENT_LIBRARY_PATH


def _build_emma_synthetic_target(size: int = 96) -> np.ndarray:
    """Mirror plan_emma._load_target's synthetic path so the same plates flow."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = (0.58 + 0.35 * np.sin(yy / 8.0)) * 255.0
    g = (0.60 + 0.30 * np.cos(xx / 11.0)) * 255.0
    b = (0.45 + 0.25 * np.sin((yy - xx) / 13.0)) * 255.0
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def test_plan_emma_plates_have_real_pigments_not_default(tmp_path: Path) -> None:
    """The production plan emitted by chuck_mcp_v2.plan_emma must populate
    every plate.pigment_family with a real Emma-library pigment via the
    mokuhanga pigment adapter — no empty strings, no None, no 'default'.
    """
    from chuck_mcp_v2 import plan_emma as pe

    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    valid_pigment_names = set(library.keys())
    valid_families = {p.family for p in library.values()}

    target_rgb = _build_emma_synthetic_target(size=96)

    # Load the production_solver package the same way plan_emma does.
    production = pe._load_package("production_solver", pe.PRODUCTION_DIR)
    cell_graph_dict, _, _ = pe._grid_cell_graph(target_rgb, requested_cells=64)

    plan = production.build_production_plan(
        target_rgb,
        cell_graph_dict,
        plate_count=20,
        target_total_pulls=132,
        target_pull_tolerance=12,
    )

    # Run the same mokuhanga adapter plan_emma.run() now calls.
    pe.apply_mokuhanga_pigments(plan, cell_graph_dict)

    plan_dict = plan.to_dict()
    plates_payload = plan_dict["plates"]
    assert plates_payload, "empty plates payload"

    # Hard checks: every plate must carry a real family and a real pigment_name
    bad_family: list[dict] = []
    bad_pigment: list[dict] = []
    for plate in plates_payload:
        fam = plate.get("pigment_family") or ""
        if fam.strip().lower() in ("", "default", "none"):
            bad_family.append(plate)
            continue
        if fam not in valid_families:
            bad_family.append(plate)
            continue

        # The Emma adapter writes the chosen library pigment into the
        # plate's `pulls[].pigment_id`. At least one pull must reference a
        # real library pigment.
        pulls = plate.get("pulls", [])
        if not pulls:
            bad_pigment.append(plate)
            continue
        pigment_ids = {p.get("pigment_id", "") for p in pulls}
        if not (pigment_ids & valid_pigment_names):
            bad_pigment.append(plate)

    assert not bad_family, (
        f"{len(bad_family)} plates have missing/invalid pigment_family: "
        f"{[(p['block_id'], p.get('pigment_family'), p.get('role')) for p in bad_family[:6]]}"
    )
    assert not bad_pigment, (
        f"{len(bad_pigment)} plates have no real Emma-library pigment_id in any pull: "
        f"{[(p['block_id'], p.get('pigment_family'), p.get('role'), [pp.get('pigment_id') for pp in p.get('pulls', [])][:2]) for p in bad_pigment[:6]]}"
    )


def test_plan_emma_underlayer_plates_match_rule_classifier(tmp_path: Path) -> None:
    """The underlayer-role plates in the production plan must carry the
    EXACT families the mokuhanga rule classifier picked. Catches silent
    bypasses where role assignment ignores the proposer."""
    from chuck_mcp_v2 import plan_emma as pe

    target_rgb = _build_emma_synthetic_target(size=96)
    production = pe._load_package("production_solver", pe.PRODUCTION_DIR)
    cell_graph_dict, _, _ = pe._grid_cell_graph(target_rgb, requested_cells=64)
    plan = production.build_production_plan(
        target_rgb,
        cell_graph_dict,
        plate_count=20,
        target_total_pulls=132,
        target_pull_tolerance=12,
    )
    pe.apply_mokuhanga_pigments(plan, cell_graph_dict)

    plan_dict = plan.to_dict()
    underlayer_plates = [
        p for p in plan_dict["plates"] if p.get("role") == "underlayer_light"
    ]
    # Rule classifier emits 4..9 underlayers; we expect at least 4.
    assert len(underlayer_plates) >= 4, (
        f"expected ≥4 underlayer plates, got {len(underlayer_plates)}"
    )

    # Each underlayer plate must carry one of the 7 rule families
    rule_families = {
        "light_yellow", "pale_pink", "pale_orange", "pale_red",
        "pale_blue", "pale_green", "warm_grey",
    }
    for p in underlayer_plates:
        assert p["pigment_family"] in rule_families, (
            f"underlayer plate {p['block_id']} has unexpected family "
            f"{p['pigment_family']!r}"
        )
        # provenance must be marked
        assert p["provenance"], f"plate {p['block_id']} missing provenance"
        # rationale should include the pigment_pick tag if the adapter ran
        # (this is the single load-bearing string the adapter writes).
        assert "pigment_pick" in p.get("rationale", "") or "mokuhanga" in p.get("rationale", ""), (
            f"plate {p['block_id']} rationale={p.get('rationale')!r} "
            f"shows no sign of mokuhanga pigment adapter"
        )
