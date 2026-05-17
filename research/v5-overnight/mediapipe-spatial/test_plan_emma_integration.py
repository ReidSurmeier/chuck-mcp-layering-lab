"""TDD Cycle 4 — plan_emma underlayer plates respect face_region_constraint.

This test verifies the inline patch: after `production.build_production_plan`
returns plates, plan_emma must run the v5 mediapipe pipeline and re-tag every
underlayer_light plate with:

    plate.face_region_constraint = ["left_cheek", ...]   # 1-3 region names

AND every cell_zone_id on that plate must satisfy:

    cell_to_region[cell_id] in plate.face_region_constraint

The test runs the real plan_emma `run()` end-to-end on the Chuck Close Emma
image at a reduced size/cells/plate-count for speed, then inspects the
production_plan JSON.

We exercise the helper that does the work
(`plan_emma.apply_face_region_constraints`) directly so the test is fast and
deterministic, then a second end-to-end check on the JSON dump confirms the
helper is wired into the CLI flow.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


EMMA_PATH = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")


@pytest.fixture(scope="module")
def emma_path() -> Path:
    if not EMMA_PATH.exists():
        pytest.skip(f"Emma input image not found at {EMMA_PATH}")
    return EMMA_PATH


def test_apply_face_region_constraints_helper_exists() -> None:
    """plan_emma must expose `apply_face_region_constraints(plan, image,
    cell_graph)` so unit tests can exercise the integration without running
    the full CLI."""
    from chuck_mcp_v2 import plan_emma
    assert hasattr(plan_emma, "apply_face_region_constraints"), (
        "plan_emma must expose apply_face_region_constraints() — the v5 patch "
        "wires this to the underlayer-plate tagging step"
    )


def test_apply_face_region_constraints_tags_underlayer_plates(emma_path: Path) -> None:
    """The helper must:
      1. Run face_spatial.extract_face_regions on the input image.
      2. Build a per-cell region label from the grid cell graph.
      3. Mutate or replace every underlayer_light plate so that
         - plate.face_region_constraint is a list[str] of length 1..6
         - every cell_zone_id on the plate has a primary region in that list
    """
    import importlib
    from chuck_mcp_v2 import plan_emma
    importlib.reload(plan_emma)

    # Build a tiny but real pipeline state to exercise the helper.
    target_rgb = plan_emma._load_target(str(emma_path), synthetic=False, size=256)
    cell_graph_dict, _, _ = plan_emma._grid_cell_graph(target_rgb, requested_cells=64)

    production = plan_emma._load_package("production_solver", plan_emma.PRODUCTION_DIR)
    plan = production.build_production_plan(
        target_rgb,
        cell_graph_dict,
        plate_count=20,
        target_total_pulls=60,
        target_pull_tolerance=20,
    )

    new_plan = plan_emma.apply_face_region_constraints(
        plan, target_rgb, cell_graph_dict, image_path=str(emma_path),
    )

    underlayers = [p for p in new_plan.plates if p.role == "underlayer_light"]
    assert underlayers, "no underlayer_light plates in the plan — invalid fixture"

    for plate in underlayers:
        constraint = getattr(plate, "face_region_constraint", None)
        assert constraint, (
            f"underlayer plate {plate.block_id} has no face_region_constraint"
        )
        assert isinstance(constraint, list), (
            f"plate {plate.block_id}.face_region_constraint is "
            f"{type(constraint)}, expected list"
        )
        assert 1 <= len(constraint) <= 6, (
            f"plate {plate.block_id}.face_region_constraint has "
            f"{len(constraint)} regions; expected 1..6"
        )


def test_apply_face_region_constraints_filters_cells_to_assigned_regions(
    emma_path: Path,
) -> None:
    """Every cell on an underlayer plate must belong to one of its constraint
    regions. Hair / background cells must NOT appear on a skin-area plate.
    """
    import importlib
    from chuck_mcp_v2 import plan_emma
    importlib.reload(plan_emma)

    target_rgb = plan_emma._load_target(str(emma_path), synthetic=False, size=256)
    cell_graph_dict, _, _ = plan_emma._grid_cell_graph(target_rgb, requested_cells=64)

    production = plan_emma._load_package("production_solver", plan_emma.PRODUCTION_DIR)
    plan = production.build_production_plan(
        target_rgb,
        cell_graph_dict,
        plate_count=20,
        target_total_pulls=60,
        target_pull_tolerance=20,
    )

    new_plan = plan_emma.apply_face_region_constraints(
        plan, target_rgb, cell_graph_dict, image_path=str(emma_path),
    )

    # Recompute the cell->region map to validate independently of helper internals.
    import face_spatial
    import merge_cells_with_regions as mcr

    # Build the same SNIC-like label image plan_emma uses (grid -> per-pixel ID).
    H, W = target_rgb.shape[:2]
    label_img = plan_emma._grid_cell_label_image(cell_graph_dict, (H, W))
    regions = face_spatial.extract_face_regions(str(emma_path))
    # Downscale region masks to (H, W) for compatibility with the grid labels.
    regions_resized = {
        name: _resize_region(r, (H, W)) for name, r in regions.items()
    }
    cell_to_region = mcr.assign_cells_to_regions(regions_resized, label_img)

    for plate in new_plan.plates:
        if plate.role != "underlayer_light":
            continue
        constraint = set(getattr(plate, "face_region_constraint", []) or [])
        for cid in plate.cell_zone_ids:
            region = cell_to_region.get(int(cid))
            assert region in constraint, (
                f"underlayer plate {plate.block_id} (constraint={sorted(constraint)}) "
                f"contains cell {cid} whose primary region is {region!r} — "
                f"hair/background/lip/other cells must NOT end up here"
            )


def test_plan_emma_run_writes_constraint_to_plan_json(
    emma_path: Path, tmp_path: Path,
) -> None:
    """End-to-end: invoking `plan_emma.run` writes a production_plan JSON in
    which every underlayer plate carries a non-empty face_region_constraint.
    """
    from chuck_mcp_v2 import plan_emma

    plan_out = tmp_path / "production_plan.json"
    result_out = tmp_path / "emma_hybrid_result.json"

    args = plan_emma.build_parser().parse_args([
        str(emma_path),
        "--output", str(result_out),
        "--plan-output", str(plan_out),
        "--size", "256",
        "--cells", "64",
        "--plate-count", "20",
        "--target-pulls", "60",
        "--target-pull-tolerance", "20",
        "--max-outer-iters", "1",
        "--max-inner-iters", "1",
    ])
    plan_emma.run(args)

    assert plan_out.exists(), "plan JSON not written"
    data = json.loads(plan_out.read_text())

    underlayers = [p for p in data["plates"] if p["role"] == "underlayer_light"]
    assert underlayers, "no underlayer_light plates in serialized plan"
    for p in underlayers:
        c = p.get("face_region_constraint")
        assert c and isinstance(c, list), (
            f"plate {p['block_id']} missing face_region_constraint in JSON; "
            f"got {c!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_region(region, target_shape: tuple[int, int]):
    """Resize a FaceRegion's mask (and polygon) to `target_shape` (H, W) for
    test-side validation against the grid label image."""
    import face_spatial
    H, W = target_shape
    src_h, src_w = region.mask.shape
    if (src_h, src_w) == (H, W):
        return region
    new_mask = cv2.resize(region.mask, (W, H), interpolation=cv2.INTER_NEAREST)
    new_poly = None
    if region.polygon is not None:
        sx, sy = W / src_w, H / src_h
        new_poly = [(int(round(x * sx)), int(round(y * sy)))
                    for (x, y) in region.polygon]
    # Recompute bbox+centroid from resized mask
    ys, xs = np.where(new_mask > 0)
    if len(xs) == 0:
        bbox = None
        centroid = None
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        centroid = (float(xs.mean()), float(ys.mean()))
    return face_spatial.FaceRegion(
        name=region.name, polygon=new_poly, mask=new_mask,
        source=region.source, confidence=region.confidence,
        centroid=centroid, bbox=bbox,
    )
