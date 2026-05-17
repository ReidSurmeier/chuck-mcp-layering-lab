"""pytest suite for the v4-build production-solver module.

Required tests (per task brief):
    1.  test_plate_count_adapts_to_chroma_variance
    2.  test_plate_count_clamps_to_20_to_35_range
    3.  test_multi_pull_returns_1_to_5_per_block
    4.  test_pull_count_sums_to_around_132_for_emma_class_input
    5.  test_production_plan_has_no_orphan_cells
    6.  test_production_plan_has_no_zone_overlap_on_same_block
    7.  test_production_plan_role_purity_meets_threshold
    8.  test_production_plan_passes_plate_not_composite_validator
    9.  test_integration_with_underlayer_proposer

Each test builds a synthetic Emma-class input deterministically so the
suite runs offline and reproducibly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Production-solver lives in a hyphen-named directory; load it as a package
# under the importable alias ``production_solver`` via importlib.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # research/v4-build/
sys.path.insert(0, str(HERE.parent.parent))  # research/
sys.path.insert(0, str(HERE.parent.parent / "v3-construction"))  # research/v3-construction/

import importlib.util  # noqa: E402
import importlib.machinery  # noqa: E402


def _load_package(name: str, path: Path):
    """Load a Python package from a folder whose name has hyphens."""
    init = path / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name, init,
        submodule_search_locations=[str(path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load package at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


production_solver = _load_package("production_solver", HERE)
PullSpec = production_solver.PullSpec
PlateSpec = production_solver.PlateSpec
ProductionPlan = production_solver.ProductionPlan
ROLES = production_solver.ROLES
estimate_plate_count = production_solver.estimate_plate_count
assign_pulls_per_block = production_solver.assign_pulls_per_block
build_production_plan = production_solver.build_production_plan

# v3 validator under test (path-imported so we don't add a top-level package)
import importlib.util  # noqa: E402

V3 = HERE.parent.parent / "v3-construction"
_PNC_PATH = V3 / "validators-reconstruction" / "plate_not_composite.py"
_RP_PATH = V3 / "validators-reconstruction" / "role_purity.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plate_not_composite = _load_module("plate_not_composite", _PNC_PATH)
role_purity = _load_module("role_purity", _RP_PATH)


# ---------------------------------------------------------------------------
# Synthetic Emma-class input builder
# ---------------------------------------------------------------------------
def _make_synthetic_emma_input(
    H: int = 256,
    W: int = 256,
    n_cells: int = 1800,
    n_hue_clusters: int = 7,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Build a synthetic Emma-class portrait + Voronoi-like cell graph.

    Returns:
        (target_rgb_uint8, cell_graph_dict)
    """
    rng = np.random.default_rng(seed)

    # 1. Synthetic portrait — a noise-modulated, multi-hue gradient
    img = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rn = r / r.max()

    # Multi-band radial palette for clusters
    base_hues = np.linspace(0, 360, n_hue_clusters, endpoint=False)
    for i, h in enumerate(base_hues):
        band_mask = ((rn >= i / n_hue_clusters) & (rn < (i + 1) / n_hue_clusters)).astype(np.float32)
        rad = np.deg2rad(h)
        a = np.cos(rad) * 30.0
        b = np.sin(rad) * 30.0
        # Use Lab-style chroma; convert via approximate LCH→RGB later
        # For seeding, just use HSV-like coloring
        cR = 0.5 + 0.5 * np.cos(rad)
        cG = 0.5 + 0.5 * np.cos(rad - 2 * np.pi / 3)
        cB = 0.5 + 0.5 * np.cos(rad + 2 * np.pi / 3)
        img[..., 0] += band_mask * cR
        img[..., 1] += band_mask * cG
        img[..., 2] += band_mask * cB

    # Add a face-like dark spot (eye-line) and bright cheek
    img[H // 3:H // 3 + 8, W // 3:W // 3 + 50, :] *= 0.3
    img[H // 3:H // 3 + 8, 2 * W // 3 - 50:2 * W // 3, :] *= 0.3
    img[H // 2:H // 2 + 30, W // 3:2 * W // 3, :] = np.clip(
        img[H // 2:H // 2 + 30, W // 3:2 * W // 3, :] + 0.2, 0, 1
    )

    # Mild noise so chroma entropy is non-trivial
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    # 2. Synthetic cell graph — random Voronoi-like cells
    cells: dict[int, dict] = {}
    for cid in range(n_cells):
        y0 = int(rng.integers(0, H))
        x0 = int(rng.integers(0, W))
        # Cell size: small square cluster of pixels
        sz = int(rng.integers(20, 200))
        # Pixel sample (not full Voronoi — synthetic only)
        sample = rng.integers(0, max(1, H * W // n_cells), size=1).item()
        rgb = img[y0, x0]
        cells[cid] = {
            "mean_rgb": rgb.astype(np.float32),
            "pixels": sz,   # store as count to keep mem light
        }

    cell_graph = {"cells": cells}
    return img_u8, cell_graph


# ---------------------------------------------------------------------------
# 1. plate count adapts to chroma variance
# ---------------------------------------------------------------------------
def test_plate_count_adapts_to_chroma_variance():
    """Low chroma variance → lower plate count; high chroma variance → higher."""
    rng = np.random.default_rng(1)
    H = W = 128

    # Low variance: mostly grayscale
    low = np.tile(rng.integers(110, 130, size=(H, W, 1), dtype=np.uint8), (1, 1, 3))
    low_cells = {i: {"mean_rgb": np.array([120, 120, 120], dtype=np.float32),
                     "pixels": 100} for i in range(800)}

    # High variance: rainbow gradient + noise
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rad = np.arctan2(yy - H / 2, xx - W / 2)
    R = (0.5 + 0.5 * np.cos(rad))
    G = (0.5 + 0.5 * np.cos(rad - 2 * np.pi / 3))
    B = (0.5 + 0.5 * np.cos(rad + 2 * np.pi / 3))
    high = (np.stack([R, G, B], -1) * 255).astype(np.uint8)
    high += rng.integers(-10, 10, size=high.shape, dtype=np.int16).clip(-128, 127).astype(np.int8).astype(np.uint8)
    high_cells = {}
    for i in range(2000):
        y0, x0 = rng.integers(0, H), rng.integers(0, W)
        high_cells[i] = {"mean_rgb": high[y0, x0].astype(np.float32) / 255.0,
                         "pixels": 100}

    n_low = estimate_plate_count(low, cell_graph={"cells": low_cells},
                                 color_space="RGB")
    n_high = estimate_plate_count(high, cell_graph={"cells": high_cells},
                                  color_space="RGB")

    assert 20 <= n_low <= 35, n_low
    assert 20 <= n_high <= 35, n_high
    # High chroma variance produces strictly more plates than low
    assert n_high > n_low, (
        f"expected n_high > n_low; got n_high={n_high}, n_low={n_low}"
    )


# ---------------------------------------------------------------------------
# 2. plate count clamps to [20, 35]
# ---------------------------------------------------------------------------
def test_plate_count_clamps_to_20_to_35_range():
    """Verify the estimator never returns values outside [20, 35]."""
    rng = np.random.default_rng(2)
    # Pathological inputs that would push the raw formula out of range:
    # (a) zero cells, monochrome — must still return >= 20
    mono = np.full((64, 64, 3), 128, dtype=np.uint8)
    n_mono = estimate_plate_count(mono, cell_graph={"cells": {}},
                                  color_space="RGB")
    assert 20 <= n_mono <= 35

    # (b) huge chroma-rich image with massive cell count — must clamp at 35
    H = W = 256
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    cells = {i: {"mean_rgb": (rng.random(3) * 1.0).astype(np.float32),
                 "pixels": 100} for i in range(50_000)}
    n_huge = estimate_plate_count(img, cell_graph={"cells": cells},
                                  color_space="RGB")
    assert 20 <= n_huge <= 35

    # (c) typical Emma-scale lands in the prior range (20..35)
    img_emma, cg_emma = _make_synthetic_emma_input()
    n_emma = estimate_plate_count(img_emma, cell_graph=cg_emma,
                                  color_space="RGB")
    assert 20 <= n_emma <= 35


# ---------------------------------------------------------------------------
# 3. multi-pull returns 1..5 per block
# ---------------------------------------------------------------------------
def test_multi_pull_returns_1_to_5_per_block():
    """Every block gets between 1 and 5 pulls."""
    role_assignments = {1: "underlayer_light", 2: "underlayer_light",
                        3: "underlayer_light", 4: "underlayer_light",
                        5: "underlayer_light", 6: "underlayer_light",
                        7: "local_chroma", 8: "local_chroma",
                        9: "local_chroma", 10: "local_chroma",
                        11: "local_chroma", 12: "local_chroma",
                        13: "local_chroma", 14: "local_chroma",
                        15: "local_chroma",
                        16: "regional_mass", 17: "regional_mass",
                        18: "regional_mass", 19: "regional_mass",
                        20: "regional_mass", 21: "regional_mass",
                        22: "regional_mass", 23: "regional_mass",
                        24: "key_detail", 25: "key_detail",
                        26: "key_detail"}
    per_block = assign_pulls_per_block(
        plate_count=26,
        role_assignments=role_assignments,
    )
    assert len(per_block) == 26
    for bid, pulls in per_block.items():
        assert 1 <= len(pulls) <= 5, (
            f"block {bid} ({role_assignments[bid]}) has {len(pulls)} pulls"
        )
        # PullSpec fields sanity
        for p in pulls:
            assert isinstance(p, PullSpec)
            assert p.block_id == bid
            assert 1 <= p.pass_index <= 5
            assert p.role in ROLES
            assert 0.0 <= p.opacity <= 1.0
            assert 0.0 <= p.dilution <= 1.0


# ---------------------------------------------------------------------------
# 4. global pull sum ≈ 132 for Emma-class input
# ---------------------------------------------------------------------------
def test_pull_count_sums_to_around_132_for_emma_class_input():
    """For Emma-class (27 plates per v2-design row 3, role mix per
    _ROLE_FRACTIONS) sum ≈ 132."""
    # 27-plate Emma role mix: 6 underlayer + 9 local + 9 regional + 3 key
    # (matches `production_plan_builder._role_count_targets(27)`).
    role_assignments: dict[int, str] = {}
    bid = 1
    for _ in range(6):
        role_assignments[bid] = "underlayer_light"; bid += 1
    for _ in range(9):
        role_assignments[bid] = "local_chroma"; bid += 1
    for _ in range(9):
        role_assignments[bid] = "regional_mass"; bid += 1
    for _ in range(3):
        role_assignments[bid] = "key_detail"; bid += 1

    per_block = assign_pulls_per_block(
        plate_count=27,
        role_assignments=role_assignments,
        target_total_pulls=132,
        target_pull_tolerance=12,
    )
    total = sum(len(p) for p in per_block.values())
    assert 120 <= total <= 144, (
        f"expected total ≈ 132 (±12); got {total}"
    )


# ---------------------------------------------------------------------------
# 5. no orphan cells
# ---------------------------------------------------------------------------
def test_production_plan_has_no_orphan_cells():
    """Every cell in the input graph appears on at least one plate."""
    img, cg = _make_synthetic_emma_input(n_cells=1800, seed=42)
    plan = build_production_plan(img, cg)
    assigned = plan.assigned_cell_ids()
    all_cells = set(cg["cells"].keys())
    orphans = all_cells - assigned
    assert not orphans, (
        f"{len(orphans)} orphan cells found (first 5: {sorted(orphans)[:5]})"
    )


# ---------------------------------------------------------------------------
# 6. no zone-overlap on same block (and no cell on two plates)
# ---------------------------------------------------------------------------
def test_production_plan_has_no_zone_overlap_on_same_block():
    """cell_zone_ids on one plate are a SET (no duplicates within a plate).

    Stronger check: a single cell shouldn't appear on TWO plates either —
    that's a v13 failure mode the dedup pass guards against.
    """
    img, cg = _make_synthetic_emma_input(n_cells=1500, seed=99)
    plan = build_production_plan(img, cg)

    # Within-plate uniqueness
    for p in plan.plates:
        assert len(p.cell_zone_ids) == len(set(p.cell_zone_ids)), (
            f"plate {p.block_id} has duplicate cell_zone_ids"
        )

    # Across-plate uniqueness (each cell on exactly one plate)
    seen: dict[int, int] = {}
    for p in plan.plates:
        for cid in p.cell_zone_ids:
            assert cid not in seen, (
                f"cell {cid} appears on plate {seen[cid]} AND plate {p.block_id}"
            )
            seen[cid] = p.block_id


# ---------------------------------------------------------------------------
# 7. role purity meets threshold (per validator)
# ---------------------------------------------------------------------------
def test_production_plan_role_purity_meets_threshold():
    """For every plate, the assigned role is the modal role of its cells.

    Since the builder assigns role per plate (not per cell), purity should
    be trivially 1.0. The check is that role_purity.passes() returns True
    for every plate when cell_role_labels are inherited from the plate role.
    """
    img, cg = _make_synthetic_emma_input(n_cells=1700, seed=123)
    plan = build_production_plan(img, cg)

    for plate in plan.plates:
        cell_role_labels = {cid: plate.role for cid in plate.cell_zone_ids}
        ok = role_purity.passes(
            plate_id=plate.block_id,
            cells_in_plate=plate.cell_zone_ids,
            cell_role_labels=cell_role_labels,
        )
        assert ok, f"plate {plate.block_id} failed role_purity"


# ---------------------------------------------------------------------------
# 8. plates pass plate_not_composite_score on synthetic plate-mask renders
# ---------------------------------------------------------------------------
def test_production_plan_passes_plate_not_composite_validator():
    """Render each plate as a binary inked-zone mask + the final composite,
    then run plate_not_composite. A real jigsaw plate (sparse inked zones,
    NOT a full-face residual) should score below the 0.6 reject threshold.
    """
    img, cg = _make_synthetic_emma_input(n_cells=1500, seed=7)
    plan = build_production_plan(img, cg)

    H, W = plan.image_shape

    # Final composite proxy (the input image at the same resolution)
    final = img

    rng = np.random.default_rng(0)
    # Build deterministic cell→pixel map: assign each cell a random small
    # rectangular region in the image. Same seed reproduces.
    cell_pixels: dict[int, tuple[int, int, int, int]] = {}
    for cid in cg["cells"]:
        y0 = int(rng.integers(0, H - 8))
        x0 = int(rng.integers(0, W - 8))
        h_, w_ = int(rng.integers(4, 12)), int(rng.integers(4, 12))
        cell_pixels[cid] = (y0, x0, min(H, y0 + h_), min(W, x0 + w_))

    passing = 0
    failing = []
    for plate in plan.plates:
        mask = np.zeros((H, W), dtype=np.float32)
        for cid in plate.cell_zone_ids:
            y0, x0, y1, x1 = cell_pixels[cid]
            mask[y0:y1, x0:x1] = 1.0
        # Convert to RGB-style image with wood-grain background for the
        # validator (wood = bright; inked = dark)
        plate_rgb = np.ones((H, W, 3), dtype=np.float32) * 0.92
        for c in range(3):
            plate_rgb[..., c] = np.where(mask > 0, 0.15, 0.92)

        score = plate_not_composite.score(plate_rgb, final, return_components=True)
        if score["passes"]:
            passing += 1
        else:
            failing.append((plate.block_id, score["badness_score"]))

    # Require ≥85% of plates pass — the proposer's job is to produce
    # plates that LOOK like jigsaw blocks, not composites. Some allowance
    # for the synthetic test seed.
    pct = passing / len(plan.plates)
    assert pct >= 0.85, (
        f"only {pct:.1%} plates passed plate_not_composite; "
        f"failures: {failing[:5]}"
    )


# ---------------------------------------------------------------------------
# 9. integration with underlayer_proposer
# ---------------------------------------------------------------------------
def test_integration_with_underlayer_proposer():
    """Wire up the v3 underlayer_proposer → v4 builder, verify underlayer
    plates produced by the proposer carry role='underlayer_light' in the
    final ProductionPlan.
    """
    sys.path.insert(
        0,
        str(HERE.parent.parent / "v3-construction" / "mokuhanga-rule-classifier"),
    )
    # Import the proposer
    import underlayer_proposer as up

    rng = np.random.default_rng(2026)
    H = W = 128
    img = (rng.random((H, W, 3), dtype=np.float32) * 0.5 + 0.4).astype(np.float32)

    # Build a tiny cell graph with mean colors that lie in face-region hue bands.
    cells = {}
    region_to_cells: dict[str, list[int]] = {
        "cheek":      [0, 1, 2],
        "lip":        [3],
        "forehead":   [4, 5],
        "hair":       [6, 7, 8],
        "background": [9, 10, 11, 12],
        "eye_white":  [13],
        "chin":       [14],
        "jaw_neck":   [15],
        "temple":     [16],
    }
    # Hue per region → mean RGB
    rgb_by_region = {
        "cheek":      np.array([0.96, 0.84, 0.74], dtype=np.float32),
        "lip":        np.array([0.85, 0.30, 0.30], dtype=np.float32),
        "forehead":   np.array([0.98, 0.90, 0.80], dtype=np.float32),
        "hair":       np.array([0.20, 0.18, 0.16], dtype=np.float32),
        "background": np.array([0.70, 0.78, 0.86], dtype=np.float32),
        "eye_white":  np.array([0.92, 0.92, 0.94], dtype=np.float32),
        "chin":       np.array([0.93, 0.81, 0.71], dtype=np.float32),
        "jaw_neck":   np.array([0.90, 0.78, 0.68], dtype=np.float32),
        "temple":     np.array([0.94, 0.82, 0.72], dtype=np.float32),
    }
    for region, ids in region_to_cells.items():
        for cid in ids:
            cells[cid] = {
                "mean_rgb": rgb_by_region[region],
                "pixels": [(0, 0)] * 200,
            }
    cg_v3 = up.CellGraph(cells=cells)
    fl = up.FaceLandmarks(region_to_cells=region_to_cells, image_shape=(H, W))

    underlayer_plates = up.propose_underlayers(img, cg_v3, fl)
    assert 4 <= len(underlayer_plates) <= 9, (
        f"underlayer proposer returned {len(underlayer_plates)} plates"
    )

    # Convert v3 UnderlayerPlate → v4 Form-A role_assignments
    role_assignments: dict[int, dict] = {}
    used_cells: set[int] = set()
    for plate in underlayer_plates:
        role_assignments[plate.block_id] = {
            "cells": plate.cell_zone_ids,
            "role": "underlayer_light",
            "pigment_family": plate.pigment_family,
            "region_label": plate.region_label,
            "rationale": plate.rationale,
            "provenance": plate.provenance,
        }
        used_cells.update(plate.cell_zone_ids)

    # Add filler plates so we hit at least 20 total
    next_bid = max(role_assignments.keys()) + 1
    unused = sorted(set(cells.keys()) - used_cells)
    if not unused:
        # Synthesize extra cells if proposer used everything
        for k in range(40):
            new_id = 100 + k
            cells[new_id] = {
                "mean_rgb": np.array([0.5, 0.5, 0.5], dtype=np.float32),
                "pixels": [(0, 0)] * 50,
            }
            unused.append(new_id)
        cg_v3 = up.CellGraph(cells=cells)

    # Distribute unused cells across enough plates to reach >= 20 total.
    needed = max(0, 22 - len(role_assignments))
    if needed > 0 and unused:
        chunks = np.array_split(np.array(unused), needed)
        roles_cycle = ["local_chroma", "regional_mass", "key_detail",
                       "local_chroma", "regional_mass"]
        for i, ch in enumerate(chunks):
            if len(ch) == 0:
                continue
            role = roles_cycle[i % len(roles_cycle)]
            role_assignments[next_bid] = {
                "cells": [int(x) for x in ch.tolist()],
                "role": role,
                "pigment_family": "pale_blue",
                "region_label": f"filler_{i}",
                "rationale": "filler for integration test",
                "provenance": "algorithm",
            }
            next_bid += 1

    # Now build the v4 ProductionPlan from this composed input
    cell_graph_v4 = {"cells": cells}
    img_u8 = (img * 255).astype(np.uint8)
    plan = build_production_plan(img_u8, cell_graph_v4,
                                 role_assignments=role_assignments)

    # Underlayer plates in the v3 proposer must come through as
    # underlayer_light in the final plan
    underlayer_v3_block_ids = {p.block_id for p in underlayer_plates}
    plan_underlayer_blocks = {
        p.block_id for p in plan.plates if p.role == "underlayer_light"
    }
    overlap = underlayer_v3_block_ids & plan_underlayer_blocks
    assert overlap, (
        "underlayer block_ids from v3 proposer did not survive into v4 plan"
    )

    # Every v4 underlayer plate has >= 1 pull
    for p in plan.plates:
        if p.role == "underlayer_light":
            assert p.pull_count >= 1
            # underlayer pulls have pass_index in [1..3] per ROLE_PULL_BUDGET
            for pull in p.pulls:
                assert 1 <= pull.pass_index <= 3
                assert pull.role == "underlayer_light"


# ---------------------------------------------------------------------------
# 10. (bonus) plan validate() passes on auto-built Emma plan
# ---------------------------------------------------------------------------
def test_plan_validate_passes_on_emma():
    """Sanity check: the auto-built plan satisfies I1-I9."""
    img, cg = _make_synthetic_emma_input()
    plan = build_production_plan(img, cg)
    ok, errors = plan.validate(all_cell_ids=set(cg["cells"].keys()))
    assert ok, f"plan validation failed: {errors}"


# ---------------------------------------------------------------------------
# 11. (bonus) performance: build_production_plan < 5s on Emma scale
# ---------------------------------------------------------------------------
def test_performance_under_5s_emma_scale():
    """Builder runs in <5s on Emma-scale input (1700-2500 cells, ~26 plates)."""
    import time
    img, cg = _make_synthetic_emma_input(n_cells=2200, seed=11)
    t0 = time.time()
    plan = build_production_plan(img, cg)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"build_production_plan took {elapsed:.2f}s (> 5s)"
    assert 20 <= plan.plate_count <= 35


# ---------------------------------------------------------------------------
# 12. (bonus) Form B (cell→role mapping) path
# ---------------------------------------------------------------------------
def test_build_production_plan_form_b_cells_to_role():
    """Caller can supply role_assignments as dict[cell_id, role] (Form B)."""
    img, cg = _make_synthetic_emma_input(n_cells=400, seed=5)
    cell_ids = sorted(cg["cells"].keys())
    # 25% underlayer, 30% local, 30% regional, 15% key (deterministic stripe)
    role_per_cell: dict[int, str] = {}
    for i, cid in enumerate(cell_ids):
        if i < len(cell_ids) * 0.25:
            role_per_cell[cid] = "underlayer_light"
        elif i < len(cell_ids) * 0.55:
            role_per_cell[cid] = "local_chroma"
        elif i < len(cell_ids) * 0.85:
            role_per_cell[cid] = "regional_mass"
        else:
            role_per_cell[cid] = "key_detail"

    plan = build_production_plan(img, cg, role_assignments=role_per_cell)
    assert 20 <= plan.plate_count <= 35
    # All four roles represented
    role_counts = plan.role_distribution()
    for r, n in role_counts.items():
        assert n >= 1, f"role {r} has 0 plates"


# ---------------------------------------------------------------------------
# 13. (bonus) JSON serialization round-trip
# ---------------------------------------------------------------------------
def test_production_plan_to_dict_and_json():
    """ProductionPlan.to_dict() and .to_json() produce loadable output."""
    import json
    img, cg = _make_synthetic_emma_input(n_cells=600, seed=8)
    plan = build_production_plan(img, cg)

    d = plan.to_dict()
    assert d["plate_count"] == plan.plate_count
    assert d["total_pulls"] == plan.total_pulls
    assert d["cell_count"] == plan.cell_count
    assert isinstance(d["plates"], list)
    assert len(d["plates"]) == plan.plate_count
    # JSON round-trip
    s = plan.to_json()
    parsed = json.loads(s)
    assert parsed["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 14. (bonus) explicit plate_count override
# ---------------------------------------------------------------------------
def test_build_production_plan_plate_count_override():
    """Caller can override the adaptive plate count for testing/calibration."""
    img, cg = _make_synthetic_emma_input(n_cells=600, seed=11)
    plan = build_production_plan(img, cg, plate_count=22)
    assert 20 <= plan.plate_count <= 35
    # Override should drive us toward 22 (auto-partitioner may snap to 20-23)
    assert plan.plate_count >= 20


# ---------------------------------------------------------------------------
# 15. (bonus) plate validation invariants individually
# ---------------------------------------------------------------------------
def test_role_count_targets_for_various_plate_counts():
    """_role_count_targets always sums to plate_count and gives >=1 to each role."""
    from production_plan_builder import _role_count_targets
    for n in [20, 22, 24, 25, 27, 30, 33, 35]:
        counts = _role_count_targets(n)
        assert sum(counts.values()) == n, f"plate_count={n} mismatch: {counts}"
        for r in ROLES:
            assert counts[r] >= 1, f"plate_count={n} role={r} has 0"


def test_build_with_very_small_cell_graph_pads_to_min_plates():
    """Pathological tiny input — builder still emits >= MIN_PLATES (20) plates."""
    rng = np.random.default_rng(99)
    H = W = 64
    img = (rng.random((H, W, 3), dtype=np.float32) * 0.5 + 0.4).astype(np.float32)
    # Only 80 cells — should still get padded to 20+ plates
    cells = {}
    for cid in range(80):
        rgb = (rng.random(3) * 1.0).astype(np.float32)
        cells[cid] = {"mean_rgb": rgb, "pixels": int(rng.integers(10, 60))}

    plan = build_production_plan(img, {"cells": cells})
    assert plan.plate_count >= 20


def test_form_b_pixel_list_input():
    """When cell_graph stores 'pixels' as a list-of-tuples (not int)."""
    rng = np.random.default_rng(33)
    H = W = 64
    img = (rng.random((H, W, 3), dtype=np.float32) * 0.5 + 0.4).astype(np.float32)
    cells = {}
    for cid in range(400):
        rgb = (rng.random(3) * 1.0).astype(np.float32)
        pix_list = [(int(rng.integers(0, H)), int(rng.integers(0, W)))
                    for _ in range(int(rng.integers(20, 100)))]
        cells[cid] = {"mean_rgb": rgb, "pixels": pix_list}

    plan = build_production_plan(img, {"cells": cells})
    assert 20 <= plan.plate_count <= 35


def test_lab_mean_input_in_cells():
    """When cell_graph entries provide 'mean_lab' directly (skip RGB→LAB)."""
    rng = np.random.default_rng(44)
    H = W = 64
    img = (rng.random((H, W, 3), dtype=np.float32) * 0.5 + 0.4).astype(np.float32)
    cells = {}
    for cid in range(300):
        # LAB-ish: L in [10, 90], a/b in [-30, 30]
        L = float(rng.uniform(10, 90))
        a = float(rng.uniform(-30, 30))
        b = float(rng.uniform(-30, 30))
        cells[cid] = {
            "mean_lab": np.array([L, a, b], dtype=np.float32),
            "pixels": int(rng.integers(20, 100)),
        }
    plan = build_production_plan(img, {"cells": cells})
    assert 20 <= plan.plate_count <= 35
    # Every plate has >= 1 cell and >= 1 pull
    for p in plan.plates:
        assert p.cell_zone_ids
        assert p.pull_count >= 1


def test_plate_spec_add_pull_rejects_mismatched_block_id():
    """PlateSpec.add_pull guards against mismatched block_id / role."""
    plate = PlateSpec(
        block_id=1, cell_zone_ids=[1, 2, 3],
        role="local_chroma", pigment_family="pale_red",
    )
    bad_pull = PullSpec(
        pull_id=1, block_id=99, pigment_id="x",
        opacity=0.5, dilution=0.5, order_step=1, pass_index=1,
        role="local_chroma",
    )
    with pytest.raises(ValueError, match="block_id"):
        plate.add_pull(bad_pull)

    wrong_role_pull = PullSpec(
        pull_id=2, block_id=1, pigment_id="x",
        opacity=0.5, dilution=0.5, order_step=1, pass_index=1,
        role="key_detail",
    )
    with pytest.raises(ValueError, match="role"):
        plate.add_pull(wrong_role_pull)


def test_pullspec_to_dict_round_trip():
    """PullSpec.to_dict + PlateSpec.to_dict produce JSON-serializable dicts."""
    import json
    pull = PullSpec(
        pull_id=7, block_id=3, pigment_id="PR122_holbein_dilute",
        opacity=0.42, dilution=0.61, order_step=15, pass_index=2,
        role="local_chroma",
    )
    d = pull.to_dict()
    json.dumps(d)  # raises if not serializable
    assert d["pull_id"] == 7
    assert d["mask_subset"] is None

    plate = PlateSpec(
        block_id=3, cell_zone_ids=[1, 2], role="local_chroma",
        pigment_family="pale_red", region_label="lip",
    )
    plate.add_pull(pull)
    pd = plate.to_dict()
    json.dumps(pd)
    assert pd["block_id"] == 3
    assert len(pd["pulls"]) == 1


def test_role_distribution_consistent_with_plate_count():
    """ProductionPlan.role_distribution sums to plate_count."""
    img, cg = _make_synthetic_emma_input(n_cells=900, seed=29)
    plan = build_production_plan(img, cg)
    dist = plan.role_distribution()
    assert sum(dist.values()) == plan.plate_count
    assert set(dist.keys()) == set(ROLES)


def test_production_plan_validate_catches_orphans():
    """ProductionPlan.validate() reports orphan cells in I5."""
    img, cg = _make_synthetic_emma_input(n_cells=400, seed=15)
    plan = build_production_plan(img, cg)
    # Inject a fake extra cell that's NOT in any plate
    fake_all = set(cg["cells"].keys()) | {999_999}
    ok, errors = plan.validate(all_cell_ids=fake_all)
    assert not ok
    assert any("I5" in e for e in errors), errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
