"""End-to-end v4 Emma planning CLI.

Usage:
    python -m chuck_mcp_v2.plan_emma --synthetic --output /tmp/emma.json
    python -m chuck_mcp_v2.plan_emma path/to/image.png --output /tmp/emma.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
V4_BUILD = REPO_ROOT / "research" / "v4-build"
HYBRID_DIR = V4_BUILD / "hybrid-optimizer"
PRODUCTION_DIR = V4_BUILD / "production-solver"
V5_OVERNIGHT = REPO_ROOT / "research" / "v5-overnight"
V5_MEDIAPIPE_DIR = V5_OVERNIGHT / "mediapipe-spatial"
V5_MOKUHANGA_PIGMENTS_DIR = V5_OVERNIGHT / "mokuhanga-pigments"
V5_SNIC_REAL_DIR = V5_OVERNIGHT / "snic-real"
V5_ALPHA_PROOF_DUMPER_DIR = V5_OVERNIGHT / "alpha-proof-dumper"
V3_MOKUHANGA_RULE_DIR = (
    REPO_ROOT / "research" / "v3-construction" / "mokuhanga-rule-classifier"
)

# Make the v5 mediapipe-spatial modules importable. The folder name contains
# a hyphen so it cannot be a regular Python package; we splice the directory
# onto sys.path and import the modules by their bare names.
if str(V5_MEDIAPIPE_DIR) not in sys.path:
    sys.path.insert(0, str(V5_MEDIAPIPE_DIR))

# Same trick for the mokuhanga pigment adapter, its v3 rule classifier dep,
# and the v5 SNIC-real superpixel proposer.
for _p in (V5_MOKUHANGA_PIGMENTS_DIR, V3_MOKUHANGA_RULE_DIR, V5_SNIC_REAL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_alpha_proof_dumper() -> Any | None:
    """Lazy-load the v5 alpha-proof-dumper package (its dir name has a hyphen,
    so it can't be a regular Python package). Returns the module's
    `dump_run_artifacts` callable, or None if the package is missing.
    """
    alias = "alpha_proof_dumper"
    if alias in sys.modules:
        return getattr(sys.modules[alias], "dump_run_artifacts", None) or getattr(
            sys.modules.get(f"{alias}.dumper"), "dump_run_artifacts", None
        )
    pkg_init = V5_ALPHA_PROOF_DUMPER_DIR / "__init__.py"
    dumper_mod = V5_ALPHA_PROOF_DUMPER_DIR / "dumper.py"
    if not pkg_init.exists() or not dumper_mod.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        alias,
        pkg_init,
        submodule_search_locations=[str(V5_ALPHA_PROOF_DUMPER_DIR)],
    )
    if spec is None or spec.loader is None:
        return None
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[alias] = pkg
    spec.loader.exec_module(pkg)
    dumper_spec = importlib.util.spec_from_file_location(
        f"{alias}.dumper", dumper_mod
    )
    if dumper_spec is None or dumper_spec.loader is None:
        return None
    dumper = importlib.util.module_from_spec(dumper_spec)
    sys.modules[f"{alias}.dumper"] = dumper
    dumper_spec.loader.exec_module(dumper)
    return getattr(dumper, "dump_run_artifacts", None)


def _load_package(alias: str, package_dir: Path) -> Any:
    if alias in sys.modules:
        return sys.modules[alias]
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        alias,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load package {alias} from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_target(path: str | None, *, synthetic: bool, size: int) -> np.ndarray:
    if synthetic or path is None:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        r = (0.58 + 0.35 * np.sin(yy / 8.0)) * 255.0
        g = (0.60 + 0.30 * np.cos(xx / 11.0)) * 255.0
        b = (0.45 + 0.25 * np.sin((yy - xx) / 13.0)) * 255.0
        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _grid_cell_graph(
    target_rgb: np.ndarray,
    requested_cells: int,
) -> tuple[dict, list[tuple[int, int]], dict[tuple[int, int], float]]:
    h, w = target_rgb.shape[:2]
    rows = max(1, int(round(requested_cells ** 0.5)))
    cols = max(1, int(np.ceil(requested_cells / rows)))
    cells: dict[int, dict[str, Any]] = {}
    grid: dict[tuple[int, int], int] = {}
    cid = 0

    for row in range(rows):
        y0 = int(round(row * h / rows))
        y1 = int(round((row + 1) * h / rows))
        for col in range(cols):
            x0 = int(round(col * w / cols))
            x1 = int(round((col + 1) * w / cols))
            tile = target_rgb[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            cells[cid] = {
                "mean_rgb": tile.reshape(-1, 3).mean(axis=0).astype(np.float32),
                "pixels": int(tile.shape[0] * tile.shape[1]),
                "centroid_yx": ((y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0),
                # Pixel bounds — needed by `_grid_cell_label_image` to rebuild a
                # SNIC-style per-pixel label map that lines up with the face
                # region masks for the v5 spatial-constraint step.
                "bounds_yxyx": (int(y0), int(x0), int(y1), int(x1)),
            }
            grid[(row, col)] = cid
            cid += 1

    edges: list[tuple[int, int]] = []
    edge_weights: dict[tuple[int, int], float] = {}
    for (row, col), a in grid.items():
        for dr, dc in ((1, 0), (0, 1)):
            b = grid.get((row + dr, col + dc))
            if b is None:
                continue
            edges.append((a, b))
            ca = np.asarray(cells[a]["mean_rgb"], dtype=np.float32)
            cb = np.asarray(cells[b]["mean_rgb"], dtype=np.float32)
            dist = float(np.linalg.norm(ca - cb))
            edge_weights[(min(a, b), max(a, b))] = float(
                np.clip(255.0 / (dist + 1.0), 0.1, 10.0)
            )

    return {"cells": cells}, edges, edge_weights


def _plan_to_hybrid_input(
    target_rgb: np.ndarray,
    cell_graph_dict: dict,
    edges: list[tuple[int, int]],
    edge_weights: dict[tuple[int, int], float],
    production_plan: Any,
    hybrid: Any,
) -> Any:
    from hybrid_optimizer.jax_continuous_solve import _rgb_to_lab_np

    cell_to_role: dict[int, str] = {}
    for plate in production_plan.plates:
        for cid in plate.cell_zone_ids:
            cell_to_role[int(cid)] = plate.role

    nodes = []
    for cid, cell in sorted(cell_graph_dict["cells"].items()):
        rgb = np.asarray(cell["mean_rgb"], dtype=np.float32)
        lab = _rgb_to_lab_np((rgb / 255.0).reshape(1, 1, 3)).reshape(3)
        nodes.append(
            hybrid.CellNode(
                cell_id=int(cid),
                role=cell_to_role.get(int(cid), "regional_mass"),
                lab_color=(float(lab[0]), float(lab[1]), float(lab[2])),
                area_px=int(cell["pixels"]),
                centroid_yx=tuple(cell["centroid_yx"]),
            )
        )

    candidate_plates = []
    pull_order: dict[int, int] = {}
    for plate in production_plan.plates:
        first_pull = (
            min(plate.pulls, key=lambda p: p.order_step) if plate.pulls else None
        )
        candidate_plates.append(
            hybrid.CandidatePlate(
                plate_id=int(plate.block_id),
                role=plate.role,
                max_area_px=None,
                pigment_id=(first_pull.pigment_id if first_pull else plate.pigment_family),
            )
        )
        pull_order[int(plate.block_id)] = int(
            first_pull.order_step if first_pull else plate.block_id
        )

    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    return hybrid.ProductionPlanInput(
        cell_graph=hybrid.CellGraph(nodes=nodes, edges=edges, edge_weights=edge_weights),
        candidate_plates=candidate_plates,
        pull_order=pull_order,
        target_image_rgb=target_rgb,
        target_shape=target_rgb.shape[:2],
        cell_role_labels=cell_to_role,
        cell_adjacency=adjacency,
        cell_pixel_positions={
            int(cid): tuple(cell["centroid_yx"])
            for cid, cell in cell_graph_dict["cells"].items()
        },
        dpi=300.0,
    )


def _grid_cell_label_image(
    cell_graph_dict: dict, image_shape: tuple[int, int]
) -> np.ndarray:
    """Build a (H, W) int32 label image where every pixel holds its grid
    cell's id. Inverse of `_grid_cell_graph` — uses the bounds we record on
    each cell entry. Used to feed the v5 mediapipe spatial-constraint step a
    SNIC-style label map without re-running superpixel segmentation.
    """
    H, W = image_shape
    labels = np.full((H, W), fill_value=-1, dtype=np.int32)
    for cid, cell in cell_graph_dict["cells"].items():
        bounds = cell.get("bounds_yxyx")
        if bounds is None:
            # Fall back to centroid: paint a 1-pixel marker so the cell at
            # least exists in np.unique(labels) downstream.
            cy, cx = cell["centroid_yx"]
            iy, ix = int(round(cy)), int(round(cx))
            iy = max(0, min(H - 1, iy))
            ix = max(0, min(W - 1, ix))
            labels[iy, ix] = int(cid)
            continue
        y0, x0, y1, x1 = bounds
        labels[y0:y1, x0:x1] = int(cid)
    return labels


def _build_face_region_constraint(
    plate: Any,
    cell_to_region: dict[int, str],
    *,
    max_regions: int = 3,
) -> list[str]:
    """Pick the 1..`max_regions` regions that cover the most cells on this
    plate. Falls back to `["background"]` if the plate has no cells in any
    face region (defensive — caller filters first).
    """
    from collections import Counter
    counter: Counter = Counter()
    for cid in plate.cell_zone_ids:
        region = cell_to_region.get(int(cid))
        if region is None:
            continue
        counter[region] += 1
    if not counter:
        return ["background"]
    return [name for name, _ in counter.most_common(max_regions)]


def apply_face_region_constraints(
    plan: Any,
    target_rgb: np.ndarray,
    cell_graph_dict: dict,
    *,
    image_path: str | None = None,
    constrained_roles: tuple[str, ...] = ("underlayer_light",),
    max_regions_per_plate: int = 3,
) -> Any:
    """Tag every plate in `constrained_roles` with `face_region_constraint`
    drawn from the v5 MediaPipe + Chuck Close blur cascade pipeline, and
    drop cells that don't satisfy that constraint.

    This is the v5 patch the seam-fix run was missing — without it, the
    grid placeholder lets hair-area cells onto skin-tone underlayer plates,
    producing structurally wrong woodblock plans.

    Args:
        plan: ProductionPlan (mutated in place; also returned)
        target_rgb: the solver-space image (H, W, 3) — already resized by
            `_load_target`. Used only to get the SNIC label image shape.
        cell_graph_dict: output of `_grid_cell_graph`.
        image_path: ORIGINAL full-resolution image path. MediaPipe runs on
            this image — full res is needed to clear the Chuck Close
            σ=21 blur threshold. When None, the entire step is skipped and
            the plan is returned unchanged.
        constrained_roles: which plate roles receive constraints. Defaults to
            underlayer plates only (the seam-fix mission scope).
        max_regions_per_plate: cap on the constraint list length per plate.
    """
    if not image_path:
        return plan

    # Import the v5 helpers lazily so this module can be imported in
    # environments without mediapipe (synthetic-only runs).
    import face_spatial
    import merge_cells_with_regions as mcr
    import region_constrained_plate as rcp
    import cv2 as _cv2

    # 1. MediaPipe runs on the ORIGINAL full-res image so the gauss21 cascade
    #    has enough texture to bite on. Then masks get nearest-neighbor
    #    downscaled to the solver-space target shape.
    face_regions_full = face_spatial.extract_face_regions(image_path)
    H, W = target_rgb.shape[:2]
    face_regions = _resize_face_regions(face_regions_full, (H, W))

    # 2. Build a SNIC-style label image from the grid cell graph, at the
    #    same (H, W) as the resized face region masks.
    label_img = _grid_cell_label_image(cell_graph_dict, (H, W))

    # 3. Assign each cell to its primary face region.
    cell_to_region = mcr.assign_cells_to_regions(face_regions, label_img)

    # 4. Per plate in `constrained_roles`: pick the dominant regions, set
    #    face_region_constraint, then filter cells that violate it.
    new_plates = []
    for plate in plan.plates:
        if plate.role not in constrained_roles:
            new_plates.append(plate)
            continue
        # If the plate has no cells, leave it alone (degenerate case from the
        # auto-partitioner; the production_plan validator will flag it).
        if not plate.cell_zone_ids:
            plate.face_region_constraint = ["background"]
            new_plates.append(plate)
            continue
        constraint = _build_face_region_constraint(
            plate, cell_to_region, max_regions=max_regions_per_plate,
        )
        plate.face_region_constraint = constraint
        filtered = rcp.filter_plate_cells(
            plate, cell_to_region, allowed_regions=constraint,
        )
        # If filtering emptied the plate, keep the original cells with a
        # background constraint label rather than producing an invalid
        # empty plate (I2 violation). This is conservative — better to mark
        # the plate as ambiguous than to delete it.
        if not filtered.cell_zone_ids:
            plate.face_region_constraint = ["background"]
            new_plates.append(plate)
            continue
        new_plates.append(filtered)

    plan.plates = new_plates
    # Stamp the plan meta so downstream code knows the v5 spatial step ran.
    plan.meta = dict(plan.meta) if plan.meta else {}
    plan.meta["mediapipe_spatial_applied"] = True
    plan.meta["mediapipe_strategy"] = ",".join(sorted({
        r.source for r in face_regions_full.values()
    }))
    return plan


def _resize_face_regions(face_regions: dict, target_shape: tuple[int, int]) -> dict:
    """Nearest-neighbor resize every FaceRegion mask to `target_shape`, with
    bbox + centroid recomputed from the resized mask. Polygon coords are
    rescaled linearly. Used because MediaPipe runs on the full-resolution
    image while the production plan operates on a downsampled solver-space.
    """
    import face_spatial  # for FaceRegion dataclass
    import cv2 as _cv2

    H, W = target_shape
    out: dict = {}
    for name, region in face_regions.items():
        src_h, src_w = region.mask.shape
        if (src_h, src_w) == (H, W):
            out[name] = region
            continue
        new_mask = _cv2.resize(region.mask, (W, H), interpolation=_cv2.INTER_NEAREST)
        new_poly = None
        if region.polygon is not None:
            sx, sy = W / src_w, H / src_h
            new_poly = [
                (int(round(x * sx)), int(round(y * sy)))
                for (x, y) in region.polygon
            ]
        ys, xs = np.where(new_mask > 0)
        if len(xs) == 0:
            bbox = None
            centroid = None
        else:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            centroid = (float(xs.mean()), float(ys.mean()))
        out[name] = face_spatial.FaceRegion(
            name=region.name, polygon=new_poly, mask=new_mask,
            source=region.source, confidence=region.confidence,
            centroid=centroid, bbox=bbox,
        )
    return out


def apply_mokuhanga_pigments(production_plan: Any, cell_graph_dict: dict) -> dict:
    """v5 mokuhanga-pigments adapter: mutate every plate so it carries a
    concrete pigment from `chuck_mcp_v2/pigment_library_emma.yaml`.

    Wraps `mokuhanga_emma.apply_mokuhanga_pigments_to_plan` with a
    CellGraph adapter built from the grid cell_graph_dict that plan_emma
    emits. Returns the adapter's diagnostic summary dict.
    """
    # Late imports — the v5 mokuhanga-pigments folder is hyphenated so
    # cannot be a real package; sys.path was extended at module load.
    from mokuhanga_emma import apply_mokuhanga_pigments_to_plan  # type: ignore
    from underlayer_proposer import CellGraph  # type: ignore

    cells_view: dict[int, dict] = {}
    for cid, cell in cell_graph_dict["cells"].items():
        pixels_field = cell.get("pixels", 0)
        if isinstance(pixels_field, int):
            # Synthesize a list of fake pixel coords so CellGraph.pixel_count
            # returns the right area; only the length is read downstream.
            px = [(0, 0)] * int(pixels_field)
        else:
            px = list(pixels_field)
        cells_view[int(cid)] = {
            "mean_rgb": np.asarray(cell["mean_rgb"], dtype=np.float32),
            "pixels": px,
        }
    return apply_mokuhanga_pigments_to_plan(
        production_plan,
        CellGraph(cells=cells_view),
        pigment_library=None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    hybrid = _load_package("hybrid_optimizer", HYBRID_DIR)
    production = _load_package("production_solver", PRODUCTION_DIR)

    target_rgb = _load_target(args.image, synthetic=args.synthetic, size=args.size)
    cell_graph_dict, edges, edge_weights = _grid_cell_graph(target_rgb, args.cells)
    production_plan = production.build_production_plan(
        target_rgb,
        cell_graph_dict,
        plate_count=args.plate_count,
        target_total_pulls=args.target_pulls,
        target_pull_tolerance=args.target_pull_tolerance,
    )
    # v5 spatial constraint: tag underlayer plates with their face region(s)
    # and drop cells that don't belong (hair on a skin plate, background on a
    # lip plate, etc.). Skipped for synthetic runs since there is no face to
    # detect.
    if args.image and not args.synthetic and not getattr(args, "no_face_regions", False):
        production_plan = apply_face_region_constraints(
            production_plan, target_rgb, cell_graph_dict,
            image_path=args.image,
        )

    # v5 mokuhanga-pigments: assign every plate a CONCRETE pigment from
    # chuck_mcp_v2/pigment_library_emma.yaml using the v3 rule classifier
    # (94.4% Emma match) for underlayer plates and ΔE_76 nearest-neighbor
    # for mid/dark plates. Skipped only if explicitly disabled.
    if not getattr(args, "no_mokuhanga_pigments", False):
        apply_mokuhanga_pigments(production_plan, cell_graph_dict)
    hybrid_input = _plan_to_hybrid_input(
        target_rgb,
        cell_graph_dict,
        edges,
        edge_weights,
        production_plan,
        hybrid,
    )

    result = hybrid.optimize(
        target_rgb,
        hybrid_input,
        max_outer_iters=args.max_outer_iters,
        max_inner_iters=args.max_inner_iters,
        mill_radius_px=args.mill_radius_px,
        early_stop_on_gates=not args.no_early_stop,
        save_artifacts_dir=args.artifacts_dir,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output))
    if args.plan_output:
        plan_output = Path(args.plan_output)
        plan_output.parent.mkdir(parents=True, exist_ok=True)
        plan_output.write_text(production_plan.to_json())

    # v5 alpha-proof-dumper: emit per-pull alpha + cumulative proof PNGs +
    # mirrored plate previews into args.artifacts_dir so the
    # acceptance_harness can populate rows 2/3/4 of the contact sheet
    # without showing "NOT FOUND" placeholders. Silently skipped when
    # artifacts_dir is None or the dumper package is missing.
    dump_summary: dict[str, int] = {}
    if args.artifacts_dir and not getattr(args, "no_alpha_proof_dump", False):
        dumper_fn = _load_alpha_proof_dumper()
        if dumper_fn is not None:
            try:
                dump_result = dumper_fn(
                    target_rgb=target_rgb,
                    plates=result.plates,
                    out_dir=args.artifacts_dir,
                    checkpoint_count=getattr(args, "proof_checkpoint_count", 7),
                )
                dump_summary = {
                    k: len(v) for k, v in dump_result.items() if isinstance(v, list)
                }
            except Exception as exc:  # pragma: no cover — log and continue
                print(
                    json.dumps(
                        {"alpha_proof_dump_error": repr(exc)}, indent=2
                    ),
                    file=sys.stderr,
                )

    return {
        "output": str(output),
        "plan_id": production_plan.plan_id,
        "plate_count": len(result.plates),
        "outer_iter_count": result.outer_iter_count,
        "n_gates_passed": result.n_gates_passed(),
        "delta_e_mean": result.delta_e_mean,
        "converged": result.converged,
        "artifacts_dir": args.artifacts_dir,
        "alpha_proof_dump": dump_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v4 Emma production/hybrid pipeline."
    )
    parser.add_argument("image", nargs="?", help="Input image. Omit with --synthetic.")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic target image.")
    parser.add_argument(
        "--output",
        default="emma_hybrid_result.json",
        help="Hybrid result JSON path.",
    )
    parser.add_argument("--plan-output", help="Optional pre-solve ProductionPlan JSON path.")
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Max image side length for solve space.",
    )
    parser.add_argument(
        "--cells", type=int, default=64, help="Approximate grid cell count."
    )
    parser.add_argument(
        "--plate-count",
        type=int,
        default=20,
        help="Production-solver plate count.",
    )
    parser.add_argument(
        "--target-pulls",
        type=int,
        default=132,
        help="Production-solver pull target.",
    )
    parser.add_argument("--target-pull-tolerance", type=int, default=12)
    parser.add_argument("--max-outer-iters", type=int, default=1)
    parser.add_argument("--max-inner-iters", type=int, default=10)
    parser.add_argument("--mill-radius-px", type=int, default=2)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        help="Optional directory for per-iteration proof and plate preview PNGs.",
    )
    parser.add_argument(
        "--no-face-regions",
        action="store_true",
        help=(
            "Disable v5 MediaPipe face-region constraint on underlayer plates. "
            "Use for non-portrait inputs or when MediaPipe models are missing."
        ),
    )
    parser.add_argument(
        "--no-alpha-proof-dump",
        action="store_true",
        help=(
            "Disable v5 alpha-proof-dumper. By default, when --artifacts-dir "
            "is set, plan_emma writes per-pull alpha PNGs + cumulative proofs "
            "+ mirrored plate previews + 7 checkpoint proofs so the "
            "acceptance_harness can populate rows 2/3/4 of the contact sheet."
        ),
    )
    parser.add_argument(
        "--proof-checkpoint-count",
        type=int,
        default=7,
        help="How many evenly spaced proof checkpoints to dump (default 7).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.synthetic and not args.image:
        parser.error("provide an image path or use --synthetic")
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
