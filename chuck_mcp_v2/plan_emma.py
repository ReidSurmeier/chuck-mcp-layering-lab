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

    return {
        "output": str(output),
        "plan_id": production_plan.plan_id,
        "plate_count": len(result.plates),
        "outer_iter_count": result.outer_iter_count,
        "n_gates_passed": result.n_gates_passed(),
        "delta_e_mean": result.delta_e_mean,
        "converged": result.converged,
        "artifacts_dir": args.artifacts_dir,
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
