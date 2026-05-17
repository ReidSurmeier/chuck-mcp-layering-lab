"""One-shot script: build a synthetic Emma plan, score it against the
plate_not_composite and role_purity validators, dump a sample JSON.

Outputs to ./_artifacts/ for NOTES.md.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "v3-construction"))

import numpy as np

from production_plan_builder import build_production_plan

V3 = HERE.parent.parent / "v3-construction"
_PNC_PATH = V3 / "validators-reconstruction" / "plate_not_composite.py"
_RP_PATH = V3 / "validators-reconstruction" / "role_purity.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


plate_not_composite = _load("plate_not_composite", _PNC_PATH)
role_purity = _load("role_purity", _RP_PATH)


def _make_emma(H=256, W=256, n_cells=1800, seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rn = r / r.max()
    n_clusters = 7
    base_hues = np.linspace(0, 360, n_clusters, endpoint=False)
    for i, h in enumerate(base_hues):
        m = ((rn >= i / n_clusters) & (rn < (i + 1) / n_clusters)).astype(np.float32)
        rad = np.deg2rad(h)
        img[..., 0] += m * (0.5 + 0.5 * np.cos(rad))
        img[..., 1] += m * (0.5 + 0.5 * np.cos(rad - 2 * np.pi / 3))
        img[..., 2] += m * (0.5 + 0.5 * np.cos(rad + 2 * np.pi / 3))
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    cells = {}
    for cid in range(n_cells):
        y0 = int(rng.integers(0, H))
        x0 = int(rng.integers(0, W))
        cells[cid] = {
            "mean_rgb": img[y0, x0].astype(np.float32),
            "pixels": int(rng.integers(20, 200)),
        }
    return img_u8, {"cells": cells}


def main():
    out_dir = HERE / "_artifacts"
    out_dir.mkdir(exist_ok=True)

    print("=== Emma-class synthetic build ===")
    img, cg = _make_emma()
    t0 = time.time()
    plan = build_production_plan(img, cg)
    elapsed = time.time() - t0
    print(f"plate_count    : {plan.plate_count}")
    print(f"total_pulls    : {plan.total_pulls}")
    print(f"cell_count     : {plan.cell_count}")
    print(f"role_dist      : {plan.role_distribution()}")
    print(f"build_seconds  : {elapsed:.3f}")
    print(f"validate_ok    : {plan.meta.get('validate_ok')}")

    # Save sample plan JSON
    sample_json = out_dir / "sample_emma_plan.json"
    sample_json.write_text(plan.to_json(indent=2))
    print(f"sample plan written: {sample_json}")

    # ---------------- plate_not_composite over all plates ----------------
    H, W = plan.image_shape
    rng = np.random.default_rng(0)
    cell_pixels = {}
    for cid in cg["cells"]:
        y0 = int(rng.integers(0, H - 8))
        x0 = int(rng.integers(0, W - 8))
        cell_pixels[cid] = (y0, x0, min(H, y0 + 8), min(W, x0 + 8))

    plate_scores = []
    for plate in plan.plates:
        mask = np.zeros((H, W), dtype=np.float32)
        for cid in plate.cell_zone_ids:
            y0, x0, y1, x1 = cell_pixels[cid]
            mask[y0:y1, x0:x1] = 1.0
        plate_rgb = np.where(
            mask[..., None] > 0,
            np.array([0.15, 0.15, 0.15]),
            np.array([0.92, 0.92, 0.92]),
        ).astype(np.float32)
        s = plate_not_composite.score(plate_rgb, img, return_components=True)
        plate_scores.append(
            {
                "block_id": plate.block_id,
                "role": plate.role,
                "badness": float(s["badness_score"]),
                "passes": bool(s["passes"]),
                "cosine_similarity": float(s["cosine_similarity"]),
                "inked_area_fraction": float(s["inked_area_fraction"]),
            }
        )

    n_pass = sum(1 for s in plate_scores if s["passes"])
    n_total = len(plate_scores)
    print(
        f"plate_not_composite: {n_pass}/{n_total} pass "
        f"({n_pass / n_total:.1%})"
    )

    # ---------------- role_purity over all plates ----------------
    role_scores = []
    for plate in plan.plates:
        labels = {cid: plate.role for cid in plate.cell_zone_ids}
        r = role_purity.score(
            plate_id=plate.block_id,
            cells_in_plate=plate.cell_zone_ids,
            cell_role_labels=labels,
            return_components=True,
        )
        role_scores.append(
            {
                "block_id": plate.block_id,
                "role": plate.role,
                "purity": float(r["purity_score"]),
                "passes": bool(r["passes"]),
                "n_cells": int(r["n_cells"]),
            }
        )

    n_rp_pass = sum(1 for s in role_scores if s["passes"])
    print(
        f"role_purity        : {n_rp_pass}/{n_total} pass "
        f"({n_rp_pass / n_total:.1%})"
    )

    # ---------------- Validator report ----------------
    report = {
        "plan_id": plan.plan_id,
        "build_seconds": elapsed,
        "plate_count": plan.plate_count,
        "total_pulls": plan.total_pulls,
        "cell_count": plan.cell_count,
        "role_distribution": plan.role_distribution(),
        "plate_not_composite_pass_rate": n_pass / n_total,
        "role_purity_pass_rate": n_rp_pass / n_total,
        "plate_scores_summary": {
            "min_badness": min(s["badness"] for s in plate_scores),
            "max_badness": max(s["badness"] for s in plate_scores),
            "mean_badness": sum(s["badness"] for s in plate_scores) / n_total,
            "n_pass": n_pass,
            "n_total": n_total,
        },
        "role_scores_summary": {
            "min_purity": min(s["purity"] for s in role_scores),
            "max_purity": max(s["purity"] for s in role_scores),
            "mean_purity": sum(s["purity"] for s in role_scores) / n_total,
            "n_pass": n_rp_pass,
            "n_total": n_total,
        },
        "first_5_plate_scores": plate_scores[:5],
        "first_5_role_scores": role_scores[:5],
    }
    report_path = out_dir / "validator_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"validator report written: {report_path}")

    # Bench: re-run with 2200 cells for performance benchmark
    print("\n=== Performance benchmark (2200 cells) ===")
    img_b, cg_b = _make_emma(n_cells=2200, seed=11)
    t0 = time.time()
    plan_b = build_production_plan(img_b, cg_b)
    bench = time.time() - t0
    print(f"build_production_plan: {bench:.3f}s")
    print(f"plate_count: {plan_b.plate_count}, total_pulls: {plan_b.total_pulls}")

    perf_path = out_dir / "performance_bench.json"
    perf_path.write_text(
        json.dumps(
            {
                "emma_class": {
                    "n_cells": 2200,
                    "build_seconds": bench,
                    "plate_count": plan_b.plate_count,
                    "total_pulls": plan_b.total_pulls,
                    "under_5s_target": bench < 5.0,
                },
                "default_test": {
                    "n_cells": 1800,
                    "build_seconds": elapsed,
                    "plate_count": plan.plate_count,
                    "total_pulls": plan.total_pulls,
                },
            },
            indent=2,
        )
    )
    print(f"performance bench written: {perf_path}")


if __name__ == "__main__":
    main()
