#!/usr/bin/env python3
"""Build a plan-dict for run_all_validators from chuck_mcp_v2.plan_emma outputs.

The validator expects (per its module docstring):
    plan = {
        "plan_id", "target_image", "final_composite",
        "plates": [{block_id, plate_preview, plate_svg, pull_preview,
                    cells_in_plate, role, dpi, ...}],
        "cell_role_labels", "cell_pixel_positions", "cell_adjacency",
        "proof_states": [...],
    }

We pull what we can from:
- hybrid_result.json  (validator_scores live, plate list, dE)
- production_plan.json (plate -> cells, roles, pulls)
- artifacts/ (cumulative_pull_NN.png, plates/, alpha_masks/, final_composite.png)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-result", required=True, type=Path)
    ap.add_argument("--production-plan", required=True, type=Path)
    ap.add_argument("--artifacts-dir", required=True, type=Path)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    plan: dict = {
        "plan_id": f"v5-overnight-{args.job_dir.name}",
        "target_image": str(args.input_image),
        "plates": [],
        "cell_role_labels": {},
        "cell_pixel_positions": {},
        "cell_adjacency": {},
        "proof_states": [],
    }

    # production plan (cells/roles)
    if args.production_plan.exists():
        try:
            prod = json.loads(args.production_plan.read_text())
        except Exception as e:
            print(f"WARN production_plan parse: {e}", file=sys.stderr)
            prod = {}
        for plate in prod.get("plates", []):
            block_id = int(plate.get("block_id", 0))
            cells = list(plate.get("cell_zone_ids", []))
            role = plate.get("role", "regional_mass")
            plan["plates"].append({
                "block_id": block_id,
                "cells_in_plate": cells,
                "role": role,
                "dpi": 300,
            })

    # hybrid result -> may have authoritative plates
    if args.hybrid_result.exists():
        try:
            hyb = json.loads(args.hybrid_result.read_text())
        except Exception as e:
            print(f"WARN hybrid_result parse: {e}", file=sys.stderr)
            hyb = {}
        # rebuild plate index from hybrid plates
        plates_by_id = {int(p["block_id"]): p for p in plan["plates"]}
        for plate in hyb.get("plates", []):
            bid = int(plate.get("block_id", 0))
            entry = plates_by_id.get(bid)
            if entry is None:
                entry = {
                    "block_id": bid,
                    "cells_in_plate": list(plate.get("cell_zone_ids", [])),
                    "role": plate.get("role", "regional_mass"),
                    "dpi": 300,
                }
                plan["plates"].append(entry)
                plates_by_id[bid] = entry
            # may have updated cell list
            if plate.get("cell_zone_ids"):
                entry["cells_in_plate"] = list(plate["cell_zone_ids"])
            if plate.get("role"):
                entry["role"] = plate["role"]

    # role labels per cell (merge from plates)
    role_labels: dict = {}
    for p in plan["plates"]:
        for cid in p.get("cells_in_plate", []):
            role_labels[int(cid)] = p["role"]
    plan["cell_role_labels"] = role_labels

    # artifacts: look for per-plate previews, alphas, cumulative proofs, final composite
    art = args.artifacts_dir
    plates_dir = art / "plates"
    if plates_dir.is_dir():
        for p in plan["plates"]:
            bid = p["block_id"]
            for pat in (f"block_{bid:02d}.preview.png", f"plate_{bid:02d}.preview.png",
                        f"block_{bid}.preview.png", f"plate_{bid}.preview.png"):
                cand = plates_dir / pat
                if cand.exists():
                    p["plate_preview"] = str(cand)
                    break
            for pat in (f"block_{bid:02d}.svg", f"plate_{bid:02d}.svg"):
                cand = plates_dir / pat
                if cand.exists():
                    p["plate_svg"] = str(cand)
                    break

    alpha_dir = art / "alpha_masks"
    if alpha_dir.is_dir():
        for p in plan["plates"]:
            bid = p["block_id"]
            for pat in (f"alpha_{bid:02d}.png", f"alpha_{bid}.png"):
                cand = alpha_dir / pat
                if cand.exists():
                    p["alpha_preview"] = str(cand)
                    break

    proofs = sorted([str(x) for x in art.glob("cumulative_pull_*.png")]) \
        or sorted([str(x) for x in art.glob("pull_*.png")]) \
        or sorted([str(x) for x in art.glob("proof_*.png")])
    plan["proof_states"] = proofs
    # also attach pull_preview to each plate (cumulative AFTER pull N)
    for i, p in enumerate(plan["plates"]):
        if i < len(proofs):
            p["pull_preview"] = proofs[i]

    fc = art / "final_composite.png"
    if fc.exists():
        plan["final_composite"] = str(fc)
    elif proofs:
        plan["final_composite"] = proofs[-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2))
    print(f"wrote plan with {len(plan['plates'])} plates, {len(plan['proof_states'])} proofs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
