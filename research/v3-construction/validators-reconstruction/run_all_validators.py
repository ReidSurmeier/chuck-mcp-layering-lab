"""Master validator runner — runs all 6 validators against a plan
and emits a unified JSON report compatible with the web app
verification UI.

Plan input shape (dict, kwargs-style):

    plan = {
        "plan_id": "emma-2026-05-16-v3",
        "target_image": "/path/to/close_emma_2002_2048.jpg",
        "final_composite": "/path/to/final_composite.png",
        "plates": [
            {
                "block_id": 1,
                "plate_preview": "/path/to/block_01.png"   # rendered plate
                "plate_svg": "/path/to/block_01.svg",      # for reversal check
                "pull_preview": "/path/to/pull_001.png",   # cumulative after this pull
                "cells_in_plate": [12, 34, 56, ...],       # cell ids on this plate
                "role": "underlayer_light",                # validated against allowed
                "dpi": 300,                                # physical resolution
                # OPTIONAL — for zone-distance check on rendered mask:
                "inked_mask": numpy 2-D array,
            },
            ...27 entries...
        ],
        "cell_role_labels": {1: "underlayer_light", 2: "key_detail", ...},
        "cell_pixel_positions": {1: (y, x), ...},
        "cell_adjacency": {1: [2, 3], 2: [1, 4], ...},
        "proof_states": [
            "/path/to/proof_00.png",    # blank / first plate
            "/path/to/proof_01.png",
            ...
            "/path/to/proof_06.png",    # final
        ],
        "visibility_mask": optional numpy bool array for final_match,
    }

Output JSON shape (also written to disk if `output_path` given):

    {
        "plan_id": "...",
        "summary": {
            "passes_overall": bool,
            "n_gates_passed": int,    # out of 5 gating validators
            "n_gates_total": 5,
            "advisory_score": dict,   # final_match details
            "elapsed_ms": float,
        },
        "validators": {
            "plate_not_composite": {
                "per_plate": [{block_id, score, passes, ...}, ...],
                "aggregate": {n_pass, n_fail, worst_block_id, worst_score},
                "passes": bool,
            },
            "role_purity":           {...},
            "jigsaw_separation":     {...},
            "proof_progression":     {...},
            "underlayer_reversal":   {...},
            "final_match":           {...},   # advisory
        },
    }
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Support both package-style and direct-module imports (the directory
# contains a hyphen so package-style only works through a symlink).
try:
    from . import (
        final_match,
        jigsaw_separation,
        plate_not_composite,
        proof_progression,
        role_purity,
        underlayer_reversal_check,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import final_match  # type: ignore
    import jigsaw_separation  # type: ignore
    import plate_not_composite  # type: ignore
    import proof_progression  # type: ignore
    import role_purity  # type: ignore
    import underlayer_reversal_check  # type: ignore


def _safe_call(fn, *args, **kwargs):
    """Wrap a validator call to return error info instead of crashing."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"passes": False, "error": f"{type(e).__name__}: {e}"}


def run_all_validators(
    plan: dict,
    output_path: Optional[str] = None,
) -> dict:
    """Run all 6 validators against a plan and return unified JSON.

    See module docstring for plan shape.
    """
    t0 = time.time()
    report: dict[str, Any] = {
        "plan_id": plan.get("plan_id", "unknown"),
        "summary": {},
        "validators": {},
    }

    plates = plan.get("plates", [])
    target = plan.get("target_image")
    final_composite = plan.get("final_composite")
    role_labels = plan.get("cell_role_labels", {})
    cell_positions = plan.get("cell_pixel_positions", {})
    cell_adjacency = plan.get("cell_adjacency", {})
    proof_states = plan.get("proof_states", [])
    visibility = plan.get("visibility_mask")

    # ---- Validator 1: plate_not_composite ----
    pnc_results = []
    if final_composite is None:
        report["validators"]["plate_not_composite"] = {
            "passes": False,
            "error": "no final_composite supplied",
        }
    else:
        # Cache final composite as numpy once -- saves ~250ms/plate of repeated decode
        try:
            from PIL import Image as _PIL
            if isinstance(final_composite, np.ndarray):
                _final_cached = final_composite
            else:
                _final_cached = np.asarray(_PIL.open(final_composite).convert("RGB"))
        except Exception:
            _final_cached = final_composite

        for p in plates:
            block_id = p.get("block_id")
            plate_img = p.get("plate_preview")
            if plate_img is None:
                pnc_results.append({"block_id": block_id, "error": "no plate_preview", "passes": False})
                continue
            r = _safe_call(
                plate_not_composite.score,
                plate_img, _final_cached, return_components=True,
            )
            r["block_id"] = block_id
            pnc_results.append(r)
        n_pass = sum(1 for r in pnc_results if r.get("passes"))
        worst = max(pnc_results, key=lambda x: x.get("badness_score", 0.0), default=None)
        report["validators"]["plate_not_composite"] = {
            "per_plate": pnc_results,
            "aggregate": {
                "n_pass": n_pass,
                "n_fail": len(pnc_results) - n_pass,
                "worst_block_id": worst.get("block_id") if worst else None,
                "worst_score": worst.get("badness_score") if worst else None,
                "reject_threshold": plate_not_composite.REJECT_THRESHOLD,
            },
            "passes": n_pass == len(pnc_results) and len(pnc_results) > 0,
        }

    # ---- Validator 2: role_purity ----
    rp_results = []
    for p in plates:
        block_id = p.get("block_id")
        cells = p.get("cells_in_plate", [])
        # Allow per-plate override of role labels
        labels = p.get("cell_role_labels_override", role_labels)
        r = _safe_call(role_purity.score, block_id, cells, labels, return_components=True)
        if "block_id" not in r:
            r["block_id"] = block_id
        rp_results.append(r)
    n_pass = sum(1 for r in rp_results if r.get("passes"))
    report["validators"]["role_purity"] = {
        "per_plate": rp_results,
        "aggregate": {
            "n_pass": n_pass,
            "n_fail": len(rp_results) - n_pass,
            "purity_threshold": role_purity.PURITY_THRESHOLD,
        },
        "passes": n_pass == len(rp_results) and len(rp_results) > 0,
    }

    # ---- Validator 3: jigsaw_separation ----
    js_results = []
    for p in plates:
        block_id = p.get("block_id")
        dpi = p.get("dpi", jigsaw_separation.DEFAULT_DPI)
        mask = p.get("inked_mask")
        if mask is not None:
            r = _safe_call(
                jigsaw_separation.score_from_mask, mask, dpi, return_components=True
            )
        else:
            r = _safe_call(
                jigsaw_separation.score,
                cells_in_plate=p.get("cells_in_plate", []),
                cell_pixel_positions=cell_positions,
                adjacency=cell_adjacency,
                dpi=dpi,
                return_components=True,
            )
        r["block_id"] = block_id
        js_results.append(r)
    n_pass = sum(1 for r in js_results if r.get("passes"))
    report["validators"]["jigsaw_separation"] = {
        "per_plate": js_results,
        "aggregate": {
            "n_pass": n_pass,
            "n_fail": len(js_results) - n_pass,
            "min_threshold_mm": jigsaw_separation.MIN_SEPARATION_MM,
        },
        "passes": n_pass == len(js_results) and len(js_results) > 0,
    }

    # ---- Validator 4: proof_progression ----
    if len(proof_states) < 2:
        report["validators"]["proof_progression"] = {
            "passes": False,
            "error": f"need >= 2 proof states, got {len(proof_states)}",
        }
    else:
        r = _safe_call(proof_progression.score, proof_states, return_components=True)
        report["validators"]["proof_progression"] = r

    # ---- Validator 5: underlayer_reversal ----
    # Cache target once to avoid re-decoding per plate
    _target_cached = None
    if target is not None:
        try:
            from PIL import Image as _PIL
            if isinstance(target, np.ndarray):
                _target_cached = target
            else:
                _target_cached = np.asarray(_PIL.open(target).convert("RGB"))
        except Exception:
            _target_cached = target

    ur_results = []
    for p in plates:
        block_id = p.get("block_id")
        svg = p.get("plate_svg")
        pull = p.get("pull_preview")
        if svg is None:
            ur_results.append({"block_id": block_id, "error": "no plate_svg", "passes": False})
            continue
        r = _safe_call(
            underlayer_reversal_check.check,
            svg, pull, _target_cached, return_components=True,
        )
        r["block_id"] = block_id
        ur_results.append(r)
    n_pass = sum(1 for r in ur_results if r.get("passes"))
    report["validators"]["underlayer_reversal"] = {
        "per_plate": ur_results,
        "aggregate": {
            "n_pass": n_pass,
            "n_fail": len(ur_results) - n_pass,
        },
        "passes": n_pass == len(ur_results) and len(ur_results) > 0,
    }

    # ---- Validator 6: final_match (ADVISORY) ----
    if target is None or final_composite is None:
        report["validators"]["final_match"] = {
            "passes": True,
            "advisory_only": True,
            "error": "no target or final_composite",
        }
    else:
        fm = _safe_call(
            final_match.score, target, final_composite,
            visibility_mask=visibility,
        )
        report["validators"]["final_match"] = fm

    # ---- Summary ----
    gating = [
        "plate_not_composite",
        "role_purity",
        "jigsaw_separation",
        "proof_progression",
        "underlayer_reversal",
    ]
    n_pass = sum(1 for k in gating if report["validators"].get(k, {}).get("passes"))
    elapsed_ms = (time.time() - t0) * 1000.0
    report["summary"] = {
        "passes_overall": n_pass == len(gating),
        "n_gates_passed": n_pass,
        "n_gates_total": len(gating),
        "advisory_score": report["validators"].get("final_match", {}),
        "elapsed_ms": float(elapsed_ms),
    }

    if output_path:
        # Convert numpy types for JSON
        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return str(o)
        Path(output_path).write_text(json.dumps(report, indent=2, default=_default))

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_json", help="path to plan JSON file")
    ap.add_argument("--output", "-o", help="write report JSON here")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan_json).read_text())
    out = run_all_validators(plan, output_path=args.output)
    print(json.dumps(out["summary"], indent=2))
