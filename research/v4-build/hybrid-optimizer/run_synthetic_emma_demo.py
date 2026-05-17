"""Synthetic Emma-scale optimize() demo + metric capture.

Runs the hybrid optimizer on a synthetic 24-plate plan at 256x256, then
serializes the validator scores, wall time, and convergence stats so they
can be quoted in NOTES.md and persisted to the v4-build memory namespace.

Usage:
    python run_synthetic_emma_demo.py [--out /path/to/result.json]

The script does NOT touch real Emma data — it uses the same synthetic plan
fixture from test_hybrid_optimizer.py. Real-data integration happens in
backend/services/v23 via the plan_emma_print MCP tool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Same alias trick the tests use
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import importlib.util as _ilu  # noqa: E402

_pkg_spec = _ilu.spec_from_file_location(
    "hybrid_optimizer", HERE / "__init__.py", submodule_search_locations=[str(HERE)]
)
_pkg = _ilu.module_from_spec(_pkg_spec)
sys.modules["hybrid_optimizer"] = _pkg
for _name in (
    "optimization_result",
    "graph_cut_assigner",
    "morphology_repair",
    "jax_continuous_solve",
    "alternating_loop",
):
    _s = _ilu.spec_from_file_location(
        f"hybrid_optimizer.{_name}", HERE / f"{_name}.py"
    )
    _m = _ilu.module_from_spec(_s)
    sys.modules[f"hybrid_optimizer.{_name}"] = _m
    _s.loader.exec_module(_m)
_pkg_spec.loader.exec_module(_pkg)

from hybrid_optimizer import optimize  # noqa: E402

# Reuse synthetic fixture
sys.path.insert(0, str(HERE))
from test_hybrid_optimizer import _full_synthetic_plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=120)
    ap.add_argument("--per-role", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-outer", type=int, default=3)
    ap.add_argument("--max-inner", type=int, default=60)
    ap.add_argument(
        "--out",
        default=str(HERE / "synthetic_emma_result.json"),
    )
    args = ap.parse_args()

    print(
        f"[demo] building synthetic plan: n_cells={args.n_cells}, "
        f"per_role={args.per_role}, size={args.size}x{args.size}"
    )
    target, plan = _full_synthetic_plan(
        n_cells=args.n_cells,
        per_role=args.per_role,
        H=args.size,
        W=args.size,
    )

    print(f"[demo] candidate plates: {len(plan.candidate_plates)}")
    t0 = time.time()
    result = optimize(
        target,
        plan,
        max_outer_iters=args.max_outer,
        max_inner_iters=args.max_inner,
        early_stop_on_gates=True,
    )
    wall = time.time() - t0

    summary = {
        "wall_time_s": wall,
        "outer_iter_count": result.outer_iter_count,
        "n_plates": len(result.plates),
        "n_gates_passed": result.n_gates_passed(),
        "converged": result.converged,
        "delta_e_mean": result.delta_e_mean,
        "delta_e_p95": result.delta_e_p95,
        "stage_timings": result.stage_timings,
        "validator_passes": {
            k: bool((result.validator_scores.get(k) or {}).get("passes"))
            for k in (
                "plate_not_composite",
                "role_purity",
                "jigsaw_separation",
                "proof_progression",
                "underlayer_reversal",
                "final_match",
            )
        },
        "notes": result.notes,
    }

    print("\n[demo] SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n[demo] writing full result to {args.out}")
    out = Path(args.out)
    result.save(str(out))
    print(f"[demo] wrote {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
