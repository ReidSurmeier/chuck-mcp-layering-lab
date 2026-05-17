#!/usr/bin/env python3
"""CLI wrapper for validator-plan construction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validator_plan import ValidatorPlanInputs, build_validator_plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-result", required=True, type=Path)
    ap.add_argument("--production-plan", required=True, type=Path)
    ap.add_argument("--artifacts-dir", required=True, type=Path)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    plan = build_validator_plan(
        ValidatorPlanInputs(
            hybrid_result=args.hybrid_result,
            production_plan=args.production_plan,
            artifacts_dir=args.artifacts_dir,
            job_dir=args.job_dir,
            input_image=args.input_image,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2))
    print(
        f"wrote plan with {len(plan['plates'])} plates, "
        f"{len(plan['proof_states'])} proofs to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
