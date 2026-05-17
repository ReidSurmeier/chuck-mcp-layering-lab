"""CLI entry point: `python -m acceptance_harness <plan_dir> [--output sheet.png]`.

When this package is invoked as a module (after sys.path includes the parent
folder) this script is what runs. Designed to be wrapped by an MCP tool later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .acceptance_harness import render_acceptance_sheet
from .example_loader import REFERENCE_EXAMPLES_DIR


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acceptance_harness",
        description=(
            "Render the chuck-mcp v4 4-row acceptance contact sheet "
            "(reference proofs / current proofs / current plates / alpha maps)."
        ),
    )
    p.add_argument(
        "plan_dir",
        type=Path,
        help="path to a chuck-mcp plan output directory (e.g. .../2026-05-17_v3-audit-thorough-main)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="output PNG path (default: <plan_dir>/acceptance_sheet.png)",
    )
    p.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_EXAMPLES_DIR,
        help=f"reference examples directory (default: {REFERENCE_EXAMPLES_DIR})",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit AcceptanceSheetResult as JSON to stdout instead of human text",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = render_acceptance_sheet(
            plan_output_dir=args.plan_dir,
            reference_examples_dir=args.reference_dir,
            output_path=args.output,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    print(f"acceptance sheet: {result.sheet_path}")
    print(f"  proof checkpoints rendered: {result.proof_checkpoints_rendered}")
    print(f"  plates rendered: {result.plate_count_rendered}")
    print(f"  alpha maps rendered: {result.alpha_count_rendered}")
    print(f"  proof progression score: {result.proof_progression_score:.4f}")
    print("  plate metrics (idx | coverage | plate_not_composite):")
    for pm in result.plate_metrics:
        print(
            f"    {pm.plate_index:>2}  cov={pm.coverage_fraction:.3f}  "
            f"pnc={pm.plate_not_composite_score:.3f}"
        )
    if result.warnings:
        print("  warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    print("\nHUMAN EYEBALL REQUIRED: open the PNG and judge against the reference row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
