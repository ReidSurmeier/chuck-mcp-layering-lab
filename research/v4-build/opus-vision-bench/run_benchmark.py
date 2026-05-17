"""
run_benchmark.py — end-to-end runner for the Opus 4.7 vision cell-ID bench.

For each of 10 ground-truth overlays:
  1. load ground_truth_regions.json
  2. call Opus via opus_cell_id_extractor (one claude -p invocation)
  3. compare predictions vs ground truth -> per-region Jaccard / F1
On completion:
  4. aggregate -> overall Jaccard
  5. compute routing decision
  6. write bench_results.json + bench_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fallback_router import RoutingDecision, route_to_opus_or_mediapipe
from generate_ground_truth import GroundTruthEntry, generate_dataset
from jaccard_evaluator import BenchResult, ImageScore, aggregate, compare
from opus_cell_id_extractor import (
    ExtractionResult,
    OpusExtractionError,
    extract_cell_ids_from_overlay,
)


def _load_ground_truth(entry: GroundTruthEntry) -> tuple[dict[str, list[int]], int]:
    payload = json.loads(entry.regions_path.read_text())
    return payload["regions"], int(payload["n_cells"])


def run_benchmark(
    *,
    out_dir: Path,
    region_names: list[str] | None = None,
    timeout_s: int = 240,
    max_retries: int = 1,
    force_ground_truth: bool = False,
    dry_run: bool = False,
) -> tuple[BenchResult, RoutingDecision, dict]:
    """Execute the full benchmark and write artifacts.

    dry_run: if True, skip the Opus calls and use ground-truth-as-prediction
    (perfect score) for the structural smoke test in CI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "ground_truth"
    print(f"# building ground truth in {gt_dir}", file=sys.stderr)
    entries = generate_dataset(gt_dir, force=force_ground_truth)

    image_scores: list[ImageScore] = []
    extraction_records: list[dict] = []
    skipped: list[dict] = []
    cost_total = 0.0
    durations_ms: list[int] = []

    for entry in entries:
        gt_map, n_cells = _load_ground_truth(entry)
        regions = sorted(set(region_names or gt_map.keys()))

        print(f"\n--> {entry.image_id}  cells={n_cells}  "
              f"regions={len(regions)}", file=sys.stderr)

        if dry_run:
            # Smoke-test path — skip Opus, use ground truth as prediction.
            preds = {r: list(gt_map.get(r, [])) for r in regions}
            extract: ExtractionResult | None = None
        else:
            try:
                extract = extract_cell_ids_from_overlay(
                    entry.overlay_path,
                    regions,
                    image_id=entry.image_id,
                    max_cell_id=max(n_cells - 1, 0),
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                )
                preds = extract.predictions
            except (OpusExtractionError, FileNotFoundError) as exc:
                print(f"    SKIP {entry.image_id}: {exc}", file=sys.stderr)
                skipped.append({"image_id": entry.image_id, "error": str(exc)})
                continue

        score = compare(preds, gt_map, image_id=entry.image_id, regions=regions)
        image_scores.append(score)

        if extract is not None:
            cost_total += extract.cost_usd
            durations_ms.append(extract.duration_ms)

        extraction_records.append({
            "image_id": entry.image_id,
            "n_cells": n_cells,
            "n_regions": len(regions),
            "overlay_path": str(entry.overlay_path),
            "mean_jaccard": score.mean_jaccard(),
            "median_jaccard": score.median_jaccard(),
            "min_jaccard": score.min_jaccard(),
            "mean_f1": score.mean_f1(),
            "cost_usd": extract.cost_usd if extract else 0.0,
            "duration_ms": extract.duration_ms if extract else 0,
            "session_id": extract.session_id if extract else None,
        })
        print(
            f"    jaccard mean={score.mean_jaccard():.3f}  "
            f"min={score.min_jaccard():.3f}  "
            f"f1={score.mean_f1():.3f}", file=sys.stderr)

    bench = aggregate(image_scores)
    decision = route_to_opus_or_mediapipe(bench)

    durations_summary = {
        "median_ms": statistics.median(durations_ms) if durations_ms else 0,
        "min_ms": min(durations_ms) if durations_ms else 0,
        "max_ms": max(durations_ms) if durations_ms else 0,
        "n": len(durations_ms),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "n_images_scored": len(image_scores),
        "n_images_skipped": len(skipped),
        "skipped": skipped,
        "cost_usd_total": cost_total,
        "latency_summary_ms": durations_summary,
        "extraction_records": extraction_records,
        "bench": bench.to_dict(),
        "routing_decision": decision.to_dict(),
    }

    (out_dir / "bench_results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "bench_report.md").write_text(_render_report(payload))
    return bench, decision, payload


def _render_report(payload: dict) -> str:
    bench = payload["bench"]
    routing = payload["routing_decision"]
    lines: list[str] = []
    lines.append("# Opus 4.7 vision cell-ID benchmark — report")
    lines.append("")
    lines.append(f"_generated: {payload['generated_at']}_")
    lines.append(f"_dry-run: {payload['dry_run']}_")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- images scored: {payload['n_images_scored']} / 10")
    lines.append(f"- images skipped: {payload['n_images_skipped']}")
    lines.append(f"- **overall mean Jaccard: {bench['overall_mean_jaccard']:.3f}**")
    lines.append(f"- overall median Jaccard: {bench['overall_median_jaccard']:.3f}")
    lines.append(f"- overall min Jaccard: {bench['overall_min_jaccard']:.3f}")
    lines.append(f"- overall mean F1: {bench['overall_mean_f1']:.3f}")
    lines.append(
        f"- cost spent: ${payload['cost_usd_total']:.4f} "
        f"over {payload['n_images_scored']} Opus calls"
    )
    lat = payload["latency_summary_ms"]
    lines.append(
        f"- latency (ms): median={lat['median_ms']}  "
        f"min={lat['min_ms']}  max={lat['max_ms']}"
    )
    lines.append("")
    lines.append("## Routing decision")
    lines.append("")
    lines.append(
        f"**Global route: `{routing['global_route'].upper()}`** "
        f"(threshold {routing['global_threshold']})"
    )
    lines.append("")
    lines.append(f"> {routing['reason']}")
    lines.append("")
    lines.append("Per-region routes (floor "
                 f"{routing['per_region_floor']}):")
    lines.append("")
    if not routing["per_region_route"]:
        lines.append("- (no per-region results)")
    else:
        for region, route in sorted(routing["per_region_route"].items()):
            stats = bench["per_region_summary"].get(region, {})
            mean = stats.get("mean", 0.0)
            n = stats.get("n", 0)
            lines.append(
                f"- `{region:18s}` -> `{route:9s}`  "
                f"mean Jaccard {mean:.3f}  (n={n})"
            )
    lines.append("")
    lines.append("## Per-image headline")
    lines.append("")
    lines.append("| image_id | mean Jaccard | min | F1 | cost USD | duration ms |")
    lines.append("|---|---|---|---|---|---|")
    for rec in payload["extraction_records"]:
        lines.append(
            f"| {rec['image_id']} | {rec['mean_jaccard']:.3f} | "
            f"{rec['min_jaccard']:.3f} | {rec['mean_f1']:.3f} | "
            f"{rec['cost_usd']:.4f} | {rec['duration_ms']} |"
        )
    if payload["skipped"]:
        lines.append("")
        lines.append("### Skipped")
        for s in payload["skipped"]:
            lines.append(f"- {s['image_id']}: {s['error']}")
    lines.append("")
    lines.append("## Per-region detail")
    lines.append("")
    lines.append("| region | mean Jaccard | median | min | max | n |")
    lines.append("|---|---|---|---|---|---|")
    summary = bench["per_region_summary"]
    for region in sorted(summary):
        s = summary[region]
        lines.append(
            f"| {region} | {s['mean']:.3f} | {s['median']:.3f} | "
            f"{s['min']:.3f} | {s['max']:.3f} | {s['n']} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(_HERE / "bench_output"),
        help="output directory for bench_results.json + bench_report.md",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=240,
        help="per-overlay subprocess timeout",
    )
    parser.add_argument(
        "--retries", type=int, default=1,
        help="extra retries per Opus call",
    )
    parser.add_argument(
        "--regions", default="",
        help="comma-separated region subset (default = ground-truth keys)",
    )
    parser.add_argument(
        "--force-gt", action="store_true",
        help="regenerate ground truth even if cached",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="skip Opus calls; use ground truth as prediction (smoke test)",
    )
    args = parser.parse_args(argv)

    region_names = ([r.strip() for r in args.regions.split(",") if r.strip()]
                    if args.regions else None)
    out_dir = Path(args.out)

    t0 = time.time()
    _, decision, payload = run_benchmark(
        out_dir=out_dir,
        region_names=region_names,
        timeout_s=args.timeout_s,
        max_retries=args.retries,
        force_ground_truth=args.force_gt,
        dry_run=args.dry_run,
    )
    wall = time.time() - t0

    print(
        f"\n# done in {wall:.1f}s  ->  global_route={decision.global_route}  "
        f"mean_jaccard={payload['bench']['overall_mean_jaccard']:.3f}  "
        f"cost=${payload['cost_usd_total']:.4f}"
    )
    return 0 if decision.global_route == "opus" else 1


if __name__ == "__main__":
    raise SystemExit(main())
