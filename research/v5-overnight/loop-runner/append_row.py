#!/usr/bin/env python3
"""Append one row to iterations.csv summarizing this iteration."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", required=True)
    ap.add_argument("--wall", required=True, type=int)
    ap.add_argument("--hybrid-result", required=True, type=Path)
    ap.add_argument("--validator-report", required=True, type=Path)
    ap.add_argument("--underlayer-match", required=True, type=Path)
    ap.add_argument("--sheet", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    hyb = _read_json(args.hybrid_result)
    vr = _read_json(args.validator_report)
    ulm = _read_json(args.underlayer_match)

    plate_count = int(hyb.get("plate_count") or len(hyb.get("plates") or []))
    dE_mean = float(hyb.get("delta_e_mean") or 0.0)
    dE_p95 = float(hyb.get("delta_e_p95") or 0.0)

    # PNC: from validator_report (run_all_validators output)
    pnc = (vr.get("validators") or {}).get("plate_not_composite") or {}
    pnc_agg = pnc.get("aggregate") or {}
    plates_pass_pnc = int(pnc_agg.get("n_pass", 0))

    # validators_passed: count gating validators only (out of 5). final_match
    # is advisory and is already represented by dE_mean / dE_p95 columns.
    gates = ("plate_not_composite", "role_purity", "jigsaw_separation",
             "proof_progression", "underlayer_reversal")
    vmap = vr.get("validators") or {}
    validators_passed = sum(1 for g in gates if (vmap.get(g) or {}).get("passes"))

    underlayer_match_pct = float(ulm.get("match_pct") or 0.0)

    row = {
        "iter_n": args.iter,
        "wall_s": args.wall,
        "plate_count": plate_count,
        "plates_pass_pnc": plates_pass_pnc,
        "dE_mean": round(dE_mean, 4),
        "dE_p95": round(dE_p95, 4),
        "validators_passed": validators_passed,
        "underlayer_match_pct": round(underlayer_match_pct, 2),
        "sheet_path": str(args.sheet),
        "notes": args.notes,
    }

    write_header = not args.csv.exists() or args.csv.stat().st_size == 0
    fields = list(row.keys())
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
