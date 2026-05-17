"""Test harness for all 6 validators.

This is the SMOKING GUN proof — synthetic plans that pass + the real
v13 contact sheet blocks that MUST fail plate_not_composite_score.

Run:
    cd /home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/validators-reconstruction
    source .venv/bin/activate
    python -m pytest test_validators.py -v
    # or
    python test_validators.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Allow direct script run (no -m flag) -- import sibling modules directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plate_not_composite  # noqa: E402
import role_purity  # noqa: E402
import jigsaw_separation  # noqa: E402
import proof_progression  # noqa: E402
import underlayer_reversal_check  # noqa: E402
import final_match  # noqa: E402


def run_all_validators(plan, output_path=None):
    """Local copy of the runner using direct sibling-module imports.

    The package-style import in run_all_validators.py uses `from . import ...`
    which doesn't work when the directory has a hyphen in its name.
    For tests, we replicate the logic here. The production runner is callable
    via `python -m` from a parent that adds this dir to sys.path.
    """
    import json
    import time
    t0 = time.time()
    report = {"plan_id": plan.get("plan_id", "unknown"), "summary": {}, "validators": {}}
    plates = plan.get("plates", [])
    target = plan.get("target_image")
    final_composite = plan.get("final_composite")
    role_labels = plan.get("cell_role_labels", {})
    cell_positions = plan.get("cell_pixel_positions", {})
    cell_adjacency = plan.get("cell_adjacency", {})
    proof_states = plan.get("proof_states", [])
    visibility = plan.get("visibility_mask")

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return {"passes": False, "error": f"{type(e).__name__}: {e}"}

    # V1 -- cache final composite once
    pnc = []
    final_cached = None
    if final_composite is not None:
        try:
            import numpy as _np
            from PIL import Image as _PIL
            if isinstance(final_composite, _np.ndarray):
                final_cached = final_composite
            else:
                final_cached = _np.asarray(_PIL.open(final_composite).convert("RGB"))
        except Exception:
            final_cached = final_composite
    for p in plates:
        block_id = p.get("block_id")
        if final_cached is None or p.get("plate_preview") is None:
            pnc.append({"block_id": block_id, "passes": False, "error": "missing inputs"})
            continue
        r = _safe(plate_not_composite.score, p["plate_preview"], final_cached, return_components=True)
        r["block_id"] = block_id
        pnc.append(r)
    np_ = sum(1 for r in pnc if r.get("passes"))
    worst = max(pnc, key=lambda x: x.get("badness_score", 0.0), default=None)
    report["validators"]["plate_not_composite"] = {
        "per_plate": pnc,
        "aggregate": {"n_pass": np_, "n_fail": len(pnc) - np_,
                      "worst_block_id": worst.get("block_id") if worst else None,
                      "worst_score": worst.get("badness_score") if worst else None},
        "passes": np_ == len(pnc) and len(pnc) > 0,
    }
    # V2
    rp = []
    for p in plates:
        labels = p.get("cell_role_labels_override", role_labels)
        r = _safe(role_purity.score, p.get("block_id"), p.get("cells_in_plate", []), labels, return_components=True)
        if "block_id" not in r:
            r["block_id"] = p.get("block_id")
        rp.append(r)
    np_ = sum(1 for r in rp if r.get("passes"))
    report["validators"]["role_purity"] = {
        "per_plate": rp, "aggregate": {"n_pass": np_, "n_fail": len(rp) - np_},
        "passes": np_ == len(rp) and len(rp) > 0,
    }
    # V3
    js = []
    for p in plates:
        if p.get("inked_mask") is not None:
            r = _safe(jigsaw_separation.score_from_mask, p["inked_mask"],
                      p.get("dpi", jigsaw_separation.DEFAULT_DPI), return_components=True)
        else:
            r = _safe(jigsaw_separation.score, cells_in_plate=p.get("cells_in_plate", []),
                     cell_pixel_positions=cell_positions, adjacency=cell_adjacency,
                     dpi=p.get("dpi", jigsaw_separation.DEFAULT_DPI), return_components=True)
        r["block_id"] = p.get("block_id")
        js.append(r)
    np_ = sum(1 for r in js if r.get("passes"))
    report["validators"]["jigsaw_separation"] = {
        "per_plate": js, "aggregate": {"n_pass": np_, "n_fail": len(js) - np_},
        "passes": np_ == len(js) and len(js) > 0,
    }
    # V4
    if len(proof_states) < 2:
        report["validators"]["proof_progression"] = {"passes": False, "error": "need >= 2 proofs"}
    else:
        report["validators"]["proof_progression"] = _safe(
            proof_progression.score, proof_states, return_components=True
        )
    # V5 -- cache target once
    target_cached = None
    if target is not None:
        try:
            from PIL import Image as _PIL
            if isinstance(target, np.ndarray):
                target_cached = target
            else:
                target_cached = np.asarray(_PIL.open(target).convert("RGB"))
        except Exception:
            target_cached = target
    ur = []
    for p in plates:
        if p.get("plate_svg") is None:
            ur.append({"block_id": p.get("block_id"), "passes": False, "error": "no svg"})
            continue
        r = _safe(underlayer_reversal_check.check, p["plate_svg"], p.get("pull_preview"), target_cached, return_components=True)
        r["block_id"] = p.get("block_id")
        ur.append(r)
    np_ = sum(1 for r in ur if r.get("passes"))
    report["validators"]["underlayer_reversal"] = {
        "per_plate": ur, "aggregate": {"n_pass": np_, "n_fail": len(ur) - np_},
        "passes": np_ == len(ur) and len(ur) > 0,
    }
    # V6
    if target is None or final_composite is None:
        report["validators"]["final_match"] = {"passes": True, "advisory_only": True, "error": "no target/final"}
    else:
        report["validators"]["final_match"] = _safe(final_match.score, target, final_composite,
                                                     visibility_mask=visibility)

    gating = ["plate_not_composite", "role_purity", "jigsaw_separation", "proof_progression", "underlayer_reversal"]
    n_pass = sum(1 for k in gating if report["validators"].get(k, {}).get("passes"))
    elapsed = (time.time() - t0) * 1000.0
    report["summary"] = {
        "passes_overall": n_pass == len(gating),
        "n_gates_passed": n_pass,
        "n_gates_total": len(gating),
        "advisory_score": report["validators"].get("final_match", {}),
        "elapsed_ms": float(elapsed),
    }
    if output_path:
        def _d(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.bool_): return bool(o)
            return str(o)
        Path(output_path).write_text(json.dumps(report, indent=2, default=_d))
    return report


V13_DIR = Path(
    "/srv/woodblock-share/chuck-mcp-iterations/current-review/"
    "2026-05-14_iter-v13_methodology-adaptive-proofs-emma"
)
EMMA_TARGET = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")


# ============================================================
# Validator 1: plate_not_composite — against real v13 blocks
# ============================================================

def test_v13_late_blocks_fail_plate_not_composite():
    """v13's blocks 24, 25, 26 must FAIL the validator (they look like composites)."""
    final = V13_DIR / "final_methodology_composite.png"
    assert final.exists(), f"missing v13 final composite: {final}"

    results = {}
    for block_id in [24, 25, 26]:
        block = V13_DIR / f"block_{block_id:02d}.png"
        if not block.exists():
            print(f"  SKIP block_{block_id:02d} (file missing)")
            continue
        r = plate_not_composite.score(str(block), str(final), return_components=True)
        results[block_id] = r
        print(
            f"  v13 block_{block_id:02d}: badness={r['badness_score']:.3f} "
            f"cos_sim={r['cosine_similarity']:.3f} "
            f"area_frac={r['inked_area_fraction']:.3f} "
            f"PASS={r['passes']}"
        )

    # ASSERT: every late block (24, 25, 26) must FAIL the validator
    for block_id, r in results.items():
        assert not r["passes"], (
            f"VALIDATOR FAILED — v13 block_{block_id:02d} should FAIL "
            f"plate_not_composite but got badness={r['badness_score']:.3f}"
        )
    print(f"  PASS: all {len(results)} v13 late blocks correctly REJECTED")
    return results


def test_v13_early_blocks_results():
    """v13's blocks 01-03 are nearly empty -- expected to PASS (mostly white)."""
    final = V13_DIR / "final_methodology_composite.png"
    if not final.exists():
        print("  SKIP (no v13 dir)")
        return
    for block_id in [1, 2, 3, 10, 15, 20]:
        block = V13_DIR / f"block_{block_id:02d}.png"
        if not block.exists():
            continue
        r = plate_not_composite.score(str(block), str(final), return_components=True)
        print(
            f"  v13 block_{block_id:02d}: badness={r['badness_score']:.3f} "
            f"cos_sim={r['cosine_similarity']:.3f} "
            f"area_frac={r['inked_area_fraction']:.3f} "
            f"PASS={r['passes']}"
        )


def test_synthetic_good_plate_passes():
    """Synthetic plate with sparse jigsaw zones - should PASS."""
    H, W = 1000, 1000
    final = np.full((H, W, 3), 180, dtype=np.uint8)  # gray-ish composite
    final[200:600, 200:800] = [100, 50, 50]  # bulk red region

    # GOOD plate: sparse small zones, NOT spanning the full image
    plate = np.full((H, W, 3), 230, dtype=np.uint8)  # wood-grain ground
    plate[100:200, 100:200] = [50, 50, 50]
    plate[700:780, 700:780] = [50, 50, 50]

    r = plate_not_composite.score(plate, final, return_components=True)
    print(f"  synthetic GOOD plate: badness={r['badness_score']:.3f} PASS={r['passes']}")
    assert r["passes"], f"expected good plate to PASS, got {r}"


def test_synthetic_bad_plate_fails():
    """Synthetic plate that's just a faded copy of final - should FAIL."""
    H, W = 1000, 1000
    final = np.random.RandomState(42).randint(0, 255, (H, W, 3), dtype=np.uint8)
    # "Bad plate" = exactly the final, lightened
    plate = (final.astype(np.float32) * 0.7 + 76).clip(0, 255).astype(np.uint8)

    r = plate_not_composite.score(plate, final, return_components=True)
    print(f"  synthetic BAD plate (faded composite): badness={r['badness_score']:.3f} PASS={r['passes']}")
    assert not r["passes"], f"expected bad plate to FAIL, got {r}"


# ============================================================
# Validator 2: role_purity
# ============================================================

def test_role_purity_pure_plate_passes():
    labels = {c: "underlayer_light" for c in range(10)}
    r = role_purity.score(plate_id=1, cells_in_plate=list(range(10)),
                          cell_role_labels=labels, return_components=True)
    print(f"  PURE plate: purity={r['purity_score']:.2f} PASS={r['passes']}")
    assert r["passes"]


def test_role_purity_mixed_plate_fails():
    labels = {0: "underlayer_light", 1: "underlayer_light", 2: "key_detail",
              3: "key_detail", 4: "local_chroma", 5: "regional_mass"}
    r = role_purity.score(plate_id=1, cells_in_plate=list(range(6)),
                          cell_role_labels=labels, return_components=True)
    print(f"  MIXED plate: purity={r['purity_score']:.2f} "
          f"distinct={r['distinct_role_count']} PASS={r['passes']}")
    assert not r["passes"]


# ============================================================
# Validator 3: jigsaw_separation
# ============================================================

def test_jigsaw_far_zones_pass():
    H, W = 1000, 1000
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    mask[800:900, 800:900] = 1
    # 300dpi -> 5mm = 5 * 300/25.4 = ~59 px. Far zones are ~700px apart.
    r = jigsaw_separation.score_from_mask(mask, dpi=300, return_components=True)
    print(f"  FAR zones @ 300dpi: min_mm={r['min_separation_mm']:.2f} PASS={r['passes']}")
    assert r["passes"]


def test_jigsaw_close_zones_fail():
    H, W = 1000, 1000
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    mask[201:300, 100:200] = 1   # 1px gap = ~0.08mm
    r = jigsaw_separation.score_from_mask(mask, dpi=300, return_components=True)
    print(f"  CLOSE zones @ 300dpi: min_mm={r['min_separation_mm']:.2f} "
          f"PASS={r['passes']} violations={len(r['violations'])}")
    assert not r["passes"]


# ============================================================
# Validator 4: proof_progression
# ============================================================

def test_proof_progression_good():
    H, W = 200, 200
    base = np.full((H, W, 3), 240, dtype=np.uint8)
    a = base.copy()
    b = base.copy(); b[10:60, 10:60] = [180, 60, 60]
    c = b.copy(); c[80:140, 80:140] = [40, 40, 200]
    d = c.copy(); d[160:190, 160:190] = [10, 10, 10]
    r = proof_progression.score([a, b, c, d], return_components=True)
    print(f"  GOOD progression: score={r['progression_score']:.2f} PASS={r['passes']}")
    assert r["passes"]


def test_proof_progression_stalled():
    H, W = 200, 200
    base = np.full((H, W, 3), 240, dtype=np.uint8)
    a = base.copy()
    b = base.copy(); b[10:60, 10:60] = [180, 60, 60]
    c = b.copy()  # no change
    r = proof_progression.score([a, b, c], return_components=True)
    print(f"  STALLED progression: score={r['progression_score']:.2f} PASS={r['passes']}")
    assert not r["passes"]


# ============================================================
# Validator 5: underlayer_reversal_check
# ============================================================

def test_reversal_svg_flipped():
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<g transform="matrix(-1, 0, 0, 1, 100, 0)"><rect x="10" y="10" width="20" height="20"/></g>'
            '</svg>'
        )
        path = f.name
    r = underlayer_reversal_check.check(path, return_components=True)
    print(f"  FLIPPED svg: passes={r['passes']} evidence={r['plate_evidence']}")
    assert r["plate_is_flipped"]


def test_reversal_svg_not_flipped():
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<rect x="10" y="10" width="20" height="20"/>'
            '</svg>'
        )
        path = f.name
    r = underlayer_reversal_check.check(path, return_components=True)
    print(f"  NOT-FLIPPED svg: passes={r['passes']} evidence={r['plate_evidence']}")
    assert not r["plate_is_flipped"]


# ============================================================
# Validator 6: final_match (advisory)
# ============================================================

def test_final_match_identical():
    H, W = 200, 200
    target = np.full((H, W, 3), 128, dtype=np.uint8)
    r = final_match.score(target, target)
    print(f"  IDENTICAL match: dE_mean={r['delta_e_mean']:.2f} "
          f"advisory_passes={r['advisory_passes']}")
    assert r["delta_e_mean"] < 1.0


def test_final_match_different():
    H, W = 200, 200
    target = np.full((H, W, 3), 50, dtype=np.uint8)
    final = np.full((H, W, 3), 200, dtype=np.uint8)
    r = final_match.score(target, final)
    print(f"  DIFFERENT: dE_mean={r['delta_e_mean']:.2f} "
          f"advisory_passes={r['advisory_passes']}")
    assert r["delta_e_mean"] > 10.0


# ============================================================
# Master runner integration test
# ============================================================

def test_run_all_validators_on_v13_sample():
    """Run the master runner against a minimal v13-derived plan.

    Builds a synthetic plan using real v13 block images + final composite
    and expects FAILURE on plate_not_composite for late blocks.
    """
    if not V13_DIR.exists():
        print("  SKIP (no v13 dir)")
        return
    final = V13_DIR / "final_methodology_composite.png"
    target = EMMA_TARGET if EMMA_TARGET.exists() else final

    plates = []
    role_labels = {}
    # Synthetic role assignment for testing
    for n in [1, 5, 12, 20, 24, 25, 26]:
        block = V13_DIR / f"block_{n:02d}.png"
        if not block.exists():
            continue
        # Make a fake mirrored SVG for each block so the reversal check passes
        svg_path = block.with_suffix(".test.svg")
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<g transform="matrix(-1, 0, 0, 1, 100, 0)"><rect/></g></svg>'
        )
        # Pretend each block has cells N*10..N*10+9 all of one role
        cells = list(range(n * 10, n * 10 + 10))
        role = "underlayer_light" if n < 5 else ("regional_mass" if n < 20 else "key_detail")
        for c in cells:
            role_labels[c] = role
        plates.append({
            "block_id": n,
            "plate_preview": str(block),
            "plate_svg": str(svg_path),
            "pull_preview": str(block),  # placeholder
            "cells_in_plate": cells,
            "role": role,
            "dpi": 300,
        })

    # Synthetic proof states from existing pull-preview images
    pulls_dir = V13_DIR
    proof_candidates = sorted(pulls_dir.glob("block_*.png"))[:7]
    proof_states = [str(p) for p in proof_candidates]

    plan = {
        "plan_id": "v13-synthetic-test",
        "target_image": str(target),
        "final_composite": str(final),
        "plates": plates,
        "cell_role_labels": role_labels,
        "cell_pixel_positions": {},
        "cell_adjacency": {},
        "proof_states": proof_states,
    }

    t0 = time.time()
    report = run_all_validators(plan)
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"\n  Master runner elapsed: {elapsed_ms:.1f} ms")
    print(f"  Overall passes: {report['summary']['passes_overall']}")
    print(f"  Gates passed: {report['summary']['n_gates_passed']}/{report['summary']['n_gates_total']}")
    print(f"  plate_not_composite: passes={report['validators']['plate_not_composite']['passes']}")
    print(f"     aggregate: {report['validators']['plate_not_composite']['aggregate']}")

    # ASSERT: this plan SHOULD FAIL plate_not_composite because late blocks are composites
    assert not report["validators"]["plate_not_composite"]["passes"], (
        "v13 plates should fail plate_not_composite validator"
    )
    # Performance gate: < 3000 ms
    assert elapsed_ms < 3000, f"validation took {elapsed_ms:.0f}ms — too slow"
    print(f"  PASS: master runner correctly rejected v13 plates in {elapsed_ms:.1f}ms")


def main():
    """Run all tests."""
    tests = [
        ("V1: v13 late blocks FAIL plate_not_composite", test_v13_late_blocks_fail_plate_not_composite),
        ("V1: v13 early/mid blocks results",             test_v13_early_blocks_results),
        ("V1: synthetic GOOD plate passes",              test_synthetic_good_plate_passes),
        ("V1: synthetic BAD plate fails",                test_synthetic_bad_plate_fails),
        ("V2: pure plate passes",                        test_role_purity_pure_plate_passes),
        ("V2: mixed-role plate fails",                   test_role_purity_mixed_plate_fails),
        ("V3: far zones pass",                           test_jigsaw_far_zones_pass),
        ("V3: close zones fail",                         test_jigsaw_close_zones_fail),
        ("V4: good progression passes",                  test_proof_progression_good),
        ("V4: stalled progression fails",                test_proof_progression_stalled),
        ("V5: flipped SVG detected",                     test_reversal_svg_flipped),
        ("V5: unflipped SVG detected",                   test_reversal_svg_not_flipped),
        ("V6: identical final-match",                    test_final_match_identical),
        ("V6: different final-match",                    test_final_match_different),
        ("MASTER: run_all_validators on v13",            test_run_all_validators_on_v13_sample),
    ]
    n_pass = n_fail = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            n_pass += 1
            print(f"  -> PASS")
        except AssertionError as e:
            n_fail += 1
            print(f"  -> FAIL: {e}")
        except Exception as e:
            n_fail += 1
            print(f"  -> ERROR: {type(e).__name__}: {e}")
    print(f"\n========\nTotal: {n_pass}/{len(tests)} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
