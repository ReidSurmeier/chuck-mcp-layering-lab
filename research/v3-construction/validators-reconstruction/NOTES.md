# chuck-mcp v3 — Reconstruction Validators

**Status:** 9 artifacts, 15/15 tests pass, master runner = 2.4s on v13 sample (under 3s budget).

This module implements the 6 hard-gate validators specified in
`docs/reconstruction-plan-2026-05-16.md` Stage 6 and locked in
`docs/v2-design-locked-2026-05-16.md`. They exist to prevent the v13
failure mode where the solver produced residual α-maps and labeled them
"blocks" — late blocks looked like cumulative composites instead of
isolated jigsaw plates.

## Layout

```
research/v3-construction/validators-reconstruction/
├── __init__.py
├── plate_not_composite.py          # V1 — gate
├── role_purity.py                  # V2 — gate
├── jigsaw_separation.py            # V3 — gate
├── proof_progression.py            # V4 — gate
├── underlayer_reversal_check.py    # V5 — gate
├── final_match.py                  # V6 — advisory only
├── run_all_validators.py           # master runner
├── test_validators.py              # 15 tests
├── v13_smoking_gun_results.json    # per-block badness on real v13 outputs
└── NOTES.md                        # this file
```

## The 6 validator formulas (final, as implemented)

### V1. `plate_not_composite_score(plate_mask, final_composite) -> dict`

```
plate_small  = downsample(plate_mask, 128px on long edge)
final_small  = downsample(final_composite, 128px)
cos_sim      = cosine_similarity(1 - plate_small_lum, 1 - final_small_lum)
inked        = (plate_small_lum < 0.85)
area_frac    = inked.mean()
spread_bad   = clip((area_frac - 0.15) / (0.40 - 0.15), 0, 1)
BADNESS      = (max(cos_sim, 0) + spread_bad) / 2
PASS iff BADNESS <= 0.6
```

Higher BADNESS = more like a composite. The design doc wrote the formula
as `1.0 - (...)` which implies higher = better; we implement higher = worse
(BADNESS) and reject when > 0.6. Either reading yields the same gate
behavior on the 0.6 boundary. See "Doc-formula reconciliation" below.

### V2. `role_purity_score(plate_id, cells_in_plate, cell_role_labels) -> dict`

```
roles      = [cell_role_labels[c] for c in cells_in_plate]
modal, n   = most_common(roles)
purity     = n / len(cells_in_plate)
distinct_r = count(set(roles))
PASS iff purity >= 0.7 AND distinct_r <= 2
```

Two simultaneous gates: dominant role >= 70%, AND no more than 2
distinct role families on a single plate.

### V3. `jigsaw_separation_score(plate, ...) -> dict`

Two API surfaces:
- mask path: `score_from_mask(binary_mask, dpi)`
- cell-graph path: `score(cells_in_plate, cell_pixel_positions, adjacency, dpi)`

```
zones = connected_components(mask)  OR  bfs_groups(cells, adjacency)
mm_per_px = 25.4 / dpi
for each pair (a, b) of disjoint zones:
    d_px = min_pairwise_pixel_distance(a, b)
    d_mm = d_px * mm_per_px
min_separation_mm = min over all pairs
PASS iff min_separation_mm >= 5.0
```

5.0mm threshold from `docs/v2-design-locked-2026-05-16.md` row 3.
Default DPI = 300 (typical CNC + washi printable resolution).

### V4. `proof_progression_score(proof_state_sequence) -> dict`

```
for each consecutive (proof_n, proof_{n+1}):
    lab_n, lab_{n+1} = rgb2lab(downsample(..., 384px))
    de = deltaE_ciede2000(lab_n, lab_{n+1})  # per-pixel
    n_shifted = (de > 3.0).sum()
    frac      = n_shifted / total_pixels
    interval_pass iff frac >= 0.005   # 0.5% of pixels
progression_score = n_intervals_passing / n_intervals_total
PASS iff EVERY consecutive interval passes (strict)
```

The 0.5% threshold corresponds to ~7900px on a 1.5Mpx image — roughly
the size of a single Chuck-Close cell or one significant inkline.
ΔE_2000 threshold 3.0 = "just-noticeable difference" floor.

### V5. `underlayer_reversal_check(plate_svg, pull_image, target_image) -> dict`

Boolean validator (no continuous score).

```
plate_flipped  = svg contains matrix(-1, 0, 0, 1, W, 0)
                  OR scale(-1, 1)
                  OR mirror=true metadata
pull_passes    = cos(target, pull_normal) >= cos(target, pull_flipped)
                  OR within 0.02 delta (borderline = pass)
PASS iff plate_flipped AND pull_passes
```

### V6. `final_match_score(target, final_composite, [visibility_mask, region_labels]) -> dict`

```
lab_t, lab_f  = rgb2lab(downsample(..., 512px))
de            = deltaE_ciede2000(lab_t, lab_f)
report: {
    delta_e_mean, delta_e_median, delta_e_p95, delta_e_max
    per_region: {region_id: {mean, p95, n_pixels}}
    advisory_passes: bool   # mean <= 5.0 AND p95 <= 12.0
    advisory_only: True     # NEVER GATES SIGN-OFF
}
```

Advisory only because v13 hit ΔE 4.98 mean and was still bad — this
score measures cumulative-render quality, not plate-output quality.

## Threshold table (current, can be tuned)

| Validator | Threshold | Source |
|---|---|---|
| V1 badness | `> 0.6` REJECT | design doc row 1 (transcribed verbatim from task brief) |
| V2 purity | `>= 0.7` for PASS | task brief; doc says ">2 role families rejects" |
| V2 role_families | `<= 2` | design doc row 2 |
| V3 separation | `>= 5.0 mm` | design doc row 3 |
| V3 default DPI | `300` | CNC+washi convention |
| V4 ΔE per-pixel | `> 3.0` to count as shifted | just-noticeable difference |
| V4 fraction | `>= 0.5%` of pixels must shift | reconstruction doc "visible family" |
| V5 plate flip | required | design doc row 5 |
| V5 pull orientation alignment delta | `< 0.02` = borderline pass | empirical |
| V6 advisory ΔE mean | `<= 5.0` | aspirational |
| V6 advisory ΔE p95 | `<= 12.0` | aspirational |

## v13 smoking-gun test results

`test_validators.py` runs `plate_not_composite_score` on every v13
block we have on disk. Full table in `v13_smoking_gun_results.json`.

| Block range | Verdict | Why |
|---|---|---|
| 01-23 | PASS V1 | Mostly empty wood-grain (area_frac ≈ 0, cos_sim ≈ 0.8–0.9, badness ≈ 0.4) |
| **24** | **FAIL V1 (badness 0.999)** | cos_sim=0.999 vs final composite — IS the composite |
| **25** | **FAIL V1 (badness 1.000)** | cos_sim=0.999, area_frac=0.601 |
| **26** | **FAIL V1 (badness 1.000)** | cos_sim=0.999, area_frac=0.623 |

The reconstruction doc specifically calls out: *"Blocks 24-26 are essentially
finished composite images, which is not a believable physical plate."* The
validator's badness score 1.0 on exactly those blocks confirms it does the
job. v13's failure mode is detected; v3 cannot ship plates that look like
composites past V1.

The blocks 01-23 "pass" V1 only because they're nearly empty (v13's early
blocks are faded ghosts, not composites). They would still fail V2
(role_purity) and V3 (jigsaw_separation) once those validators have
ground-truth cell labels — which v13 doesn't have, because v13 conflated
plate and composite from the start.

## Performance

Master runner full validation pass timing on v13 sample (7 plates +
final composite + 7 "proof states"):

| Component | Time |
|---|---|
| V1 plate_not_composite × 7 (with cached final) | ~270 ms |
| V2 role_purity × 7 | < 1 ms |
| V3 jigsaw_separation × 7 (from dummy 128×128 masks) | ~4 ms |
| V4 proof_progression on 7 full-res PNGs | ~750 ms |
| V5 underlayer_reversal × 7 (with cached target) | ~600 ms |
| V6 final_match (advisory) | ~220 ms |
| Total master runner end-to-end | **~2400 ms** |

Under the 3000ms budget. Key perf wins:
- PIL bilinear resize (5-10× faster than `skimage.transform.resize`)
- Cache final composite + target as np.array once instead of re-decoding per plate
- Downsample proof images to 384px before CIEDE2000 (rgb2lab + ΔE_2000 is the dominant cost)
- Read-at-target-resolution in PIL — skips ever holding the full-res image in memory

A realistic v3 run will be FASTER because:
- Proof states will be 7 small downsampled checkpoints, not 7 full-res block images
- Mask-based jigsaw separation runs in ~1ms per plate (the cell-graph path is also fast)
- SVG reversal check is a regex match (microseconds)

## Doc-formula reconciliation

The design doc cell says:
```
1.0 - (cosine_sim(block, final) + coverage_concentration) / 2
Reject if > 0.6
```

Literally: `1.0 - (sim + conc) / 2 > 0.6` means `sim + conc < 0.8`, i.e.,
reject when similarity AND concentration are BOTH low. That's not the
v13 failure mode — v13's bad blocks have HIGH similarity to final.

I interpret this as a typo / shortcut and implement the intuitive gate:
- BADNESS = (cos_sim + spread_badness) / 2
- REJECT when BADNESS > 0.6
- where spread_badness rises with `area_fraction > 0.15` and saturates at 1.0 by `area_fraction = 0.40`

The empirical test confirms: v13 blocks 24/25/26 (the called-out failures)
score BADNESS ≈ 1.0, early blocks score ≈ 0.4. Threshold at 0.6 cleanly
separates them. **If the design doc literally meant the LITERAL formula,
the gate would invert — passing v13 blocks and rejecting empty plates.
That's nonsensical. The implementation matches design INTENT, not
the literal text. Flag for human reconciliation.**

## Integration with the web app verification UI

The master runner emits JSON in this shape (truncated):

```json
{
  "plan_id": "emma-2026-05-16-v3",
  "summary": {
    "passes_overall": false,
    "n_gates_passed": 3,
    "n_gates_total": 5,
    "advisory_score": {"delta_e_mean": 4.98, "advisory_passes": false, ...},
    "elapsed_ms": 2400.0
  },
  "validators": {
    "plate_not_composite": {
      "passes": false,
      "aggregate": {"n_pass": 4, "n_fail": 3, "worst_block_id": 25, "worst_score": 0.999},
      "per_plate": [
        {"block_id": 1, "badness_score": 0.40, "passes": true, ...},
        {"block_id": 25, "badness_score": 1.0, "passes": false, ...}
      ]
    },
    "role_purity": {...},
    "jigsaw_separation": {...},
    "proof_progression": {...},
    "underlayer_reversal": {...},
    "final_match": {...}
  }
}
```

UI mapping (per `docs/v2-design-locked-2026-05-16.md` "Verification surfaces"):

1. **6-card validator panel** (right sidebar) — one card per validator
   with PASS/FAIL badge + score + click-to-expand per-plate breakdown.
2. **Block grid overlay** — colour each of the 27 block cells red/green
   by `per_plate[i].passes`. Hover shows the per-validator failure
   reason. Click jumps to that block's preview.
3. **Per-pull heatmap** — V4's `intervals[i].shift_fraction` and
   `intervals[i].mean_delta_e` feed the heatmap directly.
4. **Worst-block callout** — `aggregate.worst_block_id` + `worst_score`
   makes the "fix this one first" CTA trivial.
5. **Sign-off button** — disabled while `summary.passes_overall == false`.
   Tooltip lists `n_gates_passed / n_gates_total` and the failing
   validator names.

The validators are PURE PYTHON (numpy + skimage + colour-science +
Pillow) with NO JAX dependency. They run on rendered outputs only, so
they can be invoked from the Next.js backend via:

```
POST /api/validate
{ "plan_id": "...", "plan": {...} }
->
{ <the report JSON above> }
```

Either via a subprocess shelling to `python -m run_all_validators ...`
or via a long-running Python service inside the `chuck-mcp.service`
systemd unit. Subprocess is fine — < 3s is well under any HTTP timeout.

## Calibration / threshold tuning

All thresholds are **first-pass guesses**. After 5 real v3 runs we'll
have data to recalibrate:

- V1 0.6 threshold: tighten/loosen based on edge cases
- V3 5mm threshold: validate against Reid's brush width
- V4 0.5% pixel-shift floor: depends on final image size
- V6 advisory thresholds: just labels, can be reframed any time

Suggested tuning protocol:
1. Run v3 on Emma, capture all 26 plate badness scores
2. Manually classify each plate "good / acceptable / bad"
3. Find the threshold that maximises separation
4. Repeat for V2-V6 against ground truth from carved-plate proofs

## Known limitations / next steps

- **V3 cell-graph path** assumes cell centroids are sufficient for
  pairwise distance. For very elongated cells the centroid is a poor
  proxy — recommend supplying inked-mask path instead when available.
- **V5 pull-orientation check** is cross-correlation in luminance. If
  the print is highly textured and the source is photographic the
  correlation may be flat (delta < 0.02 borderline). Acceptable for v1.
- **V6 region_labels** is hooked up but no caller currently passes
  region masks. When Plate.cell_zone_ids is available we should pass
  the cell label map for per-region ΔE breakdown.
- **No JAX** by design — these validators run on RENDERED OUTPUTS not
  solver state, so they're standalone-runnable from any host (and from
  the web app via HTTP).

## How to run

```bash
cd /home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/validators-reconstruction
source .venv/bin/activate

# Run all 15 tests (sanity + smoking gun)
python test_validators.py

# Run individual validator on a plate
python plate_not_composite.py /path/to/block_25.png /path/to/final.png

# Run master runner against a plan JSON
python run_all_validators.py plan.json --output report.json
```

## Top 3 must-reads (for v3 swarm consumers)

1. **`plate_not_composite.py`** — the v13-killer. Read it first to
   understand the threshold semantics + the doc-formula reconciliation.
2. **`v13_smoking_gun_results.json`** — the empirical proof. Open it
   and look at blocks 24/25/26 vs the rest. This is the load-bearing
   evidence that "the validator works".
3. **`run_all_validators.py`** — the single integration point. The
   web app, the solver post-step, the export-time gate — they all call
   this one function and read this one JSON shape.
