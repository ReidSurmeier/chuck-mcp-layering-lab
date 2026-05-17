# v5-overnight loop-runner FINAL_REPORT

**Run window:** 2026-05-17 00:15 EDT → 00:25 EDT (10 min, JAX-warm)
**Swarm:** swarm-1778989256284-xvs2l5
**Input:** /srv/woodblock-share/input-images/close_emma_2002_2048.jpg

## Iterations completed
**12 of 12** (cap hit). Historical acceptance gate (6/6 validators + dE_mean<8 + underlayer_match>=85%) **never triggered**. As of ADR-0006, the canonical gate is **5 gating validators + final-match advisory/dE + underlayer_match**.

## Per-iteration metrics

| iter | profile | plates | dE_mean | dE_p95 | val | ulm% | wall s |
|------|---------|--------|---------|--------|-----|------|--------|
| 01 | thorough        | 20 | 20.621 | 49.00 | 2 | 33.95 | 120 |
| 02 | thorough        | 20 | 20.621 | 49.00 | 2 | 33.95 |  24 |
| 03 | thorough m=28   | 20 | 20.621 | 49.00 | 2 | 33.95 |  24 |
| 04 | fast            | 20 | 21.124 | 47.89 | 2 | 34.05 |  12 |
| 05 | finer-cells 192 | 20 | 19.569 | 44.01 | 2 | 33.96 |  24 |
| 06 | more-plates 28  | 26 | 20.652 | 48.35 | 2 | 34.34 |  29 |
| 07 | finer-more      | 28 | 19.426 | 43.91 | 2 | 34.33 |  31 |
| 08 | deeper-inner 50 | 20 | 20.553 | 49.66 | 2 | 33.93 |  24 |
| 09 | bigger-canvas 384 | 20 | 20.770 | 49.26 | 2 | 34.44 |  42 |
| **10** | **extreme-cells 288** | **28** | **19.053** | **40.58** | **2** | **34.38** | **31** |
| 11 | max-cells 384   | 32 | 19.418 | 41.49 | 2 | 34.55 |  31 |
| 12 | extreme-cells repro | 28 | 19.053 | 40.58 | 2 | 34.38 |  31 |

## Best iteration (lowest dE_mean)
No iteration passed acceptance. Best overall by dE_mean: **iter 10** (= iter 12 repro).
- dE_mean: 19.053
- dE_p95: 40.576
- validators_passed: 2 / 5 gating validators
- underlayer_match: 34.38 %
- plates: 28
- sheet: `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-10/sheet_iter_10.png`
- job dir: `/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-10/`

## Trend analysis (did dE monotonically improve?)
**No — exhaustive parameter sweep, not monotonic.** dE_mean ranged 19.05 → 21.12 across iters. Improvement pattern:
- Cells is the only knob that actually moves the loss (192→288 = -1.6 dE; past 288 regresses).
- Plate count: 20→28 helps marginally only when combined with cells>=192; 32 regresses.
- m_prior (target_pulls): **zero effect** — likely not threaded into the hybrid optimizer's actual cost function.
- Inner iters past 25: wasted (Stage 3 loss plateaus at 28.357 by inner 25 every outer iter).
- Outer iters past 1: also wasted (every outer reports identical `Stage 3 loss: 34.493 -> 28.357`). Optimizer hits same local min from same init each time.
- Canvas size 384: regresses (20.77).

Trough is **dE 19.05 with cells=288 + plates=28**. Floor is the hybrid optimizer's local minimum, not a hyperparameter problem.

## 2026-05-17 validator-truth correction

After ADR-0006, iter 10 was re-scored from
`/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-10/validator_plan_masked.json`,
which maps each block to `artifacts/alphas/pull_NNN_alpha.png` as `inked_mask`.

Corrected gate result: **3/5 gating validators pass**.

- `plate_not_composite`: **passes** — 28/28 blocks pass when scored from masks.
- `role_purity`: **passes**.
- `proof_progression`: **passes**.
- `jigsaw_separation`: **fails** — 14/28 blocks fail minimum separation.
- `underlayer_reversal`: **fails**.
- `final_match`: advisory **fails** — dE_mean 19.31 / p95 43.57 against the input.

This changes the diagnosis: v5 is not a composite-plate failure. It is still a
bad reconstruction, but the remaining hard failures are topology/spacing,
underprint methodology, final color match, and the no-op outer loop.

## 2026-05-17 iter 13 after outer-loop fix

Iter 13 was run after fixing the outer loop to warm-start from previous
solved/repaired masks and continuous variables.

- Job dir: `/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/`
- Sheet: `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-13/sheet_iter_13.png`
- Gates: **3/5**
- `plate_not_composite`: **28/28 pass**
- `jigsaw_separation`: **14/28 pass**
- `dE_mean`: **18.778**
- `dE_p95`: **40.3015**
- legacy underlayer overlap: **34.35%**

Stage 3 now carries state forward:

```text
outer 1: 38.001 -> 29.535
outer 2: 29.535 -> 28.947
outer 3: 28.947 -> 28.821
```

This fixes the replay bug, but it does not solve the print. The next build
priority is carved-region Mask topology before continuous color solving, not
more random restarts.

## Honest verdict: did v5 achieve V1.0-quality Emma reconstruction overnight?
**No.**

The 4 sibling TDD agents (snic-real, mediapipe-spatial, alpha-proof-dumper, mokuhanga-pigments) landed structural fixes that produce a **runnable, deterministic, 20-32 plate Emma plan**, which is real progress vs prior v4 builds. But the reconstruction quality is far from V1.0:

- **dE_mean 19.05 vs target 8** — over 2.3× the target. V1.0 woodblock proofs typically sit dE 6-9 against the photo.
- **Validators 2/5 gating** — `role_purity` and `proof_progression` pass. `jigsaw_separation` and `underlayer_reversal` fail. `final_match` is advisory and fails the target (dE is too high).
- **Historical `plate_not_composite` numbers are invalid.** The loop fed full wood-grain `plate_preview` images to the geometry gate instead of the actual inked masks, so every block was measured as full coverage. This was a validator-plan plumbing bug, not evidence that SNIC merged every block into a composite.
- **Underlayer match 34.38%** vs target 85% — the rendered underlayer plates barely overlap Reid's hand-annotated reference. The mediapipe face-region constraint is shrinking underlayer coverage, not aligning it.
- **Hybrid optimizer is in a deep local minimum.** Every outer iter reports identical Stage 3 loss trajectory (34.493 → 28.357), meaning the m-step is not re-seeding from the previous outer. The "outer" loop is effectively a no-op for cost.

What an overnight loop CAN'T fix on its own:
- Optimizer escape velocity (would need random restarts, basin-hopping, or simulated annealing).
- Mixer fidelity (Mixbox lerp at dE 19 is the K-M layered-overprint problem documented in memory `feedback_overlay_not_mixing`).
- Underlayer methodology (the reference is **hand-annotated** semantic regions; our union-of-alpha-masks IoU is the wrong metric for "match" — we'd need region-correspondence scoring).
- `plate_not_composite` failure from this report (the wrong object was scored; re-run with mask-backed validator plans before diagnosing SNIC).

Next-session recommendations (NOT executed in this loop):
1. Re-run validators using mask-backed `inked_mask` inputs and treat that as the first trustworthy v5 measurement.
2. Fix the outer loop so Stage 3 re-seeds from the previous repaired masks instead of replaying the same local solve.
3. Replace dot-cell topology with irregular continuous carved regions before tuning color.
4. Replace underlayer IoU with semantic region matching (load the hand-annotated regions as labelled polygons, score correspondence).
5. Replace Mixbox forward render with K-M or empirical overprint only after the block/mask topology is physically plausible.

## Files
- CSV: `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v5-overnight/loop-runner/iterations.csv`
- Best sheet: `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-10/sheet_iter_10.png`
- All sheets: `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-{01..12}/`
- Per-iter job dirs: `/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-{01..12}/`
- Per-iter logs: `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v5-overnight/loop-runner/iter_{01..12}.log`
