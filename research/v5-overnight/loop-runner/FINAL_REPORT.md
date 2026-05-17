# v5-overnight loop-runner FINAL_REPORT

**Run window:** 2026-05-17 00:15 EDT → 00:25 EDT (10 min, JAX-warm)
**Swarm:** swarm-1778989256284-xvs2l5
**Input:** /srv/woodblock-share/input-images/close_emma_2002_2048.jpg

## Iterations completed
**12 of 12** (cap hit). Acceptance gate (6/6 validators + dE_mean<8 + underlayer_match>=85%) **never triggered**.

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

## Best iteration (lowest dE_mean among 6/6 passes)
**No iteration passed 6/6.** Best overall by dE_mean: **iter 10** (= iter 12 repro).
- dE_mean: 19.053
- dE_p95: 40.576
- validators_passed: 2 / 6
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

## Honest verdict: did v5 achieve V1.0-quality Emma reconstruction overnight?
**No.**

The 4 sibling TDD agents (snic-real, mediapipe-spatial, alpha-proof-dumper, mokuhanga-pigments) landed structural fixes that produce a **runnable, deterministic, 20-32 plate Emma plan**, which is real progress vs prior v4 builds. But the reconstruction quality is far from V1.0:

- **dE_mean 19.05 vs target 8** — over 2.3× the target. V1.0 woodblock proofs typically sit dE 6-9 against the photo.
- **Validators 2/6** — only `final_match` advisory passes plus one other; `plate_not_composite`, `role_purity`, `jigsaw_separation`, `proof_progression`, `underlayer_reversal` all fail. `plates_pass_pnc=0/28` means **zero plates** survive the not-a-composite check.
- **Underlayer match 34.38%** vs target 85% — the rendered underlayer plates barely overlap Reid's hand-annotated reference. The mediapipe face-region constraint is shrinking underlayer coverage, not aligning it.
- **Hybrid optimizer is in a deep local minimum.** Every outer iter reports identical Stage 3 loss trajectory (34.493 → 28.357), meaning the m-step is not re-seeding from the previous outer. The "outer" loop is effectively a no-op for cost.

What an overnight loop CAN'T fix on its own:
- Optimizer escape velocity (would need random restarts, basin-hopping, or simulated annealing).
- Mixer fidelity (Mixbox lerp at dE 19 is the K-M layered-overprint problem documented in memory `feedback_overlay_not_mixing`).
- Underlayer methodology (the reference is **hand-annotated** semantic regions; our union-of-alpha-masks IoU is the wrong metric for "match" — we'd need region-correspondence scoring).
- `plate_not_composite` failure (every plate is being detected as a multi-ink composite — likely the SNIC adjacency merge is over-aggressive).

Next-session recommendations (NOT executed in this loop):
1. Add random-restart wrapper around the inner Stage-3 LBFGS to escape the 28.357 minimum.
2. Replace Mixbox forward render with K-M overprint per memory `feedback_overlay_not_mixing`.
3. Fix `plate_not_composite` validator OR fix SNIC merge to actually produce single-ink plates.
4. Replace underlayer IoU with semantic region matching (load the hand-annotated regions as labelled polygons, score correspondence).
5. Wire `--target-pulls` into the actual cost so `m_prior` is meaningful.

## Files
- CSV: `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v5-overnight/loop-runner/iterations.csv`
- Best sheet: `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-10/sheet_iter_10.png`
- All sheets: `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-{01..12}/`
- Per-iter job dirs: `/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-{01..12}/`
- Per-iter logs: `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v5-overnight/loop-runner/iter_{01..12}.log`
