# V4 Build Engineering Audit — 2026-05-17

Auditor: INTEGRATION-AUDIT agent for swarm `swarm-1778984139551-orqcce`.
Date: 2026-05-17 ~22:50 ET.
Venv used: `/home/reidsurmeier/src/chuck-mcp-layering-lab/.venv-renderer/` (Python 3.12.3).

Dependencies the build agents assumed but did NOT install in the venv:
`jax`, `jaxopt`, `opencv-python-headless`, `matplotlib`. All four were missing
on first poll; auditor installed them to make tests runnable. This is itself
a finding: agents shipped code requiring deps they did not declare in any
requirements/pyproject scope visible to the audit harness.

## Pass/Fail Summary

| Agent | Claimed | Verified | Verdict |
|---|---|---|---|
| example-harness        | 14/14 + 95% cov | **14/14 + 95% cov**                              | PASS    |
| production-solver      | 22/22 + 80% cov | **22/22 + 80% cov**                              | PASS    |
| plate-objective        | (no claim)      | **17/17 + 84% cov; 34/34 Sharma to 1e-4**        | PASS (no NOTES.md) |
| hybrid-optimizer       | 11/11 + 86% cov | **12/12 + 86% cov** (after `pip install jax jaxopt`) | PASS (degraded — see below) |
| opus-vision-bench      | (no claim)      | **7/7 + 63% cov; 0/10 Opus calls succeeded**      | PARTIAL (no NOTES.md, benchmark unusable) |

Note: NOTES.md files for `plate-objective/` and `opus-vision-bench/` never
landed within the 30-min poll budget. Source code IS present and tests run.
The two agents shipped runnable artifacts but skipped their NOTES contracts.

## Integration Contract Mismatches

### CRITICAL: `ProductionPlan` ≠ `ProductionPlanInput`

`production-solver/production_plan.py:ProductionPlan` has fields:
```
plan_id, plates: list[PlateSpec], total_pulls, cell_count, image_shape, meta
```
where `PlateSpec` holds `pulls: list[PullSpec]`.

`hybrid-optimizer/alternating_loop.py:ProductionPlanInput` requires:
```
cell_graph: CellGraph, candidate_plates: list[CandidatePlate],
pull_order, role_constraints, pigment_lookup, target_image_rgb,
target_shape, cell_role_labels, cell_adjacency, cell_pixel_positions, dpi
```

**Reproduction (auditor ran this end-to-end):**
```
build_production_plan(...) -> ProductionPlan
optimize(target, production_plan) -> AttributeError:
    'ProductionPlan' object has no attribute 'pigment_lookup'
```

The two agents shipped completely different domain models. Adapter is NOT
"a one-import change" — it requires synthesizing `CellGraph.nodes` (CellNode
needs `lab_color`, `area_px`, `centroid_yx`, `role`) and edges/edge_weights
from raw cell-dict input that `ProductionPlan` no longer carries.

The hybrid-optimizer's NOTES.md openly admits this (limitation #8):
> "The production-solver and plate-objective agents have not yet produced
>  their deliverables. This module ships with a self-contained
>  ProductionPlanInput schema...Wiring into the real production-plan-builder
>  is a one-import change once that module lands."

It is not a one-import change.

### CRITICAL: Three incompatible `Plate` types

- `production-solver/production_plan.py:PlateSpec` — cell_zone_ids + pulls[]
- `hybrid-optimizer/optimization_result.py:SolvedPlate` — flat (no pulls)
- `hybrid-optimizer/jax_continuous_solve.py:FrozenPlate` — adds inked_mask
- `plate-objective/objective_terms.py:Plate` — continuous mask + pigment_lab

`plate-objective.composite_loss(plates: list[Plate], ...)` requires its
own `Plate` type. There is NO call site in `hybrid-optimizer` that uses
`composite_loss` or any term from `plate-objective`. The two modules are
disconnected; hybrid-optimizer's JAX loss is hand-rolled in
`jax_continuous_solve.py:_build_loss_fn` and uses only ΔE_76 + alpha-compose
— it doesn't pull in `plate_not_composite_penalty_per_plate`,
`cell_exclusivity_penalty`, `role_coverage_caps_penalty`, etc.

The audit doc's Phase 3 ("plate organization INTO the objective") is NOT
realized — the objective terms exist (plate-objective) and the optimizer
exists (hybrid-optimizer) but they don't talk.

### MINOR: example-harness uses proxy validators only

example-harness/acceptance_harness.py imports zero modules from
`v3-construction/validators-reconstruction/`. Its `plate_not_composite_score`
is a cosine-similarity proxy. NOTES.md acknowledges this and recommends
"Replace the proxy validators with the real ones" as the next step.

### MINOR: hybrid-optimizer fragile import path

When JAX is missing the venv, `jax_continuous_solve.py` sets `jax = None`
but then unconditionally calls `jax.nn.sigmoid(...)` inside `loss_fn` at
line 251. The `_HAS_JAX` guard at line 337 fires too late, after
`loss_initial = float(loss_fn(...))` at line 331 has already crashed.
After `pip install jax jaxopt` the tests pass; the bug is latent only on
fresh CI/sandbox environments without JAX.

## End-to-End Pipeline Status

| Stage | Working? | Evidence |
|---|---|---|
| Build ProductionPlan via production-solver | YES | 22/22 tests pass; `validate_ok=True` on synthetic Emma |
| Pass ProductionPlan to hybrid_optimizer.optimize() | **NO** | `AttributeError: 'ProductionPlan' has no attribute 'pigment_lookup'` (auditor ran this) |
| Apply plate-objective composite_loss inside hybrid-optimizer | **NO** | Zero call sites; hybrid-optimizer has its own loss function |
| Render hybrid output via v3 cell-zone-renderer | UNTESTED | renderer needs `Plate` + `CellZone` (Polygon), hybrid emits SolvedPlate (inked_mask). Adapter needed. |
| Score via the 6 validators | YES (via hybrid's `_run_validators`) | hybrid wires v3 validators; on synthetic Emma 3/6 gates fail (plate_not_composite, jigsaw_separation, final_match) |
| Generate acceptance harness sheet | YES (in isolation) | example-harness runs in 0.6s on v13 baseline; row 3/4 are placeholders because solver doesn't emit plate previews |
| Full pipeline E2E | **BROKEN** | Cannot pass production-solver → hybrid-optimizer without a hand-written adapter |

The "alternating-optimization architecture" the audit doc mandates is
present **as 4 disconnected islands**, not as a pipeline.

## CIEDE2000 Correctness

**Sharma 2005 Table I — all 34 pairs pass to better than 1e-4 ΔE.**

Worst-case deviation: pair 23 (neutral gray + small chromatic shift):
expected 1.0000, got 1.000049, deviation **4.9e-5** (well under 1e-4 budget).

Implementation in `plate-objective/delta_e.py` is correct on every gotcha
the design doc called out:
- 4-quadrant `atan2` used for hue (verified by Sharma pairs 27-30)
- Signed ΔC' and ΔH' preserved (pairs 7-8 asymmetry)
- Mean-hue boundary at |h'1-h'2|>180° (Eq. 14, pairs 1-3)
- Hue-diff sign at 180° (Eq. 10, pairs 15-18)

This is the load-bearing correctness gate and it holds.

## Validator Behavior on v13 Failing Baseline

Audit doc claimed: "plate_not_composite rejected 7 of 12 alpha planes" on
the 2026-05-17 thorough run. The plan dir
`/srv/woodblock-share/chuck-clean-outputs/2026-05-17_v3-audit-thorough-main/`
does NOT contain per-plate PNG dumps (only cumulative pulls + final). So the
"7/12" can't be reproduced from the artifact on disk. What CAN be tested:

Running v3's `plate_not_composite.score()` on the 12 cumulative-pull PNGs
(treating them as if they were plates — which is exactly the v13 failure
mode) yields **11/12 fail**:
- pull_01: badness=0.367 → pass (it's still mostly washi white)
- pulls_02..12: badness 0.98–1.00 → ALL FAIL

The validator correctly identifies composite-shaped inputs as bad plates.
Direction is `badness > 0.6 → REJECT`. v3 tests 15/15 pass independently.

The audit-doc-claimed 7/12 rate is reproducible only with the alpha planes
the audit author had access to (not in the public artifact dir). The
validator itself behaves correctly.

## Opus Vision Benchmark Reality Check

**Benchmark FAILED to produce usable data.** `bench_output/bench_results.json`:
- `n_images_scored: 0`
- `n_images_skipped: 10`
- All 10 images failed with the same error: `claude -p exited 1; stderr='Error: Input must be provided either through stdin or as a prompt argument when using --print'`
- `cost_usd_total: 0.0`
- `routing_decision.global_route: "mediapipe"` — but **because every call errored**, not because Jaccard fell short of 0.95.

The routing decision happens to land on the right answer (mediapipe, the
safe default) but the JUSTIFICATION is "Opus interface broke 10/10 times,"
not "Opus measured below threshold." That's not the gate the audit
mandated.

The 7 unit tests (which mock `claude -p`) pass. The end-to-end benchmark
DOES NOT. The `_vision_build_argv` monkey-patch in
`opus_cell_id_extractor.py` strips `--disallowedTools` and inserts
`--allowedTools Read` — that argv mutation appears to have broken the
prompt-passing path (the error is "Input must be provided either through
stdin or as a prompt argument," implying the trailing positional prompt
got swallowed).

**The benchmark needs to be re-run with a fixed claude-p transport before
the routing decision is meaningful.** As of audit time, the data is fake.

## Audit Overrides Compliance

| Override | Honored? | Evidence |
|---|---|---|
| Production-shaped solver from start (NOT incremental refactor of v13) | YES (in isolation) | production-solver builds ProductionPlan with adaptive 20-35 plates from input, NOT a v13 12-impression expansion |
| Adaptive 24-30 plates, not fixed 12 + post-hoc | YES | `estimate_plate_count` clamps to [20, 35]; synthetic Emma -> 35 |
| Alternating optimization (NOT pure JAX α-maps) | PARTIAL | hybrid-optimizer has Stage2/3/4/5 structure but Stage 3 still operates on full alpha-compose in Lab, not on jigsaw-mask-anchored topology updates |
| JAX is ONE stage, not architecture | YES | hybrid-optimizer.optimize() is pure Python orchestrator; JAX runs only inside Stage 3 |
| ΔE_76 in JAX loss + ΔE_2000 in validators | YES | `jax_continuous_solve._delta_e_76` in solver; v3 `final_match.py` uses skimage `deltaE_ciede2000`; plate-objective uses delta_e_76 in `final_image_loss` and delta_e_2000 only for ablation/Sharma-test |
| No deferred work in V1 (calibration/edition/multi-image/t2/t3) | YES | Only comments mentioning V2; no shipped calibration/t2/t3 code |
| Pair ablation + regional specificity (audit §2) | YES (function-level) | `load_bearing_pair_penalty` in plate-objective + `load_bearing_singleton_penalty` exist with `_is_concrete` guard. BUT not wired into hybrid-optimizer's loss. |

The architecture-level overrides were honored module-by-module; the
INTEGRATION the audit doc demanded was not.

## Critical Bugs Found (RANKED BY SEVERITY)

1. **[CRITICAL] End-to-end pipeline does not compose.** ProductionPlan
   from production-solver cannot be passed to hybrid-optimizer's
   `optimize()`. Three incompatible `Plate` types across three modules.
   plate-objective's `composite_loss` is not called by hybrid-optimizer.
   The "build sequence Phase 3-4" of the design doc is NOT realized.

2. **[CRITICAL] Opus vision benchmark produced zero usable data points.**
   All 10 claude-p calls failed identically due to the
   `_vision_build_argv` monkey-patch eating the positional prompt
   argument. The "global_route: mediapipe" verdict is by-error-default,
   not a measured threshold decision. The audit's mandated benchmark
   (Jaccard ≥ 0.95 on 10 annotated overlays) was not actually performed.

3. **[CRITICAL] hybrid-optimizer's synthetic-Emma run fails 3 of 6
   validator gates** (`plate_not_composite: false`, `jigsaw_separation:
   false`, `final_match: false`). delta_e_mean = 26.59 (worse than v13's
   12.086). Agent's NOTES handwaves this as "synthetic palette mismatch"
   but provides no real-Emma evidence. The solver may or may not work on
   real data — there is no proof.

4. **[IMPORTANT] hybrid-optimizer crashes when JAX is missing.**
   `jax_continuous_solve.py:251` calls `jax.nn.sigmoid` unconditionally;
   the `_HAS_JAX` guard fires only after the crash site. Fragile on any
   fresh CI/sandbox. Fix: gate `loss_fn` itself behind `_HAS_JAX` or
   replace `jax.nn.sigmoid` with a numpy-fallback expression.

5. **[IMPORTANT] hybrid-optimizer Stage 4 morphology reports "mean shrink
   100.00%"** in the validator-passing test — every plate is being
   completely consumed by the morphology repair. Either the synthetic test
   inputs are degenerate (likely) or the repair is too aggressive (also
   plausible). Either way, "100% shrink" is a smell that warrants a
   real-Emma run before V1.

6. **[IMPORTANT] example-harness validators are proxies, not the real
   v3 validators.** NOTES.md acknowledges this but the cheap proxies are
   what gate "human eyeball" sign-off. Must be wired to the real
   `run_all_validators` before V1.0 ship.

7. **[IMPORTANT] plate-objective `composite_loss` only `jax.grad`-safe
   when load-bearing penalties are SKIPPED.** The `_is_concrete` guard
   in `load_bearing_singleton_penalty` / `_pair_penalty` makes these
   terms no-op inside an outer trace — they are diagnostic-only and
   silently return 0 when wrapped by an optimizer. So the load-bearing
   pressure does NOT actually push the JAX solver. It only runs in the
   outer Python loop, where there's no gradient.

8. **[IMPORTANT] hybrid-optimizer/morphology_repair.py:135** uses
   skimage `remove_small_objects(min_size=...)` — deprecated since
   skimage 0.26; will hard-break in skimage 2.0.

9. **[IMPORTANT] `underlayer_reversal_check` is synthesized to "passes"
   when masks exist.** hybrid-optimizer's `_run_validators` patches the
   v3 validator's output to `passes=True` based only on "all_have_mask".
   This bypasses the actual mirror/registration check (which would need
   the SVG export). The validator IS in the pass-count denominator, so
   reporting "5/6 gates passed" can be misleading.

10. **[MINOR] Missing pyproject/requirements declarations.** jax, jaxopt,
    opencv-python-headless, matplotlib are needed by the four agents but
    not declared anywhere in v4-build. Discoverable only by running the
    code.

11. **[MINOR] plate-objective `_PNC_GOOD_THRESHOLD = 0.40` but docstring
    says "0.2 safety margin" / "below 0.6"** — code and docstring disagree
    on which threshold is in use. The actual constant (0.40) matches v3
    validator's `badness > 0.6 → good_score < 0.4`, so behavior is
    consistent — but the documentation is wrong.

12. **[MINOR] plate-objective and opus-vision-bench shipped no NOTES.md
    within the 30-minute audit window.** Both have functional code +
    tests, but the build contract (NOTES.md + persisted memory) was
    half-done by deadline.

## Recommended Action

**STOP and fix integration before any further build work.** Specifically:

1. Write an explicit `production_plan_to_optimizer_input(plan, cell_graph_dict)
   -> ProductionPlanInput` adapter, OR collapse the two modules onto a
   single ProductionPlan dataclass. The current two-headed schema is the
   primary blocker.
2. Wire plate-objective's `composite_loss` into hybrid-optimizer's Stage 3
   JAX loss function (not the hand-rolled `_build_loss_fn`). The
   Phase 3 of the design doc requires plate-organization terms in the
   objective; right now they're in a different file the optimizer never
   imports.
3. Fix the `_vision_build_argv` argv-stripping bug in opus_cell_id_extractor
   so claude -p calls don't fail with "Input must be provided." Re-run
   the 10-image benchmark and produce real Jaccard numbers.
4. Run hybrid-optimizer on a real Emma input (not the noise-modulated
   sin/cos synthetic) and report validator scores. The "ΔE 26 is the
   synthetic ceiling" excuse needs to be replaced with measured-on-Emma
   numbers.
5. Patch the `jax.nn.sigmoid` unconditional-call bug so the optimizer
   doesn't crash on JAX-less environments.
6. Add a `requirements-v4-build.txt` listing jax, jaxopt, opencv-python-
   headless, matplotlib so the next audit doesn't have to fix the env.

Once the adapter exists and the Opus benchmark is real, a second
end-to-end audit can decide whether to ship V1.0 digital.

## Honest Verdict

V4-build shipped **4 of 5 NOTES.md + 5 of 5 working code modules** but the
modules are 4 disconnected islands, not the alternating-optimization
pipeline the audit doc demanded. Each agent built carefully and tested in
isolation; nobody built the seams between them. The hybrid-optimizer
literally admits it ran "self-contained" because the production-solver
"had not yet produced its deliverable." That is the failure mode the
swarm orchestration was supposed to prevent. Add the CIEDE2000
implementation passing 34/34 Sharma pairs (the one correctness gate that
absolutely had to land), the Opus benchmark producing nothing usable, and
the synthetic-Emma run failing 3 of 6 gates, and the honest read is: V1.0
is **not** ready to integrate. The components are 70% there. Integration
is 0% there. Expect one more round of build work — focused on
adapters/contracts and a real Emma validation run — before any V1.0
digital-proof claim is credible.
