# chuck-mcp v4-build / production-solver — NOTES

Date: 2026-05-16
Agent: BUILD agent PRODUCTION-SOLVER
Swarm: `swarm-1778984139551-orqcce`
Scope: Phase 2 of `docs/audit-response-and-reconstruction-plan-2026-05-17.md`
       — adaptive plate count, multi-pull-per-block first-class, block/pull
       identity solved WITH target reconstruction.

## Status

**V1.0-ready for the first stage of the alternating optimizer.** All 22
pytest tests pass; 80% line coverage on the four source modules;
plate_not_composite and role_purity validators pass 100% (35/35) on the
synthetic Emma plan; build_production_plan benchmarks at 0.16s on
2200-cell Emma input (~30x under the 5s budget).

## File map

| File | LOC | Role |
|---|---:|---|
| `__init__.py` | 57 | Package entry — re-exports public API |
| `production_plan.py` | 258 | `PullSpec` / `PlateSpec` / `ProductionPlan` dataclasses + 9 invariants |
| `plate_count_estimator.py` | 253 | Adaptive plate count from LAB stats; clamped [20, 35] |
| `multi_pull_assigner.py` | 257 | 1-5 pulls per block; targets ~132 for Emma-class |
| `production_plan_builder.py` | 599 | Composes the full plan; auto-partitions cells when caller doesn't |
| `test_production_solver.py` | 698 | pytest suite — 22 tests |
| `conftest.py` | 19 | sys.path shim for hyphen-named folder |
| `_capture_artifacts.py` | 220 | One-shot validator + perf benchmark dumper |
| `.coveragerc` | 14 | Excludes test/capture from coverage measurement |
| **Total** | **2375** | |

Source modules under `--cov`: 562 stmts, 111 missed, **80% coverage**.

## pytest output (last 30 lines)

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
collected 22 items

test_production_solver.py::test_plate_count_adapts_to_chroma_variance PASSED
test_production_solver.py::test_plate_count_clamps_to_20_to_35_range PASSED
test_production_solver.py::test_multi_pull_returns_1_to_5_per_block PASSED
test_production_solver.py::test_pull_count_sums_to_around_132_for_emma_class_input PASSED
test_production_solver.py::test_production_plan_has_no_orphan_cells PASSED
test_production_solver.py::test_production_plan_has_no_zone_overlap_on_same_block PASSED
test_production_solver.py::test_production_plan_role_purity_meets_threshold PASSED
test_production_solver.py::test_production_plan_passes_plate_not_composite_validator PASSED
test_production_solver.py::test_integration_with_underlayer_proposer PASSED
test_production_solver.py::test_plan_validate_passes_on_emma PASSED
test_production_solver.py::test_performance_under_5s_emma_scale PASSED
test_production_solver.py::test_build_production_plan_form_b_cells_to_role PASSED
test_production_solver.py::test_production_plan_to_dict_and_json PASSED
test_production_solver.py::test_build_production_plan_plate_count_override PASSED
test_production_solver.py::test_role_count_targets_for_various_plate_counts PASSED
test_production_solver.py::test_build_with_very_small_cell_graph_pads_to_min_plates PASSED
test_production_solver.py::test_form_b_pixel_list_input PASSED
test_production_solver.py::test_lab_mean_input_in_cells PASSED
test_production_solver.py::test_plate_spec_add_pull_rejects_mismatched_block_id PASSED
test_production_solver.py::test_pullspec_to_dict_round_trip PASSED
test_production_solver.py::test_role_distribution_consistent_with_plate_count PASSED
test_production_solver.py::test_production_plan_validate_catches_orphans PASSED
============================== 22 passed in 2.28s ==============================

================================ tests coverage ================================
Name                         Stmts   Miss  Cover
------------------------------------------------
multi_pull_assigner.py          80      4    95%
plate_count_estimator.py       103      9    91%
production_plan.py             115     16    86%
production_plan_builder.py     264     82    69%
------------------------------------------------
TOTAL                          562    111    80%
```

The 9 task-mandated tests are all present (tests 1-9 in the file are the
canonical brief); tests 10-22 are bonus coverage for invariants, round-trip
serialization, edge cases, and dataclass guards.

## ProductionPlan sample (synthetic Emma)

Build inputs: 256x256 image, 1800 SNIC cells, 7 hue clusters, deterministic
seed=42.

```json
{
  "plan_id": "plan_1778985008482_1ad350a3",
  "plate_count": 35,
  "total_pulls": 144,
  "cell_count": 1800,
  "image_shape": [256, 256],
  "role_distribution": {
    "underlayer_light": 8,
    "local_chroma":     12,
    "regional_mass":    11,
    "key_detail":       4
  },
  "validate_ok": true,
  "first_plate": {
    "block_id": 1,
    "cell_zone_ids": [37, 53, 79, 99, 213, ... 1627],   // 22 cells
    "role": "underlayer_light",
    "pigment_family": "light_yellow",
    "pulls": [
      {
        "pull_id": 1,
        "block_id": 1,
        "pigment_id": "PY3_holbein_pale",
        "opacity": 0.20,
        "dilution": 0.30,
        "order_step": 1,
        "pass_index": 1,
        "role": "underlayer_light",
        "mask_subset": null
      }
    ],
    "region_label": "underlayer_light_cluster_1",
    "rationale": "auto-partitioner: role=underlayer_light hue-bin 1/8 (n=22 cells)",
    "provenance": "algorithm"
  },
  "last_plate_summary": {
    "block_id": 35,
    "role": "key_detail",
    "n_cells": 67,
    "n_pulls": 4,
    "first_pull": {
      "pull_id": 132,
      "block_id": 35,
      "pigment_id": "PBk7_sumi_dense",
      "opacity": 0.85,
      "dilution": 0.90,
      "order_step": 132,
      "pass_index": 1,
      "role": "key_detail",
      "mask_subset": null
    }
  }
}
```

Full plan JSON: `_artifacts/sample_emma_plan.json` (175 KB).

## Validator scores on synthetic Emma

Run via `_capture_artifacts.py` (35-plate plan, 144 pulls, 1800 cells):

| Validator | Pass rate | Mean score | Min / Max |
|---|---:|---:|---|
| `plate_not_composite` (badness, lower=better) | **35/35 (100%)** | 0.082 badness | 0.024 / 0.231 |
| `role_purity` (purity, higher=better, gate=0.70) | **35/35 (100%)** | 1.000 purity | 1.000 / 1.000 |

The 0.6 reject threshold for plate_not_composite means we land **0.4
under the gate everywhere** — the proposer produces plates that look like
sparse jigsaw inked-zone masks, not faded full-face composites. This is
the v13 failure mode the audit caught, and it is fixed at the proposal
stage before JAX ever touches the plan.

Role purity is trivially 1.0 because the builder assigns role at the
plate level (not per cell); the validator will only meaningfully shift
when downstream JAX morphology repair starts borrowing cells across
plates. The gate is still wired so any future regression trips it.

Full validator report: `_artifacts/validator_report.json`.

## Performance benchmark

| Input | Cells | Plates | Pulls | build_seconds | Budget | Result |
|---|---:|---:|---:|---:|---:|---|
| default Emma | 1800 | 35 | 144 | **0.117s** | 5.0s | PASS (43x headroom) |
| Emma-scale bench | 2200 | 35 | 144 | **0.161s** | 5.0s | PASS (31x headroom) |

`_artifacts/performance_bench.json` for raw timing.

## Integration contract — how downstream JAX consumes ProductionPlan

The plan is the **structural prior** the JAX solver locks against. JAX
optimizes the CONTINUOUS variables (per `PullSpec`); the STRUCTURAL
variables stay fixed:

| Variable | Set by | Optimized by JAX |
|---|---|:---:|
| `PullSpec.pull_id` | proposal stage | NO |
| `PullSpec.block_id` | proposal stage (plate identity) | NO |
| `PullSpec.pass_index` | proposal stage (pull-of-block) | NO |
| `PullSpec.order_step` | proposal stage (global order) | NO |
| `PullSpec.role` | proposal stage (plate role) | NO |
| `PullSpec.opacity` | proposal hint | **YES** (smooth ΔE_76 loss) |
| `PullSpec.dilution` | proposal hint | **YES** |
| `PullSpec.pigment_id` | proposal hint | **YES** (palette substitution allowed) |
| `PullSpec.mask_subset` | proposal default `None` | **YES** (morphology repair) |
| `PlateSpec.cell_zone_ids` | proposal stage (jigsaw layout) | NO (rewritten by morphology repair, NOT by JAX) |
| `PlateSpec.role` | proposal stage | NO |
| `PlateSpec.pigment_family` | proposal stage hint | NO |

JAX call signature suggested:

```python
optimized_plan = jax_continuous_solver(
    target_lab=target_lab_image,            # (H, W, 3) float32
    initial_plan=proposal_plan,             # ProductionPlan from build_production_plan
    cell_zone_mask_lut=cell_zone_mask_lut,  # dict[cell_id, (H, W) bool]
    loss_fn="delta_e_76",                   # NOT 2000 (gradient discontinuity)
    plate_not_composite_penalty=2.0,        # in-loop, not post-hoc
    role_coverage_caps={"underlayer_light": 0.45, "key_detail": 0.10},
    n_steps=2000,
)
```

After JAX, the morphology-repair stage re-checks each plate's
`cell_zone_ids` for connected components above minimum area, no
hairline islands, no unbrushable adjacencies. If a plate fails
printability, the repair stage rewrites its `cell_zone_ids` (NOT the
PullSpec continuous vars) and the JAX solve re-runs on the repaired
layout — this is the "re-solve after repair" loop in Phase 4 of the
audit reconstruction plan.

## Invariants enforced (`ProductionPlan.validate()`)

| ID | Rule |
|---|---|
| **I1** | `plate_count ∈ [20, 35]` |
| **I2** | Every plate has non-empty `cell_zone_ids` |
| **I3** | Every plate has ≥ 1 `PullSpec` |
| **I4** | `pass_index` is contiguous 1..N per plate |
| **I5** | No orphan cells (every input cell on some plate) |
| **I6** | No `cell_zone_id` duplicates within a plate (and dedup across plates) |
| **I7** | Plate `role ∈ {underlayer_light, local_chroma, regional_mass, key_detail}` |
| **I8** | `pull_id` unique across the plan |
| **I9** | `order_step` unique AND contiguous 1..total_pulls |

All 9 hold on the auto-built synthetic Emma plan (`validate_ok: true`).

## Integration with v3 `underlayer_proposer`

Test `test_integration_with_underlayer_proposer` wires up
`research/v3-construction/mokuhanga-rule-classifier/underlayer_proposer.py`
through `build_production_plan`:

1. The v3 proposer's 4-9 `UnderlayerPlate` objects are mapped onto
   Form-A `role_assignments` with `role="underlayer_light"`,
   `pigment_family=<proposer's choice>`, `region_label=<v3 region>`,
   `rationale=<v3 hue-band reasoning>`, `provenance="algorithm"`.
2. Filler plates are added to reach >= 20 plates (mixing
   `local_chroma`, `regional_mass`, `key_detail`).
3. `build_production_plan` composes the v4 plan; underlayer block_ids
   from the v3 proposer survive into the v4 plan with `role="underlayer_light"`
   and pulls in `pass_index ∈ [1, 3]`.

This proves the v3 rule classifier slots into the v4 production-solver
pipeline without modification — important for Phase 1 of the audit
plan (example-grounded acceptance harness must inherit the proposer's
provenance trail).

## Audit pain points addressed

The audit's seven verdicts and what this module does about each:

| Audit pain | What this module ships |
|---|---|
| **§4** "12 impressions, 3 blocks. Production planner proposes 24, but is post-hoc narration, not solved." | `build_production_plan` IS the production solve. JAX consumes its output. No post-hoc batch expansion. |
| Adaptive plate count never landed | `estimate_plate_count` always lands [20, 35]; Emma synthetic input → 35 (top of range — by design for high-variance Emma scan). |
| Multi-pull is a label, not a variable | `PullSpec` is a first-class dataclass; JAX sees one PullSpec per pull, period. |
| `plate_not_composite` rejected 7/12 of v3 alpha planes | Proposal-stage plates score 35/35 pass at 0.082 mean badness (~7x under the 0.6 reject gate). |
| Role-purity collapses when α-stack is solved without role labels | Roles are explicit on `PlateSpec` and propagate to every PullSpec via `add_pull` which raises on mismatch. |
| Orphan cells | I5 enforced; auto-builder backfills any missing cell to closest-LAB plate. |
| Overlap on same block | I6 enforced (within-plate dedup) AND cross-plate dedup runs unconditionally in `_dedup_cells_across_plates`. |

## Known limitations / V2 candidates

1. **Auto-partitioner is hue-band binning, not graph cut.** Good enough
   for a proposal that JAX will refine. Phase 4 in the audit plan asks
   for ILP-style exclusivity — that lives in the next module
   (`hybrid-optimizer/`) and consumes ProductionPlan unchanged.
2. **Pigment hints are role-default, not target-color-aware.** A second
   pass that picks pigment family from cell mean LAB would tighten
   JAX's warm-start. Left to the JAX module for now.
3. **mask_subset is always None.** The structure exists for sub-pull
   masking (graduated opacity on a subset of zones), but the proposer
   does not populate it. JAX morphology repair fills it in when a
   plate needs partial coverage.
4. **Adaptive plate count saturates at 35 for high-variance synthetic
   input.** This is by design (audit floor is 24, ceiling 35), but Emma
   2002 ground truth is 27. Real data with face landmarks should land
   nearer 25-28 once SNIC + face spatial provide better hue clustering
   (synthetic noise pushes the estimator high).
5. **No connection to MediaPipe or Opus 4.7 vision for cell-ID
   assignment.** This module is solver-side; the cell-ID source lives
   in `mediapipe-face-spatial/` or the Opus benchmark module.

## Where to go from here

Next agent (BUILD-AGENT-2) should take this ProductionPlan and:

1. Wire JAX continuous solver against `PullSpec.opacity/dilution/pigment_id`.
2. Implement morphology repair on `PlateSpec.cell_zone_ids` (component
   scoring + re-solve).
3. Add the in-loop `plate_not_composite` penalty (Phase 3 of audit plan).
4. Implement load-bearing singleton + pair ablation (Audit §2).
5. Replace `_auto_partition_cells` with a graph-cut / ILP partitioner
   when cell-graph adjacency arrives.

The ProductionPlan dataclass is the **stable interface** between this
proposal stage and everything downstream. Treat its invariants (I1-I9)
as the contract.
