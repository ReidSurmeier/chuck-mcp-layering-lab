# Diagnosis — v5 validator truth and outer-loop replay

Date: 2026-05-17

## User-visible symptom

The v5 overnight report claimed every block failed plate-not-composite with
`inked_area_fraction: 1.0`, while the review sheet showed sparse block masks.
The same run also reported identical Stage 3 trajectories on every outer
iteration, which made the outer loop effectively a replay.

## Reproduce

Original artifact:

```text
/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-10/
```

Historical validator report:

- plate-not-composite: `0/28`
- role-purity: pass
- proof-progression: pass
- jigsaw-separation: fail
- underlayer-reversal: fail
- final-match advisory: fail

Historical Stage 3 history repeated the same initial/final loss each outer
iteration.

## Minimise

Two focused regression seams were added:

- `research/v3-construction/validators-reconstruction/test_validator_mask_inputs.py`
  proves mask-backed validation uses white ink on black background as sparse
  geometry and prefers `inked_mask` over `plate_preview`.
- `research/v4-build/hybrid-optimizer/test_hybrid_optimizer.py`
  now includes `test_outer_loop_warm_starts_from_previous_repaired_solution`,
  which stubs Stage 3 and asserts outer iteration 2 receives previous solved
  opacity rather than the hardcoded Stage 2 default.

## Ranked hypotheses

1. **Validator input polarity/object mismatch.**
   If plate-not-composite is scoring a full wood preview instead of a binary
   **Mask**, every block will look like full coverage.

2. **Outer loop is rebuilding from Stage 2 on every iteration.**
   If Stage 3 is always initialized from fixed defaults, repeated outer
   iterations will report identical initial and final losses.

3. **SNIC merged every block into a composite.**
   If true, corrected mask-backed validation would still fail every block.

## Instrumentation and results

Rebuilt iter 10 with mask-backed `validator_plan_masked.json` and re-ran
validators:

- plate-not-composite: `28/28` pass
- role-purity: pass
- proof-progression: pass
- jigsaw-separation: `14/28` pass
- underlayer-reversal: fail
- final-match advisory: fail, dE_mean `19.31`

This falsified the SNIC-composite diagnosis and confirmed the validator-plan
bug.

Fresh iter 13 after the outer-loop fix:

- Stage 3 loss: `38.001 -> 29.535`
- Stage 3 loss: `29.535 -> 28.947`
- Stage 3 loss: `28.947 -> 28.821`
- dE_mean improved from iter 10's `19.053` to `18.778`
- gates: `3/5`

The outer loop now carries state forward. It still does not solve the image,
which means the remaining failures are topology/methodology/color, not replay.

## Fixes

- `build_validator_plan.py` maps solved blocks to sorted pull-order alpha masks
  and emits those paths as `inked_mask`.
- `run_all_validators.py` prefers `inked_mask` / `plate_mask` /
  `alpha_preview` over review previews for geometry gates and accepts path-based
  masks from JSON plans.
- `plate_not_composite.py` treats binary white-ink masks correctly.
- `alternating_loop.py` warm-starts later outer iterations from previous
  repaired masks and solved continuous variables.
- `jax_continuous_solve.py` warm-starts pigment weights when they are provided.

## Regression commands

```bash
.venv-renderer/bin/python -m pytest -q research/v3-construction/validators-reconstruction
.venv-renderer/bin/python -m pytest -q research/v4-build/hybrid-optimizer
.venv-renderer/bin/python -m pytest -q research/v5-overnight/loop-runner
```

Latest observed results:

- validator reconstruction tests: `17 passed`
- hybrid optimizer tests: `13 passed`
- loop-runner tests: `1 passed`

## Post-mortem

The architecture allowed **Review preview** images to cross a validator seam
that needed **Validator truth**. ADR-0006 now forbids preview-backed geometry
validation when an authoritative **Mask** exists. The next deepening opportunity
is to package validator-plan creation as a typed module instead of passing
loosely shaped dictionaries between research scripts.
