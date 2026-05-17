# chuck-mcp v4 — plate-objective

Phase 3 of `docs/audit-response-and-reconstruction-plan-2026-05-17.md`.

JAX-differentiable loss terms that put plate organization INTO the solver
objective. Pairs with the v3 validators (`research/v3-construction/
validators-reconstruction/`) — these run on continuous solver state during
the inner loop; the v3 validators run on rendered outputs at gate-time.

## Status

| Gate | Result |
|---|---|
| pytest | **17/17 PASS** (target was 14/14; added 3 bonus gates: symmetric ΔE_2000, checkpoint monotonicity, visualizer smoke) |
| CIEDE2000 Sharma Table I | **34/34 PASS** at <5e-5 abs diff (target was 1e-4) |
| Coverage | **95%** total (target 85%) — delta_e 100%, objective_terms 88%, visualizer 98% |
| JAX grad compat (composite_loss → ∂/∂masks) | **PASS** — no NaN, gradient finite, non-zero |
| Hot-loop perf (Emma 27 plates 256×256) | **45 ms / eval** (target <200 ms) |
| V1.0-ready | **YES** for Phase 3 |

## File map (line counts)

```
plate-objective/
├── ciede2000_test_table.py    81 lines  — Sharma 2005 Table I, 34 reference pairs
├── conftest.py                  9 lines  — pytest sys.path fixup for flat imports
├── delta_e.py                 282 lines  — ΔE_76 (JAX), ΔE_94 (JAX), ΔE_2000 (NumPy)
├── loss_visualizer.py         116 lines  — per-term loss curve plotter (headless Agg)
├── objective_terms.py         652 lines  — canonical Plate import + 9 loss/penalty terms + composite_loss
├── test_plate_objective.py    453 lines  — 17 pytest tests
└── NOTES.md                   (this file)
```

Total: 1,618 LOC code/test, no external deps beyond {jax, numpy, matplotlib}.

## pytest output

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- python3.12
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.13.0
collected 17 items

test_plate_objective.py::test_delta_e_2000_passes_all_34_sharma_pairs PASSED [  5%]
test_plate_objective.py::test_delta_e_76_is_jax_differentiable             PASSED [ 11%]
test_plate_objective.py::test_delta_e_94_is_jax_differentiable             PASSED [ 17%]
test_plate_objective.py::test_final_image_loss_decreases_when_render_approaches_target PASSED [ 23%]
test_plate_objective.py::test_plate_not_composite_penalty_zero_for_clean_plates PASSED [ 29%]
test_plate_objective.py::test_plate_not_composite_penalty_nonzero_for_v13_residual_synthetic PASSED [ 35%]
test_plate_objective.py::test_cell_exclusivity_penalty_catches_overlap     PASSED [ 41%]
test_plate_objective.py::test_role_coverage_caps_caps_dominant_role        PASSED [ 47%]
test_plate_objective.py::test_load_bearing_singleton_flags_unused_pull     PASSED [ 52%]
test_plate_objective.py::test_load_bearing_pair_flags_cancellation_pair    PASSED [ 58%]
test_plate_objective.py::test_printability_in_loop_catches_sub_mill_feature PASSED [ 64%]
test_plate_objective.py::test_composite_loss_is_jax_grad_compatible        PASSED [ 70%]
test_plate_objective.py::test_ciede2000_signs_correct_around_blue_hue_275  PASSED [ 76%]
test_plate_objective.py::test_ciede2000_mean_hue_boundary_at_180_apart     PASSED [ 82%]
test_plate_objective.py::test_ciede2000_symmetric                          PASSED [ 88%]
test_plate_objective.py::test_loss_visualizer_smoke                        PASSED [ 94%]
test_plate_objective.py::test_checkpoint_proof_loss_decreases_toward_targets PASSED [100%]

================================ tests coverage ================================
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
ciede2000_test_table.py       5      1    80%   81
conftest.py                   5      1    80%   7
delta_e.py                   77      0   100%
loss_visualizer.py           59      1    98%   93
objective_terms.py          182     21    88%   106-107, 202, 208, 242, 281, 326, 391-394, 421, 428, 482, 486, 498, 565, 625, 633, 652-655
test_plate_objective.py     189      1    99%   102
-------------------------------------------------------
TOTAL                       517     25    95%
============================== 17 passed in 11.63s =============================
```

The 14 spec-required tests + 3 bonus:

| # | Test | Spec | Status |
|---|------|------|--------|
| 1 | test_delta_e_2000_passes_all_34_sharma_pairs | required | PASS |
| 2 | test_delta_e_76_is_jax_differentiable | required | PASS |
| 3 | test_delta_e_94_is_jax_differentiable | required | PASS |
| 4 | test_final_image_loss_decreases_when_render_approaches_target | required | PASS |
| 5 | test_plate_not_composite_penalty_zero_for_clean_plates | required | PASS |
| 6 | test_plate_not_composite_penalty_nonzero_for_v13_residual_synthetic | required | PASS |
| 7 | test_cell_exclusivity_penalty_catches_overlap | required | PASS |
| 8 | test_role_coverage_caps_caps_dominant_role | required | PASS |
| 9 | test_load_bearing_singleton_flags_unused_pull | required | PASS |
| 10 | test_load_bearing_pair_flags_cancellation_pair | required | PASS |
| 11 | test_printability_in_loop_catches_sub_mill_feature | required | PASS |
| 12 | test_composite_loss_is_jax_grad_compatible | required | PASS |
| 13 | test_ciede2000_signs_correct_around_blue_hue_275 | required | PASS |
| 14 | test_ciede2000_mean_hue_boundary_at_180_apart | required | PASS |
| 15 | test_ciede2000_symmetric | bonus | PASS |
| 16 | test_loss_visualizer_smoke | bonus (coverage) | PASS |
| 17 | test_checkpoint_proof_loss_decreases_toward_targets | bonus | PASS |

## Sharma 2005 Table I — all 34 reference pairs

This is the load-bearing correctness gate for CIEDE2000. If any single
pair drifts above 1e-4, the implementation has a bug in one of the four
known pitfalls (atan2, signed ΔC'/ΔH', mean-hue boundary, hue-diff at
180°). All 34 pass at worst-case abs-diff ≤ 4.9e-5.

| pair | L1 a1 b1 | L2 a2 b2 | expected | computed | abs_diff | pass/fail |
|---|---|---|---|---|---|---|
| 1 | 50.0000, 2.6772, -79.7751 | 50.0000, 0.0000, -82.7485 | 2.0425 | 2.0425 | 0.000040 | PASS |
| 2 | 50.0000, 3.1571, -77.2803 | 50.0000, 0.0000, -82.7485 | 2.8615 | 2.8615 | 0.000010 | PASS |
| 3 | 50.0000, 2.8361, -74.0200 | 50.0000, 0.0000, -82.7485 | 3.4412 | 3.4412 | 0.000009 | PASS |
| 4 | 50.0000, -1.3802, -84.2814 | 50.0000, 0.0000, -82.7485 | 1.0000 | 1.0000 | 0.000001 | PASS |
| 5 | 50.0000, -1.1848, -84.8006 | 50.0000, 0.0000, -82.7485 | 1.0000 | 1.0000 | 0.000005 | PASS |
| 6 | 50.0000, -0.9009, -85.5211 | 50.0000, 0.0000, -82.7485 | 1.0000 | 1.0000 | 0.000013 | PASS |
| 7 | 50.0000, 0.0000, 0.0000 | 50.0000, -1.0000, 2.0000 | 2.3669 | 2.3669 | 0.000041 | PASS |
| 8 | 50.0000, -1.0000, 2.0000 | 50.0000, 0.0000, 0.0000 | 2.3669 | 2.3669 | 0.000041 | PASS |
| 9 | 50.0000, 2.4900, -0.0010 | 50.0000, -2.4900, 0.0009 | 7.1792 | 7.1792 | 0.000028 | PASS |
| 10 | 50.0000, 2.4900, -0.0010 | 50.0000, -2.4900, 0.0010 | 7.1792 | 7.1792 | 0.000037 | PASS |
| 11 | 50.0000, 2.4900, -0.0010 | 50.0000, -2.4900, 0.0011 | 7.2195 | 7.2195 | 0.000028 | PASS |
| 12 | 50.0000, 2.4900, -0.0010 | 50.0000, -2.4900, 0.0012 | 7.2195 | 7.2195 | 0.000026 | PASS |
| 13 | 50.0000, -0.0010, 2.4900 | 50.0000, 0.0009, -2.4900 | 4.8045 | 4.8045 | 0.000022 | PASS |
| 14 | 50.0000, -0.0010, 2.4900 | 50.0000, 0.0010, -2.4900 | 4.8045 | 4.8045 | 0.000025 | PASS |
| 15 | 50.0000, -0.0010, 2.4900 | 50.0000, 0.0011, -2.4900 | 4.7461 | 4.7461 | 0.000029 | PASS |
| 16 | 50.0000, 2.5000, 0.0000 | 50.0000, 0.0000, -2.5000 | 4.3065 | 4.3065 | 0.000018 | PASS |
| 17 | 50.0000, 2.5000, 0.0000 | 73.0000, 25.0000, -18.0000 | 27.1492 | 27.1492 | 0.000031 | PASS |
| 18 | 50.0000, 2.5000, 0.0000 | 61.0000, -5.0000, 29.0000 | 22.8977 | 22.8977 | 0.000008 | PASS |
| 19 | 50.0000, 2.5000, 0.0000 | 56.0000, -27.0000, -3.0000 | 31.9030 | 31.9030 | 0.000005 | PASS |
| 20 | 50.0000, 2.5000, 0.0000 | 58.0000, 24.0000, 15.0000 | 19.4535 | 19.4535 | 0.000021 | PASS |
| 21 | 50.0000, 2.5000, 0.0000 | 50.0000, 3.1736, 0.5854 | 1.0000 | 1.0000 | 0.000026 | PASS |
| 22 | 50.0000, 2.5000, 0.0000 | 50.0000, 3.2972, 0.0000 | 1.0000 | 1.0000 | 0.000027 | PASS |
| 23 | 50.0000, 2.5000, 0.0000 | 50.0000, 1.8634, 0.5757 | 1.0000 | 1.0000 | 0.000049 | PASS |
| 24 | 50.0000, 2.5000, 0.0000 | 50.0000, 3.2592, 0.3350 | 1.0000 | 1.0000 | 0.000035 | PASS |
| 25 | 60.2574, -34.0099, 36.2677 | 60.4626, -34.1751, 39.4387 | 1.2644 | 1.2644 | 0.000020 | PASS |
| 26 | 63.0109, -31.0961, -5.8663 | 62.8187, -29.7946, -4.0864 | 1.2630 | 1.2630 | 0.000041 | PASS |
| 27 | 61.2901, 3.7196, -5.3901 | 61.4292, 2.2480, -4.9620 | 1.8731 | 1.8731 | 0.000029 | PASS |
| 28 | 35.0831, -44.1164, 3.7933 | 35.0232, -40.0716, 1.5901 | 1.8645 | 1.8645 | 0.000005 | PASS |
| 29 | 22.7233, 20.0904, -46.6940 | 23.0331, 14.9730, -42.5619 | 2.0373 | 2.0373 | 0.000042 | PASS |
| 30 | 36.4612, 47.8580, 18.3852 | 36.2715, 50.5065, 21.2231 | 1.4146 | 1.4146 | 0.000022 | PASS |
| 31 | 90.8027, -2.0831, 1.4410 | 91.1528, -1.6435, 0.0447 | 1.4441 | 1.4441 | 0.000029 | PASS |
| 32 | 90.9257, -0.5406, -0.9208 | 88.6381, -0.8985, -0.7239 | 1.5381 | 1.5381 | 0.000017 | PASS |
| 33 | 6.7747, -0.2908, -2.4247 | 5.8714, -0.0985, -2.2286 | 0.6377 | 0.6377 | 0.000028 | PASS |
| 34 | 2.0776, 0.0795, -1.1350 | 0.9033, -0.0636, -0.5514 | 0.9082 | 0.9082 | 0.000033 | PASS |

Worst-case abs_diff: **0.000049** (pair 23). Spec threshold was 1e-4; we
clear that by 2×. Cross-checked at build time against
`skimage.color.deltaE_ciede2000` — same numbers to better than 5e-5.

## Sample composite_loss output on a synthetic Emma plan

Inputs: 27 plates, 256×256 working resolution, random seed 42, default
`LossWeights`, mock linear render_fn (weighted Lab sum by mask×opacity).

```json
{
  "per_term_raw": {
    "final_image":                 78.486,
    "plate_not_composite":          0.000,
    "cell_exclusivity":             3.063,
    "role_coverage_caps":           0.000,
    "role_frequency_permission":    0.000,
    "printability":                27.000
  },
  "composite_total":              94.436,
  "weights": {
    "final_image":                 1.0,
    "checkpoint_proof":            0.5,
    "plate_not_composite":         0.7,
    "cell_exclusivity":            0.8,
    "role_coverage_caps":          0.3,
    "role_frequency_permission":   0.4,
    "load_bearing_singleton":      0.2,
    "load_bearing_pair":           0.2,
    "printability":                0.5
  }
}
```

Interpretation: this random plan is dominated by `final_image` (ΔE_76 mean
≈ 78 — bad), `printability` (random binary masks have lots of sub-mill
specks), and a small `cell_exclusivity` overlap. The `plate_not_composite`
and role terms are zero because random binary masks at 15% density rarely
look like a structured composite. This is the right shape for a solver to
chew on: dominant terms guide the gradient first, then secondary penalties
kick in as the solver hits the major constraints.

## Integration contract — how Phase 4 imports + uses this module

The Phase 4 hybrid solver (`research/v4-build/hybrid-optimizer/`) will
consume this module as follows:

### 1. Import surface

```python
# Plate is the canonical union dataclass in chuck_mcp_v2.types — re-exported
# by objective_terms for convenience. The "objective" schema accepts
# positional Plate(block_id, mask, pigment_lab, opacity, role).
from chuck_mcp_v2.types import Plate                  # canonical schema
from plate_objective.objective_terms import (
    LossWeights,                   # tunable weight bundle
    composite_loss,                # single-call inner-loop loss
    # Individual terms exposed for between-step diagnostics:
    plate_not_composite_penalty_per_plate,
    cell_exclusivity_penalty,
    load_bearing_singleton_penalty,
    load_bearing_pair_penalty,
    printability_in_loop_penalty,
)
from plate_objective.delta_e import (
    delta_e_76,                    # JAX, inner loop
    delta_e_2000,                  # NumPy, validators only
)
from plate_objective.loss_visualizer import LossHistory
```

### 2. Solver step pattern

```python
def solver_step(params, target, render_fn, opt_state, step_idx):
    # ParAMS holds plate masks/pigments/opacities as JAX arrays.

    def loss_fn(p):
        plates = build_plates_from_params(p)
        return composite_loss(
            plates=plates,
            target=target,
            render_fn=render_fn,
            plate_order=p["order"],
            mill_radius_px=2,
            enable_load_bearing=False,   # disabled in hot loop
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Every K steps, run the diagnostic ablation terms in eager mode.
    if step_idx % 50 == 0:
        plates = build_plates_from_params(params)
        load_bearing = (
            load_bearing_singleton_penalty(plates, target, render_fn)
            + load_bearing_pair_penalty(plates, target, render_fn)
        )
        # Use as termination signal or auxiliary regulariser; don't
        # backprop through it.
    return params, opt_state, loss
```

### 3. Hot-loop vs diagnostic separation

* **Hot loop** (`enable_load_bearing=False`, default): 45 ms / call at
  Emma scale on CPU/CUDA, well under the 200 ms budget.
* **Diagnostic** (`enable_load_bearing=True`): runs nested `jax.grad`
  ~20 times for pair ablation; ~5-15 s per call depending on plate count.
  Use sparingly — every 50 steps in the outer loop, never inside `jax.jit`.

### 4. Validators ↔ in-loop loss alignment

| Validator (v3) | In-loop term (v4) | Difference |
|----------------|--------------------|------------|
| `plate_not_composite_score(plate_preview, final)` | `plate_not_composite_penalty_per_plate(plates, rendered)` | v3 runs on rendered wood-grain preview; v4 runs on binary mask. v4 threshold relaxed to 0.40 to compensate for binary-mask cosine-sim bias. |
| `jigsaw_separation_score` | `cell_exclusivity_penalty` | v3 checks mm separation; v4 penalises spatial overlap (more permissive — sets up easier gradient flow). |
| `final_match_score` (ΔE_2000) | `final_image_loss` (ΔE_76) | Different metric on purpose — Sharma discontinuities. |
| `proof_progression_score` | `checkpoint_proof_loss` | v3 counts ΔE_2000 > 3 over checkpoints; v4 minimises ΔE_76 sum. |
| `role_purity_score` | (none in loop — pre-loop assignment) | Role assignment happens before the JAX continuous solve (Phase 4 step 2). |
| `underlayer_reversal_check` | (none in loop — post-solve) | Pure boolean output flip — outside the differentiable path. |

### 5. Performance budget

| Operation | Time @ 256×256 / 27 plates |
|-----------|---------------------------|
| `composite_loss(hot)` | ~45 ms |
| `jax.grad(composite_loss)(hot)` | ~120 ms (extrapolated; 2.5× forward) |
| `composite_loss(diagnostic)` | ~5-15 s (nested grad) |
| `delta_e_2000` (validator, NumPy) | <1 ms per pair, vectorisable |

## Four CIEDE2000 implementation pitfalls (Sharma 2005)

Embedded in `delta_e.py` module docstring; reproduced here for cross-link:

1. **Use atan2, not atan, for hue.** atan returns [-π/2, π/2]; need full
   [0, 2π). Wrap result to [0°, 360°).
2. **Signed ΔC' and ΔH'.** Some clean implementations absolute-value these
   and silently drift on blue samples. R_T cross-term depends on signs.
3. **Mean-hue h̄' boundary case (Eq. 14).** When `|h1' − h2'| > 180°`,
   naive `(h1+h2)/2` is off by 180°. Add 360° to whichever hue is smaller
   before averaging — convention: if `h1 + h2 < 360`, add 360 to the mean;
   otherwise subtract 360.
4. **Hue-diff Δh' sign at exactly 180°** (Eq. 10). Convention: result in
   (-180°, 180°], so +180 stays +180 not -180.

If any of those four are wrong, the Sharma Table I diverges visibly on
the blue (pairs 1-6, 27-29) and gray-axis (pairs 9-16) entries.

## Known limitations / next steps

* **`plate_not_composite` threshold** — set to 0.40 in-loop because binary
  solver-state masks have higher cosine-sim than rendered plate previews.
  Re-tune after the renderer is wired (Phase 4) so in-loop and validator
  thresholds agree at 0.60.
* **`role_coverage_caps`** — cap fixed at 60% per family. Per-role tunable
  caps (e.g., `key_detail` may want 40%) deferred until we have empirical
  data on Emma plans.
* **`load_bearing_pair`** — uses single-eps finite-difference Hessian; for
  larger plans (~50 plates) consider proper jvp-of-grad. Top-K=20 default
  per spec.
* **`printability`** — uses min-pool erosion at solver resolution. Real
  CNC printability needs additional connected-component and bit-clearance
  checks; those run in the v6 morphology repair stage.
* **`checkpoint_proof_loss`** — currently mean-ΔE_76 only. Spec also wants
  expected-proof-progression structure (which checkpoints add which color
  families). That requires a renderer + Hokusai/Chuck Close reference
  proof series — deferred to integration time.

## Verified against

* JAX `0.10.0` on CUDA (WSL2, GPU backend per kernel-driver warning).
* numpy `2.4.4`.
* `skimage.color.deltaE_ciede2000` for ground-truth Sharma cross-check.
