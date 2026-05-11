# 0005 — Topology rules 6+7 move out of the optimizer

Status: accepted (2026-05-11)
Authority: `research-v23-mcp-user-addendum-v3.md` fix 1 (binding)

The v2 plan §7 originally placed all 7 underprint-rule terms inside the differentiable JAX loss:

```
loss(α_raw) = 8 base terms + 4 underprint-rule terms (rules 1+3+4 collapsed, 2, 5, 6, 7)
```

The user's verbatim direction:

> "move island/topology constraints out of optimizer into scoring + repair"

Rules 6 (no-tiny-hidden-islands) and 7 (few-broad-over-many-tiny) are not naturally differentiable: they are discrete geometric properties of the final pixel-quantized mask. Encoding them as soft loss terms made the optimizer simultaneously chase ΔE fidelity and discrete topology — slower, less stable, and dishonest about what differentiable math can enforce.

## Decision — 5 in-optimizer, 2 post-solve

**Inside the differentiable solver (KEEP):**

- Rule 1 — skin-tone support (family-anchored)
- Rule 2 — support-larger-than-visible (capped 4×)
- Rule 3 — cool-under-shadow (ΔE improvement gate)
- Rule 4 — warm-under-warm (same family)
- Rule 5 — detail-covers-support (soft order penalty)

After collapsing rules 1+3+4 (they share family-aligned-support shape), these become **3 differentiable loss terms**: `L_family_aligned_support`, `L_support_envelope`, `L_detail_covers_support`. Net optimizer loss = 8 base + 3 rule terms = **11 active terms**.

**Post-solve topology scoring + morphological repair (MOVE):**

- Rule 6 — no-tiny-hidden-islands → `topology_score(plan)` counts hidden islands < `min_island_px`, measures mean hidden-island area. Violation > threshold triggers an automatic `morph_open + morph_close` repair pass and re-scores.
- Rule 7 — few-broad-over-many-tiny → post-solve impression-count vs hidden-area-fraction ratio metric. Poor ratio surfaces as a `merge` suggestion in `simplify_masks_for_carving` output.

Home for both: `backend/services/v23/stages/s8_carveability.py` (5-step morphology) + `simplify_masks_for_carving` MCP tool (ADR-0004's day-1 list).

## Alternative considered

**Keep all 7 rules in the loss, raise their weights until they dominate.** Rejected: weight tuning converged to a regime where rules 6+7 starved rules 1+3+4 of gradient. The `STRATEGY_MISMATCH` and `M_TOO_LOW` divergence flags fired frequently because the optimizer was choosing trivially few impressions to avoid the tiny-island penalty.

**Add a discrete projection step after every L-BFGS iteration (project-then-step).** Rejected: 4-level pyramid × 360 total iters × per-iter discrete projection blew the wall-time budget for `solve_profile=fast` (60 s). Post-solve repair runs once at the bottom pyramid level, ~50 ms.

## Trade-off accepted

- **Gained:** optimizer is simpler + more honest about what differentiable math can enforce. The 3 in-loss rule terms align cleanly with the family-anchored Emma priors (`research-v23-emma-priors.md` §9). Topology violations are visible as a separate `topology_score` field in the manifest, so Opus can decide whether to accept or re-solve.
- **Lost:** the solver no longer "knows" that tiny hidden islands are bad — it can propose them, and the post-solve repair has to clean them up. If the repair pass changes the mask shape enough, ΔE re-measurement may drift from the solver's reported value. Mitigation: the manifest carries both `reconstruction.dE_mean` (post-repair) and `solver_telemetry.dE_at_convergence` (pre-repair) — Opus reads both.

## Consequence

Net loss term count: **11 in optimizer (8 base + 3 rules) + 2 post-solve scores (topology, count-vs-hidden-area)**. New divergence flag `TOPOLOGY_REPAIR_RAN` fires in `Plan.warnings[]` when the repair pass altered any mask. `simplify_masks_for_carving` becomes a day-1 tool (ADR-0004) specifically to expose this stage to Opus on demand.
