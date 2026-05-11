# 0002 — Overlay-not-mixing: 3-tier forward-render hierarchy

Status: accepted (2026-05-11)
Authority: `research-v23-mcp-user-addendum-v4.md` (binding)

Mokuhanga is overwhelmingly **overprint glazing** (wet B printed on dry A; light path: surface → B → A → paper → A → B → eye; K-M two-flux recursion `R_total = R_top + (T_top² · R_bottom) / (1 − R_top · R_bottom)`). Mixbox models **palette mixing** (wet + wet → one mass, applied once; 7-D latent lerp). The user's verbatim correction:

> "again integrate these tools into the mcp server and the plan so it knows that its overlayer printing not mixing"

The earlier `research-v23-overlap-math.md` §1 picked Mixbox lerp (option A) as the single forward render. That is incorrect physics for the dominant case. v23 must not present Mixbox output as "this is what your print will look like" without qualification.

## Decision — 3 tiers, dispatched at solve time

| Tier | Engine | Status | When active |
|---|---|---|---|
| **T1** | Mixbox 7-D latent lerp + ink-loading (`forward_render.py`, ~110 LOC) | ships v23 | default; generic 13-Mixbox palette; stacks ≤ 3 |
| **T2** | Empirical 2-layer LUT from artist's swatch sheet (`forward_render_empirical.py`, ~250 LOC) | v23.1 | once `upload_swatch_overprint_matrix(csv_path)` has built the LUT |
| **T3** | K-M two-flux recursion with 8λ (K, S) per-pigment fit (`forward_render_spectral.py`, ~400 LOC) | v24 | spectral fit available + stack > 3 |

`get_render_tier(plan_id)` MCP tool returns the active tier. Selection is data-driven (presence of calibration artifacts), never a user toggle. New divergence flag `MIXING_CLAIMED_AS_OVERPRINT` fires if any tool docstring or output string presents tier-1 output without the `as if pre-mixed` qualifier.

## Alternatives considered

**Ship T1 only, document the limitation in a README footnote.** Rejected because absolute ΔE shift of 4–8 on stacks > 3 is large enough to mislead the artist about pigment family choice. The whole MVP framing is "plausible plan that *reduces reconstruction error under this model*" — the model has to be honest about what it models.

**Ship T3 day-1 with hand-fit spectral data.** Rejected because measured 8λ (K, S) under D65 + ColorChecker is a weeks-long calibration task per pigment; v23 must ship before that work completes. T3 is the v24 ship gate.

## Trade-off accepted

- **Gained:** physical honesty; clear upgrade path (calibration unlocks T2, spectral fit unlocks T3); `compare_render_tiers(plan_id)` becomes the artist-facing reveal that shows "Mixbox thinks X, your actual pigments do Y"; `WB-LANG-02` lint mechanizes the qualifier.
- **Lost:** day-1 absolute color is ΔE 4–8 off on deep stacks. Mitigated by shipping the qualifier sentence verbatim inside `export_print_plan`'s generated `recipe.md` ("Note on color simulation: this plan was rendered using Mixbox, which models pigments **pre-mixed in a well** before application... Expect directional accuracy but absolute color shifts of ΔE 4–8 vs. the simulated composite, especially on stacks deeper than 3 impressions.").

## Consequence

The S5 inverse solver still uses T1 internally for gradient flow (T3 K-M recursion is differentiable but adds Jacobian cost not justified day-1). T3 enters as **evaluation-only oracle** in v24. CONTEXT.md gained Overprint / Mixing / Glazing / Render tier glossary entries on 2026-05-11 (commit `e251d71`).
