# 0004 — Day-1 MCP tool surface capped at 11

Status: accepted (2026-05-11)
Authority: `research-v23-mcp-user-addendum-v3.md` fix 2 (binding)

The reconciled v2 tool surface (`research-v23-mcp-plan-v2.1.md` §10) catalogs **33 tools across 7 categories**. The user's verbatim instruction:

> "reduce day-1 tool surface to golden-path tools"

> "Net: this is the right architecture for one-image input. It just needs to be a little less absolute about the math and a little smaller for the first build."

## Decision — ship these 11 day-1, defer 22 to v23.1

| Tool | Role |
|---|---|
| `ingest_reference_image` | every flow starts here |
| `analyze_image` | measurable features feed solver + template pick (no subject_label per fix 5) |
| `build_hue_family_map` | inspection + template-pick input |
| `propose_stack` | the actual product (S1–S10 solve) |
| `inspect_plan` | how Opus reads results |
| `alternative_stacks` | top-N exploration is part of the golden path |
| `compare_plans` | Opus needs to diff alternatives |
| `simplify_masks_for_carving` | post-solve topology repair pass (ADR-0005 home) |
| `score_candidate_stack` | 5-component breakdown lets Opus reason beyond ΔE |
| `generate_print_recipe_report` | first-class output |
| `export_print_plan` | ZIP + bundled `recipe.md` in one call |

**Deferred to v23.1:** `pin_region`, `merge_impressions`, `merge_impressions_by_hue_family`, `split_impression`, `adjust_pull_groups`, `forward_render` / `simulate_candidate_stack`, `score_stack_delta_e`, `capture_swatch`, `fit_pigments`, `apply_calibration`, `list_calibrations`, `inspect_calibration`, `get_pigments`, `get_emma_priors`, `get_defaults`, `solver_telemetry`, `dE_at`, `pigment_at`, `list_sessions`, `purge_session`, `set_session`, `current_session`, `export_svg`, `export_block_svgs`, `generate_carve_order`. None block the golden image → plan → recipe + ZIP loop.

## Alternative considered

**Ship all 33 tools day-1 with skeleton stubs returning `WoodblockError(severity="degraded")`.** Rejected because skeleton tools surface in the MCP catalog as available, and Opus will call them on the assumption that they work. The model cannot tell from the tool descriptor that a stub returns a degraded error — it just retries with adjusted args. The 22 deferred tools therefore must not exist in the namespace until they have a real implementation behind them.

**Ship 5 tools day-1 (ingest, analyze, propose, inspect, export) — the absolute minimum.** Rejected because it omits `alternative_stacks` + `compare_plans` (Opus's exploration loop), `simplify_masks_for_carving` (the home for ADR-0005's post-solve topology repair), and `score_candidate_stack` (addendum-v3 fix 4 — the 5-component breakdown that lets Opus reason beyond ΔE). Cutting those breaks the user's stated workflow.

## Trade-off accepted

- **Gained:** smallest defensible surface that proves the architecture end-to-end on the 17-fixture corpus before HITL / calibration / session-juggling complexity lands. Each day-1 tool gets full 7-section UX docstring + structured `WoodblockError` envelope without bandwidth pressure.
- **Lost:** HITL refinement (pin/merge/split) requires v23.1 wait. Calibration upload (T2 render tier per ADR-0002) requires v23.1 wait. Multi-session management (`list_sessions` etc.) requires v23.1 wait. Document workaround: single active session per chat, `WB_DATA_DIR` defaults to `~/.woodblock/v23/`.

## Consequence

Realistic milestones (addendum-v3 fix 6): D23 = 11 tools wired + 3 corpus golden produces ΔE 2× ballpark; D30-35 = Tier-1 corpus 5/5 + minimal viewer + optional calibration upload (T2 unlock); D60 = full 33-tool surface. Tool descriptors for the 22 deferred tools must not be registered with FastMCP day-1.
