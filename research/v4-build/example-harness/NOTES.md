# chuck-mcp v4 — Example Harness (Phase 1)

Build agent: EXAMPLE-HARNESS in swarm `swarm-1778984139551-orqcce`.
Date: 2026-05-17.

Phase 1 deliverable from `docs/audit-response-and-reconstruction-plan-2026-05-17.md`:

> "Make a side-by-side contact sheet generator: reference proof row, current
> proof row, current block row, alpha row. Acceptance rule: if a human says
> 'this looks like slop' against the example sheet, the run fails regardless
> of dE."

## File map

| Path | Lines | Role |
|---|---|---|
| `acceptance_harness/__init__.py` | 27 | package entry, re-exports |
| `acceptance_harness/__main__.py` | 5 | `python -m acceptance_harness` glue |
| `acceptance_harness/acceptance_harness.py` | 422 | main `render_acceptance_sheet()` + helpers |
| `acceptance_harness/acceptance_result.py` | 82 | `AcceptanceSheetResult`, `PlateMetric` dataclasses |
| `acceptance_harness/cli.py` | 90 | argparse CLI + JSON / human output |
| `acceptance_harness/example_loader.py` | 135 | `/srv/woodblock-share/Examples` loader (Hokusai 8-cell split, placeholders) |
| `test_acceptance_harness.py` | 352 | 14-test pytest suite |
| `conftest.py` | 12 | pytest sys.path bootstrap |
| **Total** | **1125** | |

Sample outputs written during build:
- `sample_v13_baseline_sheet.png` — rendered against the Phase 0 failing baseline at `/srv/woodblock-share/chuck-clean-outputs/2026-05-17_v3-audit-thorough-main`. Row 2 shows the green/gold dominance audit complaint clearly. Rows 3-4 are honest placeholders because the v13 run never dumped plates/alphas — that absence is itself diagnostic.
- `sample_synthetic_sheet.png` — rendered against an in-memory 27-pull/27-plate/27-alpha synthetic plan to exercise every code path.

## pytest output (last 30 lines)

```
test_acceptance_harness.py::test_loads_reference_examples_returns_8_proofs PASSED [  7%]
test_acceptance_harness.py::test_renders_4_row_sheet_correct_dimensions PASSED [ 14%]
test_acceptance_harness.py::test_handles_missing_proof_checkpoints_gracefully PASSED [ 21%]
test_acceptance_harness.py::test_handles_fewer_than_27_plates PASSED     [ 28%]
test_acceptance_harness.py::test_normalizes_alpha_maps_to_visible_range PASSED [ 35%]
test_acceptance_harness.py::test_cli_invocation_smoke PASSED             [ 42%]
test_acceptance_harness.py::test_runs_on_synthetic_plan PASSED           [ 50%]
test_acceptance_harness.py::test_runs_on_v13_failing_baseline PASSED     [ 57%]
test_acceptance_harness.py::test_evenly_spaced_indices_small_n PASSED    [ 64%]
test_acceptance_harness.py::test_render_raises_for_missing_plan_dir PASSED [ 71%]
test_acceptance_harness.py::test_result_is_serializable PASSED           [ 78%]
test_acceptance_harness.py::test_cli_in_process PASSED                   [ 85%]
test_acceptance_harness.py::test_cli_reports_missing_plan_dir PASSED     [ 92%]
test_acceptance_harness.py::test_performance_under_10s PASSED            [100%]

Name                                       Stmts   Miss  Cover
--------------------------------------------------------------
acceptance_harness/__init__.py                 5      0   100%
acceptance_harness/__main__.py                 2      2     0%
acceptance_harness/acceptance_harness.py     186      8    96%
acceptance_harness/acceptance_result.py       24      0   100%
acceptance_harness/cli.py                     41      1    98%
acceptance_harness/example_loader.py          51      5    90%
--------------------------------------------------------------
TOTAL                                        309     16    95%
============================== 14 passed in 3.34s ==============================
```

8 required tests + 6 supporting tests, all pass. Coverage 95% (target was >80%).

## Performance benchmarks (Phase 1 budget: < 10s)

| Scenario | Wall time |
|---|---|
| v13 baseline (12 pulls, no plates/alphas) | **0.616 s** |
| Synthetic 27-pull / 27-plate / 27-alpha plan | **0.269 s** |
| Sparse 3-pull plan | **0.05 s** |
| Full pytest suite (14 tests) | **3.34 s** |

All well under budget.

## Sample sheet path

`/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v4-build/example-harness/sample_v13_baseline_sheet.png`

Visual verification (via the Read tool on the PNG during the build): rendered sheet is readable — Hokusai 8-cell reference row at top, current proof progression in row 2 clearly shows the v13 green/gold wash audit complaint, rows 3 and 4 surface the absent-plates / absent-alphas warnings as labeled placeholders.

## Integration contract (for downstream MCP tool)

A future `chuck_mcp.tools.render_acceptance_sheet` MCP tool would call into this module like this:

```python
from pathlib import Path
from acceptance_harness import render_acceptance_sheet

result = render_acceptance_sheet(
    plan_output_dir=Path("/srv/woodblock-share/chuck-clean-outputs/<run_id>"),
    reference_examples_dir=Path("/srv/woodblock-share/Examples"),
    output_path=Path("/srv/woodblock-share/chuck-acceptance-sheets/<run_id>.png"),
)

return {
    "sheet_url": f"https://chuck.reidsurmeier.wtf/acceptance/{result.sheet_path.name}",
    "proof_progression_score": result.proof_progression_score,
    "plate_metrics": [pm.__dict__ for pm in result.plate_metrics],
    "warnings": result.warnings,
    "human_eyeball_required": True,  # ALWAYS — audit rule
}
```

### Plan-dir contract (what the solver must produce for the harness to be useful)

| Path | Required? | Purpose |
|---|---|---|
| `cumulative_pull_NN.png` (1..N) | YES | row 2: cumulative proof states |
| `final_composite.png` | recommended | enables real `plate_not_composite` proxy in row 3 metrics |
| `plates/block_NN.preview.png` (1..N) | recommended | row 3: mirrored plate previews |
| `alpha_masks/alpha_NN.png` (1..N) | optional | row 4: per-plane alpha debug snapshots |
| `target.png` | informational | not used by harness directly |

Fallback patterns also supported: `pull_NN.png`, `proof_NN.png`, `plate_NN.png`, `block_NN.png`, `alpha_mask_NN.png`, `plane_NN.png`. Case-insensitive.

### Return-type contract

`AcceptanceSheetResult` is JSON-serializable via `.to_dict()` and contains:
- `sheet_path: Path`
- `reference_examples_used: list[str]` (filenames)
- `proof_checkpoints_rendered: list[int]` (pull indices used for row 2)
- `plate_count_rendered: int` (≤ 8)
- `alpha_count_rendered: int` (≤ 8)
- `proof_progression_score: float` (mean per-step pixel dissimilarity; healthy > 0.05)
- `plate_metrics: list[PlateMetric]` with `plate_index`, `coverage_fraction`, `plate_not_composite_score`
- `human_eyeball_required: bool` — always True
- `warnings: list[str]` including a `render_seconds=X.XXX` timing entry

## Known limitations

1. **Reference row is Hokusai-only.** Only `Woodblock-print-process.png` is sliced into the row 1 grid. The other four examples (Chuck Close screenshots, etc.) are merely listed in `reference_examples_used`. A future iteration could rotate the reference row by example family.
2. **`plate_not_composite_score` is a cosine-similarity proxy**, not the real validator from `backend/services/v23/validators`. It produces sensible relative numbers (jigsaw plates >> residual-composite plates) but should not gate sign-off. Integrate the real validator before V1.0 ship.
3. **`proof_progression_score` is a pixel-diff proxy**, not the design-doc validator. It catches the obvious "solver re-renders the same proof" failure mode; it does not catch the more interesting "proof progresses but in the wrong direction" failure.
4. **No font fallback.** Uses `PIL.ImageFont.load_default()`. Labels look bitmappy on a 4K monitor. A TTF font in `assets/` would be a nice polish later.
5. **Row 4 alpha normalization is per-plane**, so absolute alpha values cannot be compared between plates by eye. Trade-off: visible structure > absolute scale, because the spec says alphas are debug-only.
6. **No SVG rasterization.** If the solver writes only `block_NN.svg` (no PNG preview), row 3 will be empty placeholders. The next agent should add CairoSVG / svglib rasterization, or insist the solver always emits a `.preview.png` next to each `.svg`.
7. **Pulled `WebP-as-PNG` quirk:** `Woodblock-print-process.png` is actually a WebP (per `file(1)`). PIL opens it fine, but if the example folder ever swaps in a true-PNG version, the test still passes — that's intentional.

## What the next agent should integrate

In priority order:

1. **Replace the proxy validators with the real ones from `backend/services/v23/validators`.** Plug `plate_not_composite`, `proof_progression_score`, `role_purity_score`, `jigsaw_separation_score` directly into `AcceptanceSheetResult` and stop computing the cheap proxies.
2. **Wire the harness into the MCP server.** Add a `render_acceptance_sheet` MCP tool in `backend/mcp/v23_server.py` that takes a `plan_id` and returns the sheet URL + metrics dict. The frontend at `chuck.reidsurmeier.wtf` should embed it in the "Verification" panel.
3. **Make the solver dump per-plane alpha snapshots and `block_NN.preview.png`** during every run. Without these, rows 3 and 4 are useless. Suggested location: `<plan_dir>/plates/` and `<plan_dir>/alpha_masks/`.
4. **Wire a human-verdict capture** (thumbs-up / thumbs-down + free-text "what looks like slop?") and store it in `meta/acceptance_verdict.json` alongside the sheet. That verdict is the *gating signal* per the audit rule.
5. **Add SVG rasterization fallback** (CairoSVG → PNG at 256×256) so plates can be reviewed even when only the carving SVG is written.
6. **Add at least one Chuck Close reference example to row 1.** Even a single 8-tile montage extracted from `Screenshot 2026-05-14 at 6.13.05 PM.png` would broaden the eyeball anchor beyond Hokusai's wave.

## CLI usage

```bash
cd /home/reidsurmeier/src/chuck-mcp-layering-lab/research/v4-build/example-harness
python -m acceptance_harness /srv/woodblock-share/chuck-clean-outputs/2026-05-17_v3-audit-thorough-main \
    --output /tmp/v13_baseline_sheet.png

# JSON output for programmatic consumption:
python -m acceptance_harness <plan_dir> --json | jq '.proof_progression_score'
```

## V1.0 readiness

Phase 1 (this deliverable) is V1.0-ready for the *digital* acceptance loop: the sheet renders, the contract is stable, the tests are real, and the v13 failing baseline is correctly characterized as "row 2 is muddy, rows 3-4 missing = solver didn't even attempt jigsaw plates." V1.0 ship requires Phases 2-3 from the audit doc (production-shaped solver + plate organization in the objective); the harness is the gate they will be measured against.
